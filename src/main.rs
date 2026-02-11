mod api;
mod backend;
mod config;
mod types;

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use blocknet_pow_spec::POW_HEADER_BASE_LEN;
use crossbeam_channel::{after, unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};

use api::ApiClient;
use backend::cpu::CpuBackend;
use backend::nvidia::NvidiaBackend;
use backend::{BackendEvent, MiningSolution, MiningWork, PowBackend};
use config::{BackendKind, BenchKind, Config};
use types::{
    decode_hex, parse_target, set_block_nonce, template_difficulty, template_height,
    BlockTemplateResponse,
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const TEMPLATE_RETRY_DELAY: Duration = Duration::from_secs(2);
const MIN_EVENT_WAIT: Duration = Duration::from_millis(1);
const HASH_POLL_INTERVAL: Duration = Duration::from_millis(200);

struct Stats {
    started_at: Instant,
    hashes: AtomicU64,
    templates: AtomicU64,
    submitted: AtomicU64,
    accepted: AtomicU64,
}

impl Stats {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
            hashes: AtomicU64::new(0),
            templates: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
        }
    }

    fn print(&self) {
        let elapsed = self.started_at.elapsed().as_secs_f64().max(0.001);
        let hashes = self.hashes.load(Ordering::Relaxed);
        let templates = self.templates.load(Ordering::Relaxed);
        let submitted = self.submitted.load(Ordering::Relaxed);
        let accepted = self.accepted.load(Ordering::Relaxed);
        let hps = hashes as f64 / elapsed;

        println!(
            "[stats] {:.1}s elapsed | {} hashes | {} | templates={} submitted={} accepted={}",
            elapsed,
            hashes,
            format_hashrate(hps),
            templates,
            submitted,
            accepted,
        );
    }
}

struct BackendSlot {
    backend: Box<dyn PowBackend>,
    lane_offset: u64,
    lanes: u64,
}

enum TipEvent {
    NewBlock,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("fatal: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cfg = Config::parse()?;

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            shutdown.store(true, Ordering::SeqCst);
        })?;
    }

    if cfg.bench {
        return run_benchmark(&cfg, &shutdown);
    }

    let token = cfg
        .token
        .clone()
        .ok_or_else(|| anyhow!("missing API token in mining mode"))?;
    let client = ApiClient::new(cfg.api_url.clone(), token, REQUEST_TIMEOUT)?;

    let backend_instances = build_backend_instances(&cfg);
    let (mut backends, backend_events) = activate_backends(backend_instances)?;
    let total_lanes = total_lanes(&backends);

    println!(
        "starting bnminer | backends={} | cpu_threads={} (~{}GB RAM for CPU lanes) | lanes={} | api={} | sse={}",
        backend_names(&backends),
        cfg.threads,
        cfg.threads * 2,
        total_lanes,
        cfg.api_url,
        if cfg.sse_enabled { "on" } else { "off" }
    );

    let tip_events = if cfg.sse_enabled {
        Some(spawn_tip_listener(client.clone(), Arc::clone(&shutdown)))
    } else {
        None
    };

    let result = run_mining_loop(
        &cfg,
        &client,
        &shutdown,
        &mut backends,
        &backend_events,
        tip_events.as_ref(),
    );

    stop_backend_slots(&mut backends);
    result
}

fn run_mining_loop(
    cfg: &Config,
    client: &ApiClient,
    shutdown: &AtomicBool,
    backends: &mut [BackendSlot],
    backend_events: &Receiver<BackendEvent>,
    tip_events: Option<&Receiver<TipEvent>>,
) -> Result<()> {
    let stats = Stats::new();
    let mut nonce_cursor = cfg.start_nonce;
    let mut epoch = 0u64;
    let mut last_stats_print = Instant::now();
    let total_lanes = total_lanes(backends);

    while !shutdown.load(Ordering::Relaxed) {
        let template = match fetch_template_with_retry(client, shutdown) {
            Some(t) => t,
            None => break,
        };

        let header_base = match decode_hex(&template.header_base, "header_base") {
            Ok(v) => v,
            Err(err) => {
                eprintln!("template decode error: {err:#}");
                continue;
            }
        };

        if header_base.len() != POW_HEADER_BASE_LEN {
            eprintln!(
                "template header_base length mismatch: expected {} bytes, got {}",
                POW_HEADER_BASE_LEN,
                header_base.len()
            );
            continue;
        }

        let target = match parse_target(&template.target) {
            Ok(t) => t,
            Err(err) => {
                eprintln!("target parse error: {err:#}");
                continue;
            }
        };

        let height = template_height(&template.block)
            .map(|h| h.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let difficulty = template_difficulty(&template.block)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        epoch = epoch.wrapping_add(1).max(1);
        let stop_at = Instant::now() + cfg.refresh_interval;

        stats.templates.fetch_add(1, Ordering::Relaxed);

        println!(
            "[template] height={} difficulty={} epoch={} nonce_seed={} refresh={}s",
            height,
            difficulty,
            epoch,
            nonce_cursor,
            cfg.refresh_interval.as_secs(),
        );

        distribute_work(
            backends,
            epoch,
            Arc::from(header_base),
            target,
            nonce_cursor,
            stop_at,
        )?;

        let round_start = Instant::now();
        let mut solved: Option<MiningSolution> = None;
        let mut stale_tip_event = false;
        let mut round_hashes = 0u64;

        while !shutdown.load(Ordering::Relaxed)
            && Instant::now() < stop_at
            && solved.is_none()
            && !stale_tip_event
        {
            collect_backend_hashes(backends, epoch, Some(&stats), &mut round_hashes);

            if last_stats_print.elapsed() >= cfg.stats_interval {
                stats.print();
                last_stats_print = Instant::now();
            }

            let wait_for = next_event_wait(stop_at, last_stats_print, cfg.stats_interval);

            if let Some(tip_rx) = tip_events {
                crossbeam_channel::select! {
                    recv(backend_events) -> event => {
                        let event = event.map_err(|_| anyhow!("backend event channel closed"))?;
                        handle_mining_backend_event(
                            event,
                            epoch,
                            &mut solved,
                        )?;
                    }
                    recv(tip_rx) -> tip => {
                        if tip.is_ok() {
                            stale_tip_event = true;
                        }
                    }
                    recv(after(wait_for)) -> _ => {}
                }
            } else {
                crossbeam_channel::select! {
                    recv(backend_events) -> event => {
                        let event = event.map_err(|_| anyhow!("backend event channel closed"))?;
                        handle_mining_backend_event(
                            event,
                            epoch,
                            &mut solved,
                        )?;
                    }
                    recv(after(wait_for)) -> _ => {}
                }
            }
        }

        drain_mining_backend_events(backend_events, epoch, &mut solved)?;
        collect_backend_hashes(backends, epoch, Some(&stats), &mut round_hashes);
        if consume_tip_events(tip_events) {
            stale_tip_event = true;
        }

        if let Some(solution) = solved {
            println!(
                "[solution] backend={} nonce={} elapsed={:.2}s",
                solution.backend,
                solution.nonce,
                round_start.elapsed().as_secs_f64(),
            );

            let template_id = template.template_id.clone();
            let mut solved_block = template.block;
            if let Err(err) = set_block_nonce(&mut solved_block, solution.nonce) {
                eprintln!("failed to set nonce on solved block: {err:#}");
                nonce_cursor = advance_nonce_cursor(nonce_cursor, total_lanes, round_hashes);
                continue;
            }

            stats.submitted.fetch_add(1, Ordering::Relaxed);

            match client.submit_block(&solved_block, template_id.as_deref(), solution.nonce) {
                Ok(resp) => {
                    if resp.accepted {
                        stats.accepted.fetch_add(1, Ordering::Relaxed);
                        println!(
                            "[submit] accepted=true height={} hash={}",
                            resp.height
                                .map(|h| h.to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                            resp.hash.unwrap_or_else(|| "unknown".to_string())
                        );
                    } else {
                        println!("[submit] accepted=false");
                    }
                }
                Err(err) => {
                    eprintln!("submit failed: {err:#}");
                }
            }
        } else if stale_tip_event {
            println!("[template] stale tip event received; refreshing template immediately");
        } else {
            println!("[template] no solution before refresh or shutdown");
        }

        nonce_cursor = advance_nonce_cursor(nonce_cursor, total_lanes, round_hashes);

        if last_stats_print.elapsed() >= cfg.stats_interval {
            stats.print();
            last_stats_print = Instant::now();
        }
    }

    stats.print();
    println!("bnminer stopped");
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRun {
    round: u32,
    hashes: u64,
    elapsed_secs: f64,
    hps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchReport {
    bench_kind: String,
    backends: Vec<String>,
    total_lanes: u64,
    cpu_threads: usize,
    bench_secs: u64,
    rounds: u32,
    avg_hps: f64,
    median_hps: f64,
    min_hps: f64,
    max_hps: f64,
    runs: Vec<BenchRun>,
}

fn run_benchmark(cfg: &Config, shutdown: &AtomicBool) -> Result<()> {
    let instances = build_backend_instances(cfg);

    match cfg.bench_kind {
        BenchKind::Kernel => run_kernel_benchmark(cfg, shutdown, instances),
        BenchKind::Backend => run_worker_benchmark(cfg, shutdown, instances, false),
        BenchKind::EndToEnd => run_worker_benchmark(cfg, shutdown, instances, true),
    }
}

fn run_kernel_benchmark(
    cfg: &Config,
    shutdown: &AtomicBool,
    backends: Vec<Box<dyn PowBackend>>,
) -> Result<()> {
    let mut iter = backends.into_iter();
    let backend = iter
        .next()
        .ok_or_else(|| anyhow!("kernel benchmark requires at least one backend"))?;

    if iter.next().is_some() {
        bail!("kernel benchmark requires exactly one backend");
    }

    println!(
        "benchmark mode | kind=kernel | backend={} | rounds={} | seconds_per_round={}",
        backend.name(),
        cfg.bench_rounds,
        cfg.bench_secs
    );

    let mut runs = Vec::with_capacity(cfg.bench_rounds as usize);
    for round in 0..cfg.bench_rounds {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let round_start = Instant::now();
        let hashes = backend.kernel_bench(cfg.bench_secs, shutdown)?;
        let elapsed = round_start.elapsed().as_secs_f64().max(0.001);
        let hps = hashes as f64 / elapsed;

        println!(
            "[bench] round {}/{} | hashes={} | elapsed={:.2}s | {}",
            round + 1,
            cfg.bench_rounds,
            hashes,
            elapsed,
            format_hashrate(hps),
        );

        runs.push(BenchRun {
            round: round + 1,
            hashes,
            elapsed_secs: elapsed,
            hps,
        });
    }

    summarize_benchmark(
        cfg,
        BenchReport {
            bench_kind: "kernel".to_string(),
            backends: vec![backend.name().to_string()],
            total_lanes: backend.lanes() as u64,
            cpu_threads: cfg.threads,
            bench_secs: cfg.bench_secs,
            rounds: runs.len() as u32,
            avg_hps: 0.0,
            median_hps: 0.0,
            min_hps: 0.0,
            max_hps: 0.0,
            runs,
        },
    )
}

fn run_worker_benchmark(
    cfg: &Config,
    shutdown: &AtomicBool,
    instances: Vec<Box<dyn PowBackend>>,
    restart_each_round: bool,
) -> Result<()> {
    let (mut backends, backend_events) = activate_backends(instances)?;
    let bench_kind = if restart_each_round {
        "end_to_end"
    } else {
        "backend"
    };

    println!(
        "benchmark mode | kind={} | backends={} | lanes={} | rounds={} | seconds_per_round={}",
        bench_kind,
        backend_names(&backends),
        total_lanes(&backends),
        cfg.bench_rounds,
        cfg.bench_secs
    );

    if restart_each_round {
        stop_backend_slots(&mut backends);
    }

    let result = run_worker_benchmark_inner(
        cfg,
        shutdown,
        &mut backends,
        &backend_events,
        restart_each_round,
    );
    stop_backend_slots(&mut backends);
    result
}

fn run_worker_benchmark_inner(
    cfg: &Config,
    shutdown: &AtomicBool,
    backends: &mut [BackendSlot],
    backend_events: &Receiver<BackendEvent>,
    restart_each_round: bool,
) -> Result<()> {
    let impossible_target = [0u8; 32];
    let mut runs = Vec::with_capacity(cfg.bench_rounds as usize);
    let mut epoch = 0u64;
    let mut nonce_cursor = cfg.start_nonce;

    for round in 0..cfg.bench_rounds {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        if restart_each_round {
            start_backend_slots(backends)?;
        }

        epoch = epoch.wrapping_add(1).max(1);
        let round_start = Instant::now();
        let stop_at = round_start + Duration::from_secs(cfg.bench_secs);
        let header_base = benchmark_header_base(round);
        let nonce_seed = nonce_cursor;

        distribute_work(
            backends,
            epoch,
            header_base,
            impossible_target,
            nonce_seed,
            stop_at,
        )?;

        let mut round_hashes = 0u64;
        while Instant::now() < stop_at && !shutdown.load(Ordering::Relaxed) {
            collect_backend_hashes(backends, epoch, None, &mut round_hashes);
            let wait_for = stop_at
                .saturating_duration_since(Instant::now())
                .min(HASH_POLL_INTERVAL)
                .max(MIN_EVENT_WAIT);

            crossbeam_channel::select! {
                recv(backend_events) -> event => {
                    let event = event.map_err(|_| anyhow!("backend event channel closed"))?;
                    handle_benchmark_backend_event(event, epoch)?;
                }
                recv(after(wait_for)) -> _ => {}
            }
        }

        collect_backend_hashes(backends, epoch, None, &mut round_hashes);
        drain_benchmark_backend_events(backend_events, epoch)?;

        let elapsed = round_start.elapsed().as_secs_f64().max(0.001);
        let hps = round_hashes as f64 / elapsed;

        println!(
            "[bench] round {}/{} | hashes={} | elapsed={:.2}s | {}",
            round + 1,
            cfg.bench_rounds,
            round_hashes,
            elapsed,
            format_hashrate(hps),
        );

        runs.push(BenchRun {
            round: round + 1,
            hashes: round_hashes,
            elapsed_secs: elapsed,
            hps,
        });

        nonce_cursor = advance_nonce_cursor(nonce_cursor, total_lanes(backends), round_hashes);

        if restart_each_round {
            stop_backend_slots(backends);
        }
    }

    summarize_benchmark(
        cfg,
        BenchReport {
            bench_kind: if restart_each_round {
                "end_to_end".to_string()
            } else {
                "backend".to_string()
            },
            backends: backend_name_list(backends),
            total_lanes: total_lanes(backends),
            cpu_threads: cfg.threads,
            bench_secs: cfg.bench_secs,
            rounds: runs.len() as u32,
            avg_hps: 0.0,
            median_hps: 0.0,
            min_hps: 0.0,
            max_hps: 0.0,
            runs,
        },
    )
}

fn summarize_benchmark(cfg: &Config, mut report: BenchReport) -> Result<()> {
    if report.runs.is_empty() {
        println!("benchmark aborted before first round");
        return Ok(());
    }

    let mut sorted_hps: Vec<f64> = report.runs.iter().map(|r| r.hps).collect();
    sorted_hps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    report.avg_hps = sorted_hps.iter().sum::<f64>() / sorted_hps.len() as f64;
    report.median_hps = median(&sorted_hps);
    report.min_hps = *sorted_hps.first().unwrap_or(&0.0);
    report.max_hps = *sorted_hps.last().unwrap_or(&0.0);

    println!(
        "[bench] summary | avg={} | median={} | min={} | max={}",
        format_hashrate(report.avg_hps),
        format_hashrate(report.median_hps),
        format_hashrate(report.min_hps),
        format_hashrate(report.max_hps),
    );

    if let Some(path) = &cfg.bench_baseline {
        let baseline_text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read baseline file {}", path.display()))?;
        let baseline: BenchReport = serde_json::from_str(&baseline_text)
            .with_context(|| format!("failed to parse baseline JSON {}", path.display()))?;
        if baseline.avg_hps > 0.0 {
            let delta_pct = ((report.avg_hps - baseline.avg_hps) / baseline.avg_hps) * 100.0;
            println!(
                "[bench] baseline compare | baseline_avg={} | delta={:+.2}%",
                format_hashrate(baseline.avg_hps),
                delta_pct
            );
        }
    }

    if let Some(path) = &cfg.bench_output {
        let json = serde_json::to_string_pretty(&report)
            .context("failed to serialize benchmark report")?;
        std::fs::write(path, json)
            .with_context(|| format!("failed to write benchmark report {}", path.display()))?;
        println!("[bench] wrote report to {}", path.display());
    }

    Ok(())
}

fn benchmark_header_base(round: u32) -> Arc<[u8]> {
    let mut data = [0u8; POW_HEADER_BASE_LEN];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = (i as u8)
            .wrapping_mul(37)
            .wrapping_add(11)
            .wrapping_add((round % 251) as u8);
    }
    Arc::from(data.to_vec())
}

fn build_backend_instances(cfg: &Config) -> Vec<Box<dyn PowBackend>> {
    cfg.backends
        .iter()
        .map(|backend_kind| match backend_kind {
            BackendKind::Cpu => Box::new(CpuBackend::new(cfg.threads)) as Box<dyn PowBackend>,
            BackendKind::Nvidia => Box::new(NvidiaBackend::new()) as Box<dyn PowBackend>,
        })
        .collect()
}

fn activate_backends(
    mut backends: Vec<Box<dyn PowBackend>>,
) -> Result<(Vec<BackendSlot>, Receiver<BackendEvent>)> {
    let mut active = Vec::new();
    let (event_tx, event_rx) = unbounded::<BackendEvent>();

    for mut backend in backends.drain(..) {
        let backend_name = backend.name();
        backend.set_event_sink(event_tx.clone());
        match backend.start() {
            Ok(()) => {
                let lanes = backend.lanes() as u64;
                if lanes == 0 {
                    eprintln!("[backend] skipping {}: reported zero lanes", backend_name);
                    backend.stop();
                    continue;
                }
                active.push(BackendSlot {
                    backend,
                    lane_offset: 0,
                    lanes,
                });
            }
            Err(err) => {
                eprintln!("[backend] {} unavailable: {err:#}", backend_name);
            }
        }
    }

    if active.is_empty() {
        bail!("no mining backend could be started");
    }

    let mut lane_offset = 0u64;
    for slot in &mut active {
        slot.lane_offset = lane_offset;
        lane_offset = lane_offset.wrapping_add(slot.lanes);
    }

    Ok((active, event_rx))
}

fn start_backend_slots(backends: &mut [BackendSlot]) -> Result<()> {
    for slot in backends {
        slot.backend
            .start()
            .with_context(|| format!("failed to start backend {}", slot.backend.name()))?;
    }
    Ok(())
}

fn stop_backend_slots(backends: &mut [BackendSlot]) {
    for slot in backends {
        slot.backend.stop();
    }
}

fn distribute_work(
    backends: &[BackendSlot],
    epoch: u64,
    header_base: Arc<[u8]>,
    target: [u8; 32],
    start_nonce: u64,
    stop_at: Instant,
) -> Result<()> {
    let stride = total_lanes(backends);
    for slot in backends {
        let work = MiningWork {
            epoch,
            header_base: Arc::clone(&header_base),
            target,
            start_nonce,
            lane_offset: slot.lane_offset,
            global_stride: stride,
            stop_at,
        };
        slot.backend
            .set_work(work)
            .with_context(|| format!("failed to set work for backend {}", slot.backend.name()))?;
    }
    Ok(())
}

fn next_event_wait(
    stop_at: Instant,
    last_stats_print: Instant,
    stats_interval: Duration,
) -> Duration {
    let now = Instant::now();
    let until_stop = stop_at.saturating_duration_since(now);
    let next_stats_at = last_stats_print + stats_interval;
    let until_stats = next_stats_at.saturating_duration_since(now);
    until_stop
        .min(until_stats)
        .min(HASH_POLL_INTERVAL)
        .max(MIN_EVENT_WAIT)
}

fn advance_nonce_cursor(nonce_cursor: u64, lanes: u64, observed_hashes: u64) -> u64 {
    let stride = lanes.max(1);
    // Add one iteration per lane as a safety margin for hashes that complete
    // immediately after the final counter poll of an epoch.
    let safe_hashes = observed_hashes.saturating_add(stride).max(1);
    nonce_cursor.wrapping_add(stride.saturating_mul(safe_hashes))
}

fn collect_backend_hashes(
    backends: &[BackendSlot],
    epoch: u64,
    stats: Option<&Stats>,
    round_hashes: &mut u64,
) {
    let mut collected = 0u64;
    for slot in backends {
        collected = collected.saturating_add(slot.backend.take_hashes(epoch));
    }

    if collected == 0 {
        return;
    }

    if let Some(stats) = stats {
        stats.hashes.fetch_add(collected, Ordering::Relaxed);
    }
    *round_hashes = round_hashes.saturating_add(collected);
}

fn handle_mining_backend_event(
    event: BackendEvent,
    epoch: u64,
    solved: &mut Option<MiningSolution>,
) -> Result<()> {
    match event {
        BackendEvent::Solution(solution) => {
            if solution.epoch == epoch {
                *solved = Some(solution);
            }
        }
        BackendEvent::Error { backend, message } => {
            bail!("backend '{backend}' reported error: {message}");
        }
    }
    Ok(())
}

fn drain_mining_backend_events(
    backend_events: &Receiver<BackendEvent>,
    epoch: u64,
    solved: &mut Option<MiningSolution>,
) -> Result<()> {
    while let Ok(event) = backend_events.try_recv() {
        handle_mining_backend_event(event, epoch, solved)?;
    }
    Ok(())
}

fn handle_benchmark_backend_event(event: BackendEvent, epoch: u64) -> Result<()> {
    match event {
        BackendEvent::Solution(solution) => {
            if solution.epoch == epoch {
                println!(
                    "[bench] unexpected solution found by {} at nonce={}",
                    solution.backend, solution.nonce
                );
            }
        }
        BackendEvent::Error { backend, message } => {
            bail!("backend '{backend}' reported error: {message}");
        }
    }
    Ok(())
}

fn drain_benchmark_backend_events(
    backend_events: &Receiver<BackendEvent>,
    epoch: u64,
) -> Result<()> {
    while let Ok(event) = backend_events.try_recv() {
        handle_benchmark_backend_event(event, epoch)?;
    }
    Ok(())
}

fn consume_tip_events(event_rx: Option<&Receiver<TipEvent>>) -> bool {
    let Some(event_rx) = event_rx else {
        return false;
    };

    let mut stale = false;
    while let Ok(event) = event_rx.try_recv() {
        if matches!(event, TipEvent::NewBlock) {
            stale = true;
        }
    }
    stale
}

fn spawn_tip_listener(client: ApiClient, shutdown: Arc<AtomicBool>) -> Receiver<TipEvent> {
    let (tx, rx) = unbounded();

    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            match client.open_events_stream() {
                Ok(resp) => {
                    if let Err(err) = stream_tip_events(resp, &tx, &shutdown) {
                        if !shutdown.load(Ordering::Relaxed) {
                            eprintln!("events stream dropped: {err:#}");
                        }
                    }
                }
                Err(err) => {
                    if !shutdown.load(Ordering::Relaxed) {
                        eprintln!("failed to open events stream: {err:#}");
                    }
                }
            }

            if !shutdown.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
            }
        }
    });

    rx
}

fn stream_tip_events(
    resp: reqwest::blocking::Response,
    tx: &Sender<TipEvent>,
    shutdown: &AtomicBool,
) -> Result<()> {
    let mut event_name = String::new();
    let reader = BufReader::new(resp);

    for line_result in reader.lines() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let line = line_result.context("failed reading SSE event stream")?;

        if let Some(name) = line.strip_prefix("event:") {
            event_name = name.trim().to_string();
            continue;
        }

        if line.is_empty() {
            event_name.clear();
            continue;
        }

        if event_name == "new_block" && line.starts_with("data:") {
            let _ = tx.send(TipEvent::NewBlock);
        }
    }

    Ok(())
}

fn fetch_template_with_retry(
    client: &ApiClient,
    shutdown: &AtomicBool,
) -> Option<BlockTemplateResponse> {
    while !shutdown.load(Ordering::Relaxed) {
        match client.get_block_template() {
            Ok(template) => return Some(template),
            Err(err) => {
                eprintln!("blocktemplate request failed: {err:#}");
                thread::sleep(TEMPLATE_RETRY_DELAY);
            }
        }
    }

    None
}

fn backend_name_list(backends: &[BackendSlot]) -> Vec<String> {
    backends
        .iter()
        .map(|slot| slot.backend.name().to_string())
        .collect()
}

fn backend_names(backends: &[BackendSlot]) -> String {
    backend_name_list(backends).join(",")
}

fn total_lanes(backends: &[BackendSlot]) -> u64 {
    backends.iter().map(|slot| slot.lanes).sum::<u64>().max(1)
}

fn format_hashrate(hps: f64) -> String {
    if hps >= 1_000_000_000.0 {
        return format!("{:.3} GH/s", hps / 1_000_000_000.0);
    }
    if hps >= 1_000_000.0 {
        return format!("{:.3} MH/s", hps / 1_000_000.0);
    }
    if hps >= 1_000.0 {
        return format!("{:.3} KH/s", hps / 1_000.0);
    }
    format!("{hps:.3} H/s")
}

fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn median_handles_even_and_odd() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0, 5.0]), 3.0);
        assert_eq!(median(&[1.0, 3.0, 5.0, 7.0]), 4.0);
    }

    #[test]
    fn format_hashrate_units() {
        assert_eq!(format_hashrate(5.0), "5.000 H/s");
        assert_eq!(format_hashrate(5_000.0), "5.000 KH/s");
        assert_eq!(format_hashrate(5_000_000.0), "5.000 MH/s");
    }

    #[test]
    fn advance_nonce_cursor_uses_stride_and_safety_margin() {
        let seed = 100u64;
        let next = advance_nonce_cursor(seed, 8, 50);
        assert_eq!(next, 564);
    }

    #[test]
    fn advance_nonce_cursor_prevents_overlap_for_unbalanced_lanes() {
        let lanes = 4u64;
        let seed = 17u64;
        let lane_iters = [9u64, 2, 5, 7];
        let total_hashes = lane_iters.iter().sum::<u64>();

        let mut searched = HashSet::new();
        for (lane, iters) in lane_iters.iter().enumerate() {
            for i in 0..*iters {
                let nonce = seed
                    .wrapping_add(lane as u64)
                    .wrapping_add(i.saturating_mul(lanes));
                searched.insert(nonce);
            }
        }

        let next_seed = advance_nonce_cursor(seed, lanes, total_hashes);
        for lane in 0..lanes {
            for i in 0..lane_iters.iter().copied().max().unwrap_or(0) {
                let nonce = next_seed
                    .wrapping_add(lane)
                    .wrapping_add(i.saturating_mul(lanes));
                assert!(
                    !searched.contains(&nonce),
                    "nonce {nonce} should not overlap with previously searched space"
                );
            }
        }
    }

    #[test]
    fn stale_solution_is_ignored_for_current_epoch() {
        let mut solved = None;

        handle_mining_backend_event(
            BackendEvent::Solution(MiningSolution {
                epoch: 41,
                nonce: 9,
                backend: "cpu".to_string(),
            }),
            42,
            &mut solved,
        )
        .expect("stale solution handling should succeed");

        assert!(solved.is_none());
    }

    #[test]
    fn benchmark_ignores_stale_solution_events() {
        handle_benchmark_backend_event(
            BackendEvent::Solution(MiningSolution {
                epoch: 99,
                nonce: 123,
                backend: "cpu".to_string(),
            }),
            100,
        )
        .expect("stale benchmark solution event should be ignored");
    }
}
