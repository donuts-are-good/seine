use std::collections::VecDeque;
use std::convert::TryFrom;
use std::ffi::{c_char, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use blake2::{
    digest::{self, Digest, VariableOutput},
    Blake2b512, Blake2bVar,
};
use blocknet_pow_spec::{pow_params, CPU_LANE_MEMORY_BYTES, POW_HEADER_BASE_LEN, POW_OUTPUT_LEN};
use crossbeam_channel::{
    bounded, Receiver, RecvTimeoutError, SendTimeoutError, Sender, TryRecvError, TrySendError,
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    },
    nvrtc::{result as nvrtc_result, sys as nvrtc_sys, Ptx},
};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use crate::backend::NonceChunk;
use crate::backend::{
    AssignmentSemantics, BackendCallStatus, BackendCapabilities, BackendEvent,
    BackendExecutionModel, BackendInstanceId, BackendTelemetry, BenchBackend, DeadlineSupport,
    MiningSolution, PowBackend, PreemptionGranularity, WorkAssignment, WorkTemplate,
};
use crate::types::hash_meets_target;

const BACKEND_NAME: &str = "nvidia";
const CUDA_KERNEL_SRC: &str = include_str!("nvidia_kernel.cu");
const MAX_RECOMMENDED_LANES: usize = 16;
const ASSIGN_CHANNEL_CAPACITY: usize = 256;
const CONTROL_CHANNEL_CAPACITY: usize = 32;
const EVENT_SEND_WAIT: Duration = Duration::from_millis(5);
const KERNEL_THREADS: u32 = 32;
const DEFAULT_NVIDIA_MAX_RREGCOUNT: u32 = 224;
const NVIDIA_AUTOTUNE_SCHEMA_VERSION: u32 = 4;
const NVIDIA_AUTOTUNE_REGCAP_CANDIDATES: &[u32] = &[240, 224, 208, 192, 160];
const NVIDIA_AUTOTUNE_LOOP_UNROLL_CANDIDATES: &[bool] = &[false];
const DEFAULT_NVIDIA_AUTOTUNE_SAMPLES: u32 = 2;
const NVIDIA_MEMORY_RESERVE_MIB_FLOOR: u64 = 64;
const NVIDIA_MEMORY_RESERVE_RATIO_DENOM: u64 = 64;
const DEFAULT_DISPATCH_ITERS_PER_LANE: u64 = 1 << 20;
const DEFAULT_ALLOCATION_ITERS_PER_LANE: u64 = 1 << 21;
const DEFAULT_HASHES_PER_LAUNCH_PER_LANE: u32 = 2;
const ARGON2_VERSION_V13: u32 = 0x13;
const ARGON2_ALGORITHM_ID: u32 = 2; // Argon2id

#[derive(Debug, Clone)]
struct NvidiaDeviceInfo {
    index: u32,
    name: String,
    memory_total_mib: u64,
    memory_free_mib: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
struct NvidiaKernelTuning {
    max_rregcount: u32,
    block_loop_unroll: bool,
}

impl Default for NvidiaKernelTuning {
    fn default() -> Self {
        Self {
            max_rregcount: DEFAULT_NVIDIA_MAX_RREGCOUNT,
            block_loop_unroll: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct NvidiaAutotuneKey {
    device_name: String,
    memory_total_mib: u64,
    #[serde(default)]
    memory_budget_mib: u64,
    compute_cap_major: u32,
    compute_cap_minor: u32,
    m_cost_kib: u32,
    t_cost: u32,
    kernel_threads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NvidiaAutotuneRecord {
    key: NvidiaAutotuneKey,
    max_rregcount: u32,
    #[serde(default)]
    block_loop_unroll: bool,
    measured_hps: f64,
    autotune_secs: u64,
    #[serde(default)]
    autotune_samples: u32,
    timestamp_unix_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NvidiaAutotuneCache {
    schema_version: u32,
    records: Vec<NvidiaAutotuneRecord>,
}

struct NvidiaShared {
    event_sink: Arc<RwLock<Option<Sender<BackendEvent>>>>,
    dropped_events: AtomicU64,
    hashes: AtomicU64,
    active_lanes: AtomicU64,
    pending_work: AtomicU64,
    inflight_assignment_hashes: AtomicU64,
    inflight_assignment_started_at: Mutex<Option<Instant>>,
    completed_assignments: AtomicU64,
    completed_assignment_hashes: AtomicU64,
    completed_assignment_micros: AtomicU64,
    error_emitted: AtomicBool,
}

#[derive(Debug, Clone, Copy)]
pub struct NvidiaBackendTuningOptions {
    pub max_rregcount_override: Option<u32>,
    pub max_lanes_override: Option<usize>,
    pub autotune_samples: u32,
    pub dispatch_iters_per_lane: Option<u64>,
    pub allocation_iters_per_lane: Option<u64>,
    pub hashes_per_launch_per_lane: u32,
}

impl Default for NvidiaBackendTuningOptions {
    fn default() -> Self {
        Self {
            max_rregcount_override: None,
            max_lanes_override: None,
            autotune_samples: DEFAULT_NVIDIA_AUTOTUNE_SAMPLES,
            dispatch_iters_per_lane: None,
            allocation_iters_per_lane: None,
            hashes_per_launch_per_lane: DEFAULT_HASHES_PER_LAUNCH_PER_LANE,
        }
    }
}

impl NvidiaShared {
    fn new() -> Self {
        Self {
            event_sink: Arc::new(RwLock::new(None)),
            dropped_events: AtomicU64::new(0),
            hashes: AtomicU64::new(0),
            active_lanes: AtomicU64::new(0),
            pending_work: AtomicU64::new(0),
            inflight_assignment_hashes: AtomicU64::new(0),
            inflight_assignment_started_at: Mutex::new(None),
            completed_assignments: AtomicU64::new(0),
            completed_assignment_hashes: AtomicU64::new(0),
            completed_assignment_micros: AtomicU64::new(0),
            error_emitted: AtomicBool::new(false),
        }
    }
}

struct NvidiaWorker {
    assignment_tx: Sender<WorkerCommand>,
    control_tx: Sender<WorkerCommand>,
    handle: JoinHandle<()>,
}

enum WorkerCommand {
    Assign(WorkAssignment),
    AssignBatch(Vec<WorkAssignment>),
    Cancel(Sender<Result<()>>),
    Fence(Sender<Result<()>>),
    Stop,
}

struct ActiveAssignment {
    work: WorkAssignment,
    next_nonce: u64,
    remaining: u64,
    hashes_done: u64,
    started_at: Instant,
}

impl ActiveAssignment {
    fn new(work: WorkAssignment) -> Self {
        Self {
            next_nonce: work.nonce_chunk.start_nonce,
            remaining: work.nonce_chunk.nonce_count,
            hashes_done: 0,
            started_at: Instant::now(),
            work,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PendingControlKind {
    Cancel,
    Fence,
}

impl PendingControlKind {
    fn label(self) -> &'static str {
        match self {
            Self::Cancel => "cancel",
            Self::Fence => "fence",
        }
    }
}

struct PendingControlAck {
    kind: PendingControlKind,
    ack_rx: Receiver<Result<()>>,
}

#[derive(Default)]
struct NonblockingControlState {
    pending: Option<PendingControlAck>,
}

struct CudaArgon2Engine {
    stream: Arc<CudaStream>,
    kernel: CudaFunction,
    m_blocks: u32,
    m_cost_kib: u32,
    t_cost: u32,
    max_lanes: usize,
    hashes_per_launch_per_lane: usize,
    lane_memory: CudaSlice<u64>,
    seed_blocks: CudaSlice<u64>,
    last_blocks: CudaSlice<u64>,
    host_seed_words: Vec<u64>,
    host_last_words: Vec<u64>,
    host_hashes: Vec<[u8; POW_OUTPUT_LEN]>,
}

impl CudaArgon2Engine {
    fn new(
        device_index: u32,
        memory_total_mib: u64,
        memory_free_mib: Option<u64>,
        tuning: NvidiaKernelTuning,
        max_lanes_override: Option<usize>,
        hashes_per_launch_per_lane: u32,
    ) -> Result<Self> {
        let ctx = CudaContext::new(device_index as usize).map_err(|err| {
            anyhow!("failed to open CUDA context on device {device_index}: {err:?}")
        })?;
        let stream = ctx.default_stream();

        let (cc_major, cc_minor) = ctx
            .compute_capability()
            .map_err(|err| anyhow!("failed to query compute capability: {err:?}"))?;

        let params = pow_params()
            .map_err(|err| anyhow!("invalid Argon2 parameters for CUDA backend: {err}"))?;
        let m_blocks = params.block_count() as u32;
        let m_cost_kib = params.m_cost();
        let t_cost = params.t_cost();

        let nvrtc_options = vec![
            "--std=c++14".to_string(),
            "--restrict".to_string(),
            "--use_fast_math".to_string(),
            "--ftz=true".to_string(),
            "--fmad=true".to_string(),
            format!("-DSEINE_FIXED_M_BLOCKS={}U", m_blocks),
            format!("-DSEINE_FIXED_T_COST={}U", t_cost),
            format!("--maxrregcount={}", tuning.max_rregcount.max(1)),
            format!("--gpu-architecture=sm_{}{}", cc_major, cc_minor),
        ];
        let cubin =
            compile_cubin_with_nvrtc(CUDA_KERNEL_SRC, "seine_argon2id_fill.cu", &nvrtc_options)
                .map_err(|err| anyhow!("failed to compile CUDA CUBIN with NVRTC: {err:#}"))?;
        let module = ctx
            .load_module(Ptx::from_binary(cubin))
            .map_err(|err| anyhow!("failed to load CUDA module: {err:?}"))?;
        let kernel = module
            .load_function("argon2id_fill_kernel")
            .map_err(|err| anyhow!("failed to load CUDA kernel function: {err:?}"))?;
        let touch_kernel = module
            .load_function("touch_lane_memory_kernel")
            .map_err(|err| anyhow!("failed to load CUDA probe kernel function: {err:?}"))?;

        let lane_bytes = CPU_LANE_MEMORY_BYTES.max(u64::from(m_blocks) * 1024);
        let memory_budget_mib = derive_memory_budget_mib(memory_total_mib, memory_free_mib);
        let memory_budget_bytes = memory_budget_mib.saturating_mul(1024 * 1024);
        let usable_bytes = memory_budget_bytes;
        let by_memory = usize::try_from((usable_bytes / lane_bytes).max(1)).unwrap_or(1);
        let max_lanes_budget = by_memory.max(1).min(MAX_RECOMMENDED_LANES);
        let max_lanes = max_lanes_override
            .map(|lanes| lanes.max(1))
            .unwrap_or(max_lanes_budget)
            .min(max_lanes_budget)
            .max(1);
        let lane_stride_words = u64::from(m_blocks).saturating_mul(128);
        let hashes_per_launch_per_lane = hashes_per_launch_per_lane.max(1) as usize;

        let mut selected = None;
        for lanes in (1..=max_lanes).rev() {
            match try_allocate_cuda_buffers(&stream, m_blocks, lanes, hashes_per_launch_per_lane) {
                Ok((mut lane_memory, seed_blocks, last_blocks)) => {
                    match probe_cuda_lane_memory(
                        &stream,
                        &touch_kernel,
                        &mut lane_memory,
                        lane_stride_words,
                        lanes,
                    ) {
                        Ok(()) => {
                            selected = Some((lanes, (lane_memory, seed_blocks, last_blocks)));
                            break;
                        }
                        Err(err)
                            if lanes > 1
                                && format!("{err:?}").contains("CUDA_ERROR_OUT_OF_MEMORY") =>
                        {
                            continue;
                        }
                        Err(err) => {
                            return Err(anyhow!(
                                "failed to validate CUDA buffers for {lanes} lanes on device {device_index}: {err:?}"
                            ));
                        }
                    }
                }
                Err(err)
                    if lanes > 1 && format!("{err:?}").contains("CUDA_ERROR_OUT_OF_MEMORY") =>
                {
                    continue;
                }
                Err(err) => {
                    return Err(anyhow!(
                        "failed to allocate CUDA buffers for {lanes} lanes on device {device_index}: {err:?}"
                    ));
                }
            }
        }

        let (max_lanes, (lane_memory, seed_blocks, last_blocks)) = selected
            .ok_or_else(|| anyhow!("failed to allocate CUDA buffers for any lane count"))?;

        let launch_capacity = max_lanes.saturating_mul(hashes_per_launch_per_lane).max(1);

        Ok(Self {
            stream,
            kernel,
            m_blocks,
            m_cost_kib,
            t_cost,
            max_lanes,
            hashes_per_launch_per_lane,
            lane_memory,
            seed_blocks,
            last_blocks,
            host_seed_words: vec![0u64; launch_capacity * 256],
            host_last_words: vec![0u64; launch_capacity * 128],
            host_hashes: vec![[0u8; POW_OUTPUT_LEN]; launch_capacity],
        })
    }

    fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    fn max_hashes_per_launch(&self) -> usize {
        self.max_lanes
            .saturating_mul(self.hashes_per_launch_per_lane)
            .max(1)
    }

    fn hash_at(&self, idx: usize) -> &[u8; POW_OUTPUT_LEN] {
        &self.host_hashes[idx]
    }

    fn run_fill_batch(&mut self, header_base: &[u8], nonces: &[u64]) -> Result<usize> {
        if header_base.len() != POW_HEADER_BASE_LEN {
            bail!(
                "invalid header base length: expected {} bytes, got {}",
                POW_HEADER_BASE_LEN,
                header_base.len()
            );
        }

        if nonces.is_empty() {
            return Ok(0);
        }

        let lanes_active = self.max_lanes.min(nonces.len()).max(1);
        let max_hashes = lanes_active
            .saturating_mul(self.hashes_per_launch_per_lane)
            .max(1);
        if nonces.len() > max_hashes {
            bail!(
                "batch size {} exceeds configured CUDA launch capacity {} (lanes={} hashes_per_launch_per_lane={})",
                nonces.len(),
                max_hashes,
                lanes_active,
                self.hashes_per_launch_per_lane
            );
        }

        for (idx, nonce) in nonces.iter().copied().enumerate() {
            let start = idx * 256;
            let end = start + 256;
            build_seed_blocks_for_nonce(
                header_base,
                nonce,
                self.m_cost_kib,
                self.t_cost,
                &mut self.host_seed_words[start..end],
            )?;
        }

        let hashes_done = nonces.len();
        let seed_words = hashes_done * 256;
        {
            let mut seed_view = self
                .seed_blocks
                .try_slice_mut(0..seed_words)
                .ok_or_else(|| anyhow!("failed to slice CUDA seed buffer"))?;
            self.stream
                .memcpy_htod(&self.host_seed_words[..seed_words], &mut seed_view)
                .map_err(cuda_driver_err)?;
        }

        {
            let lanes_u32 =
                u32::try_from(lanes_active).map_err(|_| anyhow!("lane count overflow"))?;
            let active_hashes_u32 =
                u32::try_from(hashes_done).map_err(|_| anyhow!("active hash count overflow"))?;
            let lane_launch_iters_u32 = u32::try_from(
                hashes_done
                    .saturating_add(lanes_active.saturating_sub(1))
                    .saturating_div(lanes_active),
            )
            .map_err(|_| anyhow!("lane launch iteration overflow"))?;
            let cfg = LaunchConfig {
                // One cooperative block per lane. Threads in the block collaborate on Argon2
                // compression and memory transfer for that lane.
                grid_dim: (lanes_u32, 1, 1),
                block_dim: (KERNEL_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                let mut launch = self.stream.launch_builder(&self.kernel);
                launch
                    .arg(&self.seed_blocks)
                    .arg(&lanes_u32)
                    .arg(&active_hashes_u32)
                    .arg(&lane_launch_iters_u32)
                    .arg(&self.m_blocks)
                    .arg(&self.t_cost)
                    .arg(&mut self.lane_memory)
                    .arg(&mut self.last_blocks);
                launch.launch(cfg).map_err(cuda_driver_err)?;
            }
        }
        let last_words = hashes_done * 128;
        {
            let last_view = self
                .last_blocks
                .try_slice(0..last_words)
                .ok_or_else(|| anyhow!("failed to slice CUDA output buffer"))?;
            self.stream
                .memcpy_dtoh(&last_view, &mut self.host_last_words[..last_words])
                .map_err(cuda_driver_err)?;
        }

        for idx in 0..hashes_done {
            finalize_lane_hash(
                &self.host_last_words[idx * 128..(idx + 1) * 128],
                &mut self.host_hashes[idx],
            )?;
        }

        Ok(hashes_done)
    }
}

pub struct NvidiaBackend {
    instance_id: Arc<AtomicU64>,
    requested_device_index: Option<u32>,
    autotune_config_path: PathBuf,
    autotune_secs: u64,
    tuning_options: NvidiaBackendTuningOptions,
    resolved_device: RwLock<Option<NvidiaDeviceInfo>>,
    kernel_tuning: RwLock<Option<NvidiaKernelTuning>>,
    shared: Arc<NvidiaShared>,
    nonblocking_control: Mutex<NonblockingControlState>,
    worker: Mutex<Option<NvidiaWorker>>,
    max_lanes: AtomicUsize,
    hashes_per_launch_per_lane: u32,
}

impl NvidiaBackend {
    pub fn new(
        device_index: Option<u32>,
        autotune_config_path: PathBuf,
        autotune_secs: u64,
        tuning_options: NvidiaBackendTuningOptions,
    ) -> Self {
        Self {
            instance_id: Arc::new(AtomicU64::new(0)),
            requested_device_index: device_index,
            autotune_config_path,
            autotune_secs: autotune_secs.max(1),
            tuning_options: NvidiaBackendTuningOptions {
                max_rregcount_override: tuning_options.max_rregcount_override.filter(|v| *v > 0),
                max_lanes_override: tuning_options.max_lanes_override.filter(|v| *v > 0),
                autotune_samples: tuning_options.autotune_samples.max(1),
                dispatch_iters_per_lane: tuning_options
                    .dispatch_iters_per_lane
                    .filter(|v| *v > 0),
                allocation_iters_per_lane: tuning_options
                    .allocation_iters_per_lane
                    .filter(|v| *v > 0),
                hashes_per_launch_per_lane: tuning_options.hashes_per_launch_per_lane.max(1),
            },
            resolved_device: RwLock::new(None),
            kernel_tuning: RwLock::new(None),
            shared: Arc::new(NvidiaShared::new()),
            nonblocking_control: Mutex::new(NonblockingControlState::default()),
            worker: Mutex::new(None),
            max_lanes: AtomicUsize::new(1),
            hashes_per_launch_per_lane: tuning_options.hashes_per_launch_per_lane.max(1),
        }
    }

    fn validate_or_select_device(&self) -> Result<NvidiaDeviceInfo> {
        let devices = query_nvidia_devices()?;
        let selected = if let Some(requested_index) = self.requested_device_index {
            devices
                .iter()
                .find(|device| device.index == requested_index)
                .cloned()
                .ok_or_else(|| {
                    let available = devices
                        .iter()
                        .map(|device| device.index.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    anyhow!(
                        "requested NVIDIA device index {} was not found; available indices: [{}]",
                        requested_index,
                        available
                    )
                })?
        } else {
            devices[0].clone()
        };

        let memory_total_bytes = selected.memory_total_mib.saturating_mul(1024 * 1024);
        if memory_total_bytes < CPU_LANE_MEMORY_BYTES {
            bail!(
                "NVIDIA device {} ({}) reports {} MiB VRAM; at least {} MiB is required",
                selected.index,
                selected.name,
                selected.memory_total_mib,
                (CPU_LANE_MEMORY_BYTES / (1024 * 1024)).max(1)
            );
        }

        if let Ok(mut slot) = self.resolved_device.write() {
            *slot = Some(selected.clone());
        }

        Ok(selected)
    }

    fn worker_assignment_tx(&self) -> Result<Sender<WorkerCommand>> {
        let guard = self
            .worker
            .lock()
            .map_err(|_| anyhow!("nvidia worker lock poisoned"))?;
        let Some(worker) = guard.as_ref() else {
            bail!("NVIDIA backend is not started");
        };
        Ok(worker.assignment_tx.clone())
    }

    fn worker_control_tx(&self) -> Result<Sender<WorkerCommand>> {
        let guard = self
            .worker
            .lock()
            .map_err(|_| anyhow!("nvidia worker lock poisoned"))?;
        let Some(worker) = guard.as_ref() else {
            bail!("NVIDIA backend is not started");
        };
        Ok(worker.control_tx.clone())
    }

    fn clear_nonblocking_control_state(&self) {
        if let Ok(mut state) = self.nonblocking_control.lock() {
            state.pending = None;
        }
    }

    fn send_control_blocking(&self, kind: PendingControlKind) -> Result<()> {
        let tx = self.worker_control_tx()?;
        let (ack_tx, ack_rx) = bounded::<Result<()>>(1);
        let cmd = match kind {
            PendingControlKind::Cancel => WorkerCommand::Cancel(ack_tx),
            PendingControlKind::Fence => WorkerCommand::Fence(ack_tx),
        };
        tx.send(cmd).map_err(|_| {
            anyhow!(
                "nvidia worker channel closed while issuing {}",
                kind.label()
            )
        })?;
        ack_rx
            .recv()
            .map_err(|_| anyhow!("nvidia worker did not acknowledge {}", kind.label()))??;
        Ok(())
    }

    fn control_nonblocking(&self, kind: PendingControlKind) -> Result<BackendCallStatus> {
        let tx = self.worker_control_tx()?;
        let mut state = self
            .nonblocking_control
            .lock()
            .map_err(|_| anyhow!("nvidia nonblocking control lock poisoned"))?;

        if let Some(pending) = state.pending.as_mut() {
            match pending.ack_rx.try_recv() {
                Ok(result) => {
                    state.pending = None;
                    result?;
                }
                Err(TryRecvError::Empty) => {
                    if pending.kind == kind {
                        return Ok(BackendCallStatus::Pending);
                    }
                    return Ok(BackendCallStatus::Pending);
                }
                Err(TryRecvError::Disconnected) => {
                    let pending_kind = pending.kind;
                    state.pending = None;
                    bail!(
                        "nvidia worker disconnected while waiting for {} acknowledgement",
                        pending_kind.label()
                    );
                }
            }
        }

        let (ack_tx, ack_rx) = bounded::<Result<()>>(1);
        let cmd = match kind {
            PendingControlKind::Cancel => WorkerCommand::Cancel(ack_tx),
            PendingControlKind::Fence => WorkerCommand::Fence(ack_tx),
        };
        match tx.try_send(cmd) {
            Ok(()) => {
                state.pending = Some(PendingControlAck { kind, ack_rx });
                if let Some(pending) = state.pending.as_mut() {
                    match pending.ack_rx.try_recv() {
                        Ok(result) => {
                            state.pending = None;
                            result?;
                            return Ok(BackendCallStatus::Complete);
                        }
                        Err(TryRecvError::Empty) => {
                            return Ok(BackendCallStatus::Pending);
                        }
                        Err(TryRecvError::Disconnected) => {
                            let pending_kind = pending.kind;
                            state.pending = None;
                            bail!(
                                "nvidia worker disconnected while waiting for {} acknowledgement",
                                pending_kind.label()
                            );
                        }
                    }
                }
                Ok(BackendCallStatus::Pending)
            }
            Err(TrySendError::Full(_)) => Ok(BackendCallStatus::Pending),
            Err(TrySendError::Disconnected(_)) => {
                bail!(
                    "nvidia worker channel closed while issuing {}",
                    kind.label()
                )
            }
        }
    }

    fn resolve_kernel_tuning(&self, selected: &NvidiaDeviceInfo) -> NvidiaKernelTuning {
        if let Ok(slot) = self.kernel_tuning.read() {
            if let Some(tuning) = *slot {
                return tuning;
            }
        }

        if let Some(forced_rregcount) = self.tuning_options.max_rregcount_override {
            let forced = NvidiaKernelTuning {
                max_rregcount: forced_rregcount.max(1),
                block_loop_unroll: false,
            };
            if let Ok(mut slot) = self.kernel_tuning.write() {
                *slot = Some(forced);
            }
            return forced;
        }

        let key = build_nvidia_autotune_key(selected);
        if let Some(cached) = load_nvidia_cached_tuning(&self.autotune_config_path, &key) {
            if let Ok(mut slot) = self.kernel_tuning.write() {
                *slot = Some(cached);
            }
            return cached;
        }

        let tuned = autotune_nvidia_kernel_tuning(
            selected,
            &self.autotune_config_path,
            self.autotune_secs,
            self.tuning_options.autotune_samples,
            self.tuning_options.max_lanes_override,
            self.tuning_options.hashes_per_launch_per_lane,
        )
        .unwrap_or_else(|_| NvidiaKernelTuning::default());
        if let Ok(mut slot) = self.kernel_tuning.write() {
            *slot = Some(tuned);
        }
        tuned
    }
}

impl Drop for NvidiaBackend {
    fn drop(&mut self) {
        self.stop();
    }
}

impl PowBackend for NvidiaBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn lanes(&self) -> usize {
        self.max_lanes.load(Ordering::Acquire).max(1)
    }

    fn device_memory_bytes(&self) -> Option<u64> {
        self.resolved_device.read().ok().and_then(|dev| {
            dev.as_ref()
                .map(|d| d.memory_total_mib.saturating_mul(1024 * 1024))
        })
    }

    fn set_instance_id(&self, id: BackendInstanceId) {
        self.instance_id.store(id, Ordering::Release);
    }

    fn set_event_sink(&self, sink: Sender<BackendEvent>) {
        if let Ok(mut slot) = self.shared.event_sink.write() {
            *slot = Some(sink);
        }
    }

    fn start(&self) -> Result<()> {
        {
            let guard = self
                .worker
                .lock()
                .map_err(|_| anyhow!("nvidia worker lock poisoned"))?;
            if guard.is_some() {
                return Ok(());
            }
        }

        let selected = self.validate_or_select_device()?;
        let tuning = self.resolve_kernel_tuning(&selected);
        let mut engine = CudaArgon2Engine::new(
            selected.index,
            selected.memory_total_mib,
            selected.memory_free_mib,
            tuning,
            self.tuning_options.max_lanes_override,
            self.tuning_options.hashes_per_launch_per_lane,
        )
        .with_context(|| {
            format!(
                "failed to initialize CUDA engine on NVIDIA device {} ({})",
                selected.index, selected.name
            )
        })?;

        self.max_lanes
            .store(engine.max_lanes().max(1), Ordering::Release);

        self.shared.error_emitted.store(false, Ordering::Release);
        self.shared.hashes.store(0, Ordering::Release);
        self.shared.active_lanes.store(0, Ordering::Release);
        self.shared.pending_work.store(0, Ordering::Release);
        self.shared
            .inflight_assignment_hashes
            .store(0, Ordering::Release);
        if let Ok(mut slot) = self.shared.inflight_assignment_started_at.lock() {
            *slot = None;
        }
        self.clear_nonblocking_control_state();

        let (assignment_tx, assignment_rx) = bounded::<WorkerCommand>(ASSIGN_CHANNEL_CAPACITY);
        let (control_tx, control_rx) = bounded::<WorkerCommand>(CONTROL_CHANNEL_CAPACITY);
        let shared = Arc::clone(&self.shared);
        let instance_id = Arc::clone(&self.instance_id);
        let handle = thread::Builder::new()
            .name(format!(
                "seine-nvidia-worker-{}",
                self.instance_id.load(Ordering::Acquire)
            ))
            .spawn(move || worker_loop(&mut engine, assignment_rx, control_rx, shared, instance_id))
            .map_err(|err| anyhow!("failed to spawn nvidia worker thread: {err}"))?;

        let mut guard = self
            .worker
            .lock()
            .map_err(|_| anyhow!("nvidia worker lock poisoned"))?;
        *guard = Some(NvidiaWorker {
            assignment_tx,
            control_tx,
            handle,
        });

        Ok(())
    }

    fn stop(&self) {
        let worker = match self.worker.lock() {
            Ok(mut slot) => slot.take(),
            Err(_) => None,
        };
        if let Some(worker) = worker {
            if worker.control_tx.send(WorkerCommand::Stop).is_err() {
                let _ = worker.assignment_tx.send(WorkerCommand::Stop);
            }
            let _ = worker.handle.join();
        }
        self.clear_nonblocking_control_state();

        self.shared.active_lanes.store(0, Ordering::Release);
        self.shared.pending_work.store(0, Ordering::Release);
        self.shared
            .inflight_assignment_hashes
            .store(0, Ordering::Release);
        if let Ok(mut slot) = self.shared.inflight_assignment_started_at.lock() {
            *slot = None;
        }
    }

    fn assign_work(&self, work: WorkAssignment) -> Result<()> {
        self.worker_assignment_tx()?
            .send(WorkerCommand::Assign(work))
            .map_err(|_| anyhow!("nvidia worker channel closed while assigning work"))
    }

    fn assign_work_batch(&self, work: &[WorkAssignment]) -> Result<()> {
        let mut batch = normalize_assignment_batch(work)?;
        match batch.len() {
            0 => Ok(()),
            1 => self.assign_work(batch.pop().expect("single-item batch should be present")),
            _ => self
                .worker_assignment_tx()?
                .send(WorkerCommand::AssignBatch(batch))
                .map_err(|_| anyhow!("nvidia worker channel closed while assigning work")),
        }
    }

    fn assign_work_batch_with_deadline(
        &self,
        work: &[WorkAssignment],
        deadline: Instant,
    ) -> Result<()> {
        self.assign_work_batch(work)?;
        if Instant::now() > deadline {
            return Err(anyhow!(
                "assignment call exceeded deadline by {}ms",
                Instant::now()
                    .saturating_duration_since(deadline)
                    .as_millis()
            ));
        }
        Ok(())
    }

    fn supports_assignment_batching(&self) -> bool {
        true
    }

    fn supports_true_nonblocking(&self) -> bool {
        true
    }

    fn assign_work_batch_nonblocking(&self, work: &[WorkAssignment]) -> Result<BackendCallStatus> {
        let mut batch = normalize_assignment_batch(work)?;
        if batch.is_empty() {
            return Ok(BackendCallStatus::Complete);
        }

        let tx = self.worker_assignment_tx()?;
        let cmd = if batch.len() == 1 {
            WorkerCommand::Assign(batch.pop().expect("single-item batch should be present"))
        } else {
            WorkerCommand::AssignBatch(batch)
        };
        match tx.try_send(cmd) {
            Ok(()) => Ok(BackendCallStatus::Complete),
            Err(TrySendError::Full(_)) => Ok(BackendCallStatus::Pending),
            Err(TrySendError::Disconnected(_)) => {
                bail!("nvidia worker channel closed while assigning work")
            }
        }
    }

    fn wait_for_nonblocking_progress(&self, wait_for: Duration) -> Result<()> {
        let wait_for = wait_for.max(Duration::from_micros(10));
        let mut state = self
            .nonblocking_control
            .lock()
            .map_err(|_| anyhow!("nvidia nonblocking control lock poisoned"))?;
        let Some(pending) = state.pending.as_mut() else {
            drop(state);
            thread::sleep(wait_for);
            return Ok(());
        };

        match pending.ack_rx.recv_timeout(wait_for) {
            Ok(result) => {
                state.pending = None;
                result?;
                Ok(())
            }
            Err(RecvTimeoutError::Timeout) => Ok(()),
            Err(RecvTimeoutError::Disconnected) => {
                let pending_kind = pending.kind;
                state.pending = None;
                bail!(
                    "nvidia worker disconnected while waiting for {} acknowledgement",
                    pending_kind.label()
                );
            }
        }
    }

    fn cancel_work_nonblocking(&self) -> Result<BackendCallStatus> {
        self.control_nonblocking(PendingControlKind::Cancel)
    }

    fn fence_nonblocking(&self) -> Result<BackendCallStatus> {
        self.control_nonblocking(PendingControlKind::Fence)
    }

    fn cancel_work(&self) -> Result<()> {
        self.send_control_blocking(PendingControlKind::Cancel)
    }

    fn request_timeout_interrupt(&self) -> Result<()> {
        let tx = match self.worker_control_tx() {
            Ok(tx) => tx,
            Err(_) => return Ok(()),
        };
        let (ack_tx, _ack_rx) = bounded::<Result<()>>(1);
        match tx.try_send(WorkerCommand::Cancel(ack_tx)) {
            Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
            Err(TrySendError::Disconnected(_)) => Ok(()),
        }
    }

    fn fence(&self) -> Result<()> {
        self.send_control_blocking(PendingControlKind::Fence)
    }

    fn take_hashes(&self) -> u64 {
        self.shared.hashes.swap(0, Ordering::AcqRel)
    }

    fn take_telemetry(&self) -> BackendTelemetry {
        let inflight_assignment_micros = if self.shared.pending_work.load(Ordering::Acquire) > 0 {
            self.shared
                .inflight_assignment_started_at
                .lock()
                .ok()
                .and_then(|slot| {
                    slot.as_ref()
                        .map(|started| started.elapsed().as_micros().min(u64::MAX as u128) as u64)
                })
                .unwrap_or(0)
        } else {
            0
        };

        BackendTelemetry {
            active_lanes: self.shared.active_lanes.load(Ordering::Acquire),
            pending_work: self.shared.pending_work.load(Ordering::Acquire),
            dropped_events: self.shared.dropped_events.swap(0, Ordering::AcqRel),
            completed_assignments: self.shared.completed_assignments.swap(0, Ordering::AcqRel),
            completed_assignment_hashes: self
                .shared
                .completed_assignment_hashes
                .swap(0, Ordering::AcqRel),
            completed_assignment_micros: self
                .shared
                .completed_assignment_micros
                .swap(0, Ordering::AcqRel),
            inflight_assignment_hashes: self
                .shared
                .inflight_assignment_hashes
                .load(Ordering::Acquire),
            inflight_assignment_micros,
            ..BackendTelemetry::default()
        }
    }

    fn preemption_granularity(&self) -> PreemptionGranularity {
        let per_launch = self.hashes_per_launch_per_lane.max(1) as u64;
        PreemptionGranularity::Hashes((self.lanes() as u64).saturating_mul(per_launch).max(1))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            preferred_iters_per_lane: Some(
                self.tuning_options
                    .dispatch_iters_per_lane
                    .unwrap_or(DEFAULT_DISPATCH_ITERS_PER_LANE)
                    .max(1),
            ),
            preferred_allocation_iters_per_lane: Some(
                self.tuning_options
                    .allocation_iters_per_lane
                    .unwrap_or(DEFAULT_ALLOCATION_ITERS_PER_LANE)
                    .max(1),
            ),
            preferred_hash_poll_interval: Some(Duration::from_millis(25)),
            preferred_assignment_timeout: None,
            preferred_control_timeout: None,
            preferred_assignment_timeout_strikes: None,
            preferred_worker_queue_depth: Some(32),
            max_inflight_assignments: 32,
            deadline_support: DeadlineSupport::Cooperative,
            assignment_semantics: AssignmentSemantics::Replace,
            execution_model: BackendExecutionModel::Nonblocking,
            nonblocking_poll_min: Some(Duration::from_micros(50)),
            nonblocking_poll_max: Some(Duration::from_millis(1)),
        }
    }

    fn bench_backend(&self) -> Option<&dyn BenchBackend> {
        Some(self)
    }
}

impl BenchBackend for NvidiaBackend {
    fn kernel_bench(&self, seconds: u64, shutdown: &AtomicBool) -> Result<u64> {
        let selected = self.validate_or_select_device()?;
        let tuning = self.resolve_kernel_tuning(&selected);
        let mut engine = CudaArgon2Engine::new(
            selected.index,
            selected.memory_total_mib,
            selected.memory_free_mib,
            tuning,
            self.tuning_options.max_lanes_override,
            self.tuning_options.hashes_per_launch_per_lane,
        )
        .with_context(|| {
            format!(
                "failed to initialize CUDA engine for benchmark on device {} ({})",
                selected.index, selected.name
            )
        })?;

        let header = [0u8; POW_HEADER_BASE_LEN];
        let deadline = Instant::now() + Duration::from_secs(seconds.max(1));
        let mut nonce_cursor = 0u64;
        let mut total = 0u64;
        let mut nonces = vec![0u64; engine.max_hashes_per_launch()];

        while Instant::now() < deadline && !shutdown.load(Ordering::Acquire) {
            let batch_hashes = engine.max_hashes_per_launch().max(1);
            if nonces.len() < batch_hashes {
                nonces.resize(batch_hashes, 0);
            }
            for nonce in nonces.iter_mut().take(batch_hashes) {
                *nonce = nonce_cursor;
                nonce_cursor = nonce_cursor.wrapping_add(1);
            }

            let done = engine.run_fill_batch(&header, &nonces[..batch_hashes])?;
            total = total.saturating_add(done as u64);
        }

        Ok(total)
    }
}

fn worker_loop(
    engine: &mut CudaArgon2Engine,
    assignment_rx: Receiver<WorkerCommand>,
    control_rx: Receiver<WorkerCommand>,
    shared: Arc<NvidiaShared>,
    instance_id: Arc<AtomicU64>,
) {
    let mut active: Option<ActiveAssignment> = None;
    let mut queued: VecDeque<WorkAssignment> = VecDeque::new();
    let mut fence_waiters: Vec<Sender<Result<()>>> = Vec::new();
    let mut assignment_open = true;
    let mut control_open = true;
    let mut running = true;
    let mut nonce_buf = vec![0u64; engine.max_hashes_per_launch()];

    while running {
        while control_open {
            match control_rx.try_recv() {
                Ok(cmd) => {
                    running = handle_worker_command(
                        cmd,
                        &mut active,
                        &mut queued,
                        &mut fence_waiters,
                        &shared,
                    );
                    if !running {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    control_open = false;
                    break;
                }
            }
        }
        if !running {
            break;
        }

        if active.is_none() {
            if assignment_open {
                match assignment_rx.try_recv() {
                    Ok(cmd) => {
                        running = handle_worker_command(
                            cmd,
                            &mut active,
                            &mut queued,
                            &mut fence_waiters,
                            &shared,
                        );
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => assignment_open = false,
                }
            }
            if !running {
                break;
            }

            if active.is_none() {
                if !assignment_open && !control_open {
                    break;
                }
                if assignment_open && control_open {
                    crossbeam_channel::select! {
                        recv(control_rx) -> cmd => match cmd {
                            Ok(cmd) => {
                                running = handle_worker_command(
                                    cmd,
                                    &mut active,
                                    &mut queued,
                                    &mut fence_waiters,
                                    &shared,
                                );
                            }
                            Err(_) => control_open = false,
                        },
                        recv(assignment_rx) -> cmd => match cmd {
                            Ok(cmd) => {
                                running = handle_worker_command(
                                    cmd,
                                    &mut active,
                                    &mut queued,
                                    &mut fence_waiters,
                                    &shared,
                                );
                            }
                            Err(_) => assignment_open = false,
                        },
                    }
                } else if assignment_open {
                    match assignment_rx.recv() {
                        Ok(cmd) => {
                            running = handle_worker_command(
                                cmd,
                                &mut active,
                                &mut queued,
                                &mut fence_waiters,
                                &shared,
                            );
                        }
                        Err(_) => assignment_open = false,
                    }
                } else if control_open {
                    match control_rx.recv() {
                        Ok(cmd) => {
                            running = handle_worker_command(
                                cmd,
                                &mut active,
                                &mut queued,
                                &mut fence_waiters,
                                &shared,
                            );
                        }
                        Err(_) => control_open = false,
                    }
                }
                continue;
            }
        }

        loop {
            let mut handled = false;
            if control_open {
                match control_rx.try_recv() {
                    Ok(cmd) => {
                        handled = true;
                        running = handle_worker_command(
                            cmd,
                            &mut active,
                            &mut queued,
                            &mut fence_waiters,
                            &shared,
                        );
                        if !running {
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => control_open = false,
                }
            }
            if assignment_open {
                match assignment_rx.try_recv() {
                    Ok(cmd) => {
                        handled = true;
                        running = handle_worker_command(
                            cmd,
                            &mut active,
                            &mut queued,
                            &mut fence_waiters,
                            &shared,
                        );
                        if !running {
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => assignment_open = false,
                }
            }
            if !handled {
                break;
            }
        }

        if !running {
            break;
        }

        let Some(current) = active.as_mut() else {
            continue;
        };

        if current.remaining == 0 {
            let next_pending = queued.len() as u64;
            finalize_active_assignment(&shared, current, next_pending);
            if let Some(next) = queued.pop_front() {
                activate_assignment(&shared, &mut active, next, queued.len() as u64 + 1);
                continue;
            }
            active = None;
            drain_fence_waiters(&mut fence_waiters, Ok(()));
            continue;
        }

        if Instant::now() >= current.work.template.stop_at {
            queued.clear();
            finalize_active_assignment(&shared, current, 0);
            active = None;
            drain_fence_waiters(&mut fence_waiters, Ok(()));
            continue;
        }

        let hashes_per_batch = engine
            .max_hashes_per_launch()
            .min(current.remaining as usize)
            .max(1);
        if nonce_buf.len() < hashes_per_batch {
            nonce_buf.resize(hashes_per_batch, 0);
        }
        for (idx, nonce) in nonce_buf.iter_mut().take(hashes_per_batch).enumerate() {
            *nonce = current.next_nonce.wrapping_add(idx as u64);
        }

        let active_lanes = engine.max_lanes().min(hashes_per_batch).max(1);
        shared.active_lanes.store(active_lanes as u64, Ordering::Release);

        let done = match engine.run_fill_batch(
            current.work.template.header_base.as_ref(),
            &nonce_buf[..hashes_per_batch],
        ) {
            Ok(done) => done,
            Err(err) => {
                emit_worker_error(
                    &shared,
                    &instance_id,
                    format!("CUDA batch execution failed: {err:#}"),
                );
                break;
            }
        };

        if done == 0 {
            let next_pending = queued.len() as u64;
            finalize_active_assignment(&shared, current, next_pending);
            if let Some(next) = queued.pop_front() {
                activate_assignment(&shared, &mut active, next, queued.len() as u64 + 1);
                continue;
            }
            active = None;
            drain_fence_waiters(&mut fence_waiters, Ok(()));
            continue;
        }

        current.hashes_done = current.hashes_done.saturating_add(done as u64);
        current.next_nonce = current.next_nonce.wrapping_add(done as u64);
        current.remaining = current.remaining.saturating_sub(done as u64);

        shared.hashes.fetch_add(done as u64, Ordering::Relaxed);
        shared
            .inflight_assignment_hashes
            .store(current.hashes_done, Ordering::Release);

        let mut solved_nonce = None;
        for idx in 0..done {
            if hash_meets_target(engine.hash_at(idx), &current.work.template.target) {
                solved_nonce = Some(nonce_buf[idx]);
                break;
            }
        }

        if let Some(nonce) = solved_nonce {
            send_backend_event(
                &shared,
                BackendEvent::Solution(MiningSolution {
                    epoch: current.work.template.epoch,
                    nonce,
                    backend_id: instance_id.load(Ordering::Acquire),
                    backend: BACKEND_NAME,
                }),
            );
            queued.clear();
            finalize_active_assignment(&shared, current, 0);
            active = None;
            drain_fence_waiters(&mut fence_waiters, Ok(()));
            continue;
        }
    }

    if let Some(active_assignment) = active.as_ref() {
        finalize_active_assignment(&shared, active_assignment, 0);
    }
    queued.clear();
    clear_worker_pending_state(&shared);

    drain_fence_waiters(
        &mut fence_waiters,
        Err(anyhow!("nvidia worker stopped before fence completion")),
    );
}

fn handle_worker_command(
    cmd: WorkerCommand,
    active: &mut Option<ActiveAssignment>,
    queued: &mut VecDeque<WorkAssignment>,
    fence_waiters: &mut Vec<Sender<Result<()>>>,
    shared: &NvidiaShared,
) -> bool {
    match cmd {
        WorkerCommand::Assign(work) => {
            replace_assignment_queue(shared, active, queued, vec![work]);
            true
        }
        WorkerCommand::AssignBatch(work) => {
            replace_assignment_queue(shared, active, queued, work);
            true
        }
        WorkerCommand::Cancel(ack) => {
            if let Some(current) = active.as_ref() {
                finalize_active_assignment(shared, current, 0);
                *active = None;
            }
            queued.clear();
            clear_worker_pending_state(shared);
            let _ = ack.send(Ok(()));
            if active.is_none() && queued.is_empty() {
                drain_fence_waiters(fence_waiters, Ok(()));
            }
            true
        }
        WorkerCommand::Fence(ack) => {
            if active.is_none() && queued.is_empty() {
                let _ = ack.send(Ok(()));
            } else {
                fence_waiters.push(ack);
            }
            true
        }
        WorkerCommand::Stop => false,
    }
}

fn replace_assignment_queue(
    shared: &NvidiaShared,
    active: &mut Option<ActiveAssignment>,
    queued: &mut VecDeque<WorkAssignment>,
    assignments: Vec<WorkAssignment>,
) {
    let pending_after_replace = assignments.len() as u64;
    if let Some(current) = active.as_ref() {
        finalize_active_assignment(shared, current, pending_after_replace);
    }
    *active = None;
    queued.clear();

    shared.error_emitted.store(false, Ordering::Release);

    if assignments.is_empty() {
        clear_worker_pending_state(shared);
        return;
    }

    let mut assignments = assignments.into_iter();
    let first = assignments
        .next()
        .expect("non-empty assignments should include an active item");
    queued.extend(assignments);
    activate_assignment(shared, active, first, queued.len() as u64 + 1);
}

fn activate_assignment(
    shared: &NvidiaShared,
    active: &mut Option<ActiveAssignment>,
    work: WorkAssignment,
    pending_work: u64,
) {
    shared
        .pending_work
        .store(pending_work.max(1), Ordering::Release);
    shared.active_lanes.store(0, Ordering::Release);
    shared
        .inflight_assignment_hashes
        .store(0, Ordering::Release);
    if let Ok(mut slot) = shared.inflight_assignment_started_at.lock() {
        *slot = Some(Instant::now());
    }
    *active = Some(ActiveAssignment::new(work));
}

fn clear_worker_pending_state(shared: &NvidiaShared) {
    shared.active_lanes.store(0, Ordering::Release);
    shared.pending_work.store(0, Ordering::Release);
    shared
        .inflight_assignment_hashes
        .store(0, Ordering::Release);
    if let Ok(mut slot) = shared.inflight_assignment_started_at.lock() {
        *slot = None;
    }
}

fn finalize_active_assignment(
    shared: &NvidiaShared,
    active: &ActiveAssignment,
    pending_after: u64,
) {
    shared.active_lanes.store(0, Ordering::Release);
    shared.pending_work.store(pending_after, Ordering::Release);
    shared
        .inflight_assignment_hashes
        .store(0, Ordering::Release);
    if let Ok(mut slot) = shared.inflight_assignment_started_at.lock() {
        *slot = None;
    }

    shared.completed_assignments.fetch_add(1, Ordering::Relaxed);
    shared
        .completed_assignment_hashes
        .fetch_add(active.hashes_done, Ordering::Relaxed);
    shared.completed_assignment_micros.fetch_add(
        active
            .started_at
            .elapsed()
            .as_micros()
            .min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
}

fn drain_fence_waiters(waiters: &mut Vec<Sender<Result<()>>>, result: Result<()>) {
    for waiter in waiters.drain(..) {
        let payload = result
            .as_ref()
            .map(|_| ())
            .map_err(|err| anyhow!("{err:#}"));
        let _ = waiter.send(payload);
    }
}

fn send_backend_event(shared: &NvidiaShared, event: BackendEvent) {
    let outbound = match shared.event_sink.read() {
        Ok(slot) => slot.clone(),
        Err(_) => None,
    };

    let Some(outbound) = outbound else {
        shared.dropped_events.fetch_add(1, Ordering::Relaxed);
        return;
    };

    match outbound.send_timeout(event, EVENT_SEND_WAIT) {
        Ok(()) => {}
        Err(SendTimeoutError::Timeout(returned)) => {
            if outbound.send(returned).is_err() {
                shared.dropped_events.fetch_add(1, Ordering::Relaxed);
            }
        }
        Err(SendTimeoutError::Disconnected(_)) => {
            shared.dropped_events.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn emit_worker_error(shared: &NvidiaShared, instance_id: &AtomicU64, message: String) {
    if shared.error_emitted.swap(true, Ordering::AcqRel) {
        return;
    }

    send_backend_event(
        shared,
        BackendEvent::Error {
            backend_id: instance_id.load(Ordering::Acquire),
            backend: BACKEND_NAME,
            message,
        },
    );
}

fn cuda_driver_err(err: DriverError) -> anyhow::Error {
    anyhow!("CUDA driver error: {err:?}")
}

fn compile_cubin_with_nvrtc(
    source: &str,
    program_name: &str,
    options: &[String],
) -> Result<Vec<u8>> {
    let source_c = CString::new(source)
        .map_err(|_| anyhow!("CUDA kernel source contains interior NUL byte"))?;
    let program_name_c = CString::new(program_name)
        .map_err(|_| anyhow!("CUDA program name contains interior NUL byte"))?;

    let program = nvrtc_result::create_program(&source_c, Some(&program_name_c))
        .map_err(|err| anyhow!("nvrtcCreateProgram failed: {err:?}"))?;

    let compile_result = unsafe { nvrtc_result::compile_program(program, options) };
    if let Err(err) = compile_result {
        let compile_log = unsafe { nvrtc_result::get_program_log(program).ok() }
            .map(|raw| nvrtc_log_to_string(&raw))
            .unwrap_or_default();
        let _ = unsafe { nvrtc_result::destroy_program(program) };
        if compile_log.is_empty() {
            bail!("nvrtcCompileProgram failed: {err:?}");
        }
        bail!("nvrtcCompileProgram failed: {err:?}; log: {compile_log}");
    }

    let cubin = unsafe {
        let mut cubin_size: usize = 0;
        nvrtc_sys::nvrtcGetCUBINSize(program, &mut cubin_size as *mut usize)
            .result()
            .map_err(|err| anyhow!("nvrtcGetCUBINSize failed: {err:?}"))?;
        if cubin_size == 0 {
            bail!("nvrtcGetCUBINSize returned zero bytes");
        }

        let mut cubin = vec![0u8; cubin_size];
        nvrtc_sys::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast::<c_char>())
            .result()
            .map_err(|err| anyhow!("nvrtcGetCUBIN failed: {err:?}"))?;
        cubin
    };

    unsafe { nvrtc_result::destroy_program(program) }
        .map_err(|err| anyhow!("nvrtcDestroyProgram failed: {err:?}"))?;

    Ok(cubin)
}

fn nvrtc_log_to_string(raw: &[c_char]) -> String {
    if raw.is_empty() {
        return String::new();
    }
    unsafe { CStr::from_ptr(raw.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string()
}

fn build_seed_blocks_for_nonce(
    header_base: &[u8],
    nonce: u64,
    m_cost_kib: u32,
    t_cost: u32,
    out_words: &mut [u64],
) -> Result<()> {
    if out_words.len() < 256 {
        bail!("seed output buffer too small: expected at least 256 words");
    }

    let mut initial = Blake2b512::new();
    initial.update(1u32.to_le_bytes());
    initial.update((POW_OUTPUT_LEN as u32).to_le_bytes());
    initial.update(m_cost_kib.to_le_bytes());
    initial.update(t_cost.to_le_bytes());
    initial.update(ARGON2_VERSION_V13.to_le_bytes());
    initial.update(ARGON2_ALGORITHM_ID.to_le_bytes());
    initial.update((std::mem::size_of::<u64>() as u32).to_le_bytes());
    initial.update(nonce.to_le_bytes());
    initial.update((header_base.len() as u32).to_le_bytes());
    initial.update(header_base);
    initial.update(0u32.to_le_bytes());
    initial.update(0u32.to_le_bytes());

    let h0 = initial.finalize();

    let lane_index = 0u32.to_le_bytes();
    for (seed_idx, words_chunk) in out_words.chunks_exact_mut(128).take(2).enumerate() {
        let block_index = (seed_idx as u32).to_le_bytes();
        let mut block_bytes = [0u8; 1024];
        blake2b_long(&[h0.as_ref(), &block_index, &lane_index], &mut block_bytes)?;

        for (chunk, dst) in block_bytes.chunks_exact(8).zip(words_chunk.iter_mut()) {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            *dst = u64::from_le_bytes(bytes);
        }
    }

    Ok(())
}

fn finalize_lane_hash(last_block_words: &[u64], out: &mut [u8; POW_OUTPUT_LEN]) -> Result<()> {
    if last_block_words.len() < 128 {
        bail!(
            "last block word buffer too small: expected at least 128 words, got {}",
            last_block_words.len()
        );
    }

    let mut block_bytes = [0u8; 1024];
    for (dst, word) in block_bytes
        .chunks_exact_mut(8)
        .zip(last_block_words.iter().take(128))
    {
        dst.copy_from_slice(&word.to_le_bytes());
    }

    blake2b_long(&[&block_bytes], out)
}

fn blake2b_long(inputs: &[&[u8]], out: &mut [u8]) -> Result<()> {
    if out.is_empty() {
        bail!("blake2b_long output buffer is empty");
    }

    let len_bytes = u32::try_from(out.len())
        .map(|v| v.to_le_bytes())
        .map_err(|_| anyhow!("blake2b_long output length overflow"))?;

    if out.len() <= Blake2b512::output_size() {
        let mut digest = Blake2bVar::new(out.len())
            .map_err(|_| anyhow!("invalid variable Blake2b output length"))?;

        digest::Update::update(&mut digest, &len_bytes);
        for input in inputs {
            digest::Update::update(&mut digest, input);
        }

        digest
            .finalize_variable(out)
            .map_err(|_| anyhow!("failed to finalize Blake2b variable output"))?;
        return Ok(());
    }

    let half_hash_len = Blake2b512::output_size() / 2;
    let mut digest = Blake2b512::new();
    digest.update(len_bytes);
    for input in inputs {
        digest.update(input);
    }

    let mut last_output = digest.finalize();
    out[..half_hash_len].copy_from_slice(&last_output[..half_hash_len]);

    let mut counter = half_hash_len;
    while out.len().saturating_sub(counter) > Blake2b512::output_size() {
        last_output = Blake2b512::digest(last_output);
        let end = counter + half_hash_len;
        out[counter..end].copy_from_slice(&last_output[..half_hash_len]);
        counter = end;
    }

    let last_block_size = out.len().saturating_sub(counter);
    let mut final_digest = Blake2bVar::new(last_block_size)
        .map_err(|_| anyhow!("invalid final Blake2b output length"))?;
    digest::Update::update(&mut final_digest, &last_output);
    final_digest
        .finalize_variable(&mut out[counter..])
        .map_err(|_| anyhow!("failed to finalize tail Blake2b output"))?;

    Ok(())
}

fn normalize_assignment_batch(work: &[WorkAssignment]) -> Result<Vec<WorkAssignment>> {
    let Some(first) = work.first() else {
        return Ok(Vec::new());
    };

    let mut normalized = Vec::with_capacity(work.len());
    let mut expected_start = first.nonce_chunk.start_nonce;
    let mut total_nonce_count = 0u64;

    for (idx, assignment) in work.iter().enumerate() {
        ensure_compatible_template(&first.template, &assignment.template).with_context(|| {
            format!(
                "assignment batch item {} references a different template",
                idx + 1
            )
        })?;

        if assignment.nonce_chunk.start_nonce != expected_start {
            bail!(
                "assignment batch is not contiguous at item {} (expected nonce {}, got {})",
                idx + 1,
                expected_start,
                assignment.nonce_chunk.start_nonce
            );
        }
        if assignment.nonce_chunk.nonce_count == 0 {
            bail!("assignment batch item {} has zero nonce_count", idx + 1);
        }

        total_nonce_count = total_nonce_count
            .checked_add(assignment.nonce_chunk.nonce_count)
            .ok_or_else(|| anyhow!("assignment batch nonce_count overflow"))?;
        expected_start = expected_start.wrapping_add(assignment.nonce_chunk.nonce_count);
        normalized.push(assignment.clone());
    }

    if total_nonce_count == 0 {
        bail!("assignment batch nonce_count must be non-zero");
    }

    Ok(normalized)
}

fn ensure_compatible_template(first: &Arc<WorkTemplate>, second: &Arc<WorkTemplate>) -> Result<()> {
    if first.work_id != second.work_id {
        bail!(
            "mismatched work ids ({} vs {})",
            first.work_id,
            second.work_id
        );
    }
    if first.epoch != second.epoch {
        bail!("mismatched epochs ({} vs {})", first.epoch, second.epoch);
    }
    if first.target != second.target {
        bail!("mismatched target");
    }
    if first.stop_at != second.stop_at {
        bail!("mismatched stop_at");
    }
    if first.header_base.as_ref() != second.header_base.as_ref() {
        bail!("mismatched header_base");
    }
    Ok(())
}

fn query_nvidia_devices() -> Result<Vec<NvidiaDeviceInfo>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to execute nvidia-smi; ensure NVIDIA drivers are installed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            bail!(
                "nvidia-smi returned non-zero exit status ({})",
                output.status
            );
        }
        bail!("nvidia-smi query failed: {stderr}");
    }

    let stdout = String::from_utf8(output.stdout).context("nvidia-smi output was not UTF-8")?;
    let devices = parse_nvidia_smi_query_output(&stdout)?;
    if devices.is_empty() {
        bail!("nvidia-smi reported no NVIDIA devices");
    }

    Ok(devices)
}

fn parse_nvidia_smi_query_output(raw: &str) -> Result<Vec<NvidiaDeviceInfo>> {
    let mut devices = Vec::new();
    for (line_idx, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let columns = line.split(',').map(str::trim).collect::<Vec<_>>();
        if columns.len() < 3 {
            bail!(
                "unexpected nvidia-smi output at line {}: '{line}'",
                line_idx + 1
            );
        }

        let index = columns[0].parse::<u32>().with_context(|| {
            format!(
                "invalid GPU index '{}' at line {}",
                columns[0],
                line_idx + 1
            )
        })?;
        let (name_columns_end, memory_total_mib, memory_free_mib) = if columns.len() >= 4 {
            let total_idx = columns.len() - 2;
            let free_idx = columns.len() - 1;
            let total = columns[total_idx].parse::<u64>().with_context(|| {
                format!(
                    "invalid GPU memory.total value '{}' at line {}",
                    columns[total_idx],
                    line_idx + 1
                )
            })?;
            let free = columns[free_idx].parse::<u64>().with_context(|| {
                format!(
                    "invalid GPU memory.free value '{}' at line {}",
                    columns[free_idx],
                    line_idx + 1
                )
            })?;
            (total_idx, total, Some(free))
        } else {
            let total = columns[columns.len() - 1].parse::<u64>().with_context(|| {
                format!(
                    "invalid GPU memory.total value '{}' at line {}",
                    columns[columns.len() - 1],
                    line_idx + 1
                )
            })?;
            (columns.len() - 1, total, None)
        };
        let name = columns[1..name_columns_end].join(",");
        let name = name.trim().to_string();
        if name.is_empty() {
            bail!("missing GPU name at line {}", line_idx + 1);
        }

        devices.push(NvidiaDeviceInfo {
            index,
            name,
            memory_total_mib,
            memory_free_mib,
        });
    }

    Ok(devices)
}

fn derive_memory_budget_mib(memory_total_mib: u64, memory_free_mib: Option<u64>) -> u64 {
    let total = memory_total_mib.max(1);
    let reserve = (total / NVIDIA_MEMORY_RESERVE_RATIO_DENOM)
        .max(NVIDIA_MEMORY_RESERVE_MIB_FLOOR)
        .min(total.saturating_sub(1).max(1));

    let available = memory_free_mib.unwrap_or(total).max(1).min(total);
    available.saturating_sub(reserve).max(1)
}

fn try_allocate_cuda_buffers(
    stream: &Arc<CudaStream>,
    m_blocks: u32,
    lanes: usize,
    hashes_per_launch_per_lane: usize,
) -> std::result::Result<(CudaSlice<u64>, CudaSlice<u64>, CudaSlice<u64>), DriverError> {
    let lane_words_u64 = u64::from(m_blocks).saturating_mul(128);
    let total_lane_words_u64 = lane_words_u64.saturating_mul(lanes as u64);
    let total_lane_words = usize::try_from(total_lane_words_u64).unwrap_or(usize::MAX);
    let launch_capacity = lanes
        .saturating_mul(hashes_per_launch_per_lane.max(1))
        .max(1);

    let lane_memory = unsafe { stream.alloc::<u64>(total_lane_words) }?;
    let seed_blocks = unsafe { stream.alloc::<u64>(launch_capacity.saturating_mul(256)) }?;
    let last_blocks = unsafe { stream.alloc::<u64>(launch_capacity.saturating_mul(128)) }?;
    Ok((lane_memory, seed_blocks, last_blocks))
}

fn probe_cuda_lane_memory(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    lane_memory: &mut CudaSlice<u64>,
    lane_stride_words: u64,
    lanes: usize,
) -> std::result::Result<(), DriverError> {
    let lanes_u32 = u32::try_from(lanes).unwrap_or(u32::MAX);
    let cfg = LaunchConfig {
        grid_dim: (lanes_u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut launch = stream.launch_builder(kernel);
        launch
            .arg(lane_memory)
            .arg(&lanes_u32)
            .arg(&lane_stride_words);
        launch.launch(cfg)?;
    }

    stream.synchronize()?;
    Ok(())
}

fn build_nvidia_autotune_key(selected: &NvidiaDeviceInfo) -> NvidiaAutotuneKey {
    let (m_cost_kib, t_cost) = pow_params()
        .map(|params| (params.m_cost(), params.t_cost()))
        .unwrap_or((0, 0));
    let memory_budget_mib = derive_memory_budget_mib(selected.memory_total_mib, selected.memory_free_mib);
    let (compute_cap_major, compute_cap_minor) =
        query_cuda_compute_capability(selected.index).unwrap_or((0, 0));
    NvidiaAutotuneKey {
        device_name: selected.name.clone(),
        memory_total_mib: selected.memory_total_mib,
        memory_budget_mib,
        compute_cap_major,
        compute_cap_minor,
        m_cost_kib,
        t_cost,
        kernel_threads: KERNEL_THREADS,
    }
}

fn query_cuda_compute_capability(device_index: u32) -> Option<(u32, u32)> {
    let ctx = CudaContext::new(device_index as usize).ok()?;
    let (major, minor) = ctx.compute_capability().ok()?;
    Some((u32::try_from(major).ok()?, u32::try_from(minor).ok()?))
}

fn empty_nvidia_autotune_cache() -> NvidiaAutotuneCache {
    NvidiaAutotuneCache {
        schema_version: NVIDIA_AUTOTUNE_SCHEMA_VERSION,
        records: Vec::new(),
    }
}

fn load_nvidia_autotune_cache(path: &Path) -> Option<NvidiaAutotuneCache> {
    let raw = fs::read_to_string(path).ok()?;
    let parsed = serde_json::from_str::<NvidiaAutotuneCache>(&raw).ok()?;
    if parsed.schema_version != NVIDIA_AUTOTUNE_SCHEMA_VERSION {
        return None;
    }
    Some(parsed)
}

fn load_nvidia_cached_tuning(path: &Path, key: &NvidiaAutotuneKey) -> Option<NvidiaKernelTuning> {
    let cache = load_nvidia_autotune_cache(path)?;
    let record = cache
        .records
        .iter()
        .filter(|record| &record.key == key && record.max_rregcount > 0)
        .max_by_key(|record| record.timestamp_unix_secs)?;
    Some(NvidiaKernelTuning {
        max_rregcount: record.max_rregcount,
        block_loop_unroll: record.block_loop_unroll,
    })
}

fn persist_nvidia_autotune_record(
    path: &Path,
    key: NvidiaAutotuneKey,
    tuning: NvidiaKernelTuning,
    measured_hps: f64,
    autotune_secs: u64,
    autotune_samples: u32,
) -> Result<()> {
    let mut cache = load_nvidia_autotune_cache(path).unwrap_or_else(empty_nvidia_autotune_cache);
    if cache.schema_version != NVIDIA_AUTOTUNE_SCHEMA_VERSION {
        cache = empty_nvidia_autotune_cache();
    }

    let timestamp_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let updated = NvidiaAutotuneRecord {
        key: key.clone(),
        max_rregcount: tuning.max_rregcount.max(1),
        block_loop_unroll: tuning.block_loop_unroll,
        measured_hps: if measured_hps.is_finite() {
            measured_hps.max(0.0)
        } else {
            0.0
        },
        autotune_secs: autotune_secs.max(1),
        autotune_samples: autotune_samples.max(1),
        timestamp_unix_secs,
    };
    if let Some(existing) = cache.records.iter_mut().find(|record| record.key == key) {
        *existing = updated;
    } else {
        cache.records.push(updated);
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create NVIDIA autotune cache directory '{}'",
                parent.display()
            )
        })?;
    }
    let payload = serde_json::to_string_pretty(&cache)?;
    fs::write(path, payload)
        .with_context(|| format!("failed to write NVIDIA autotune cache '{}'", path.display()))?;
    Ok(())
}

fn measure_nvidia_kernel_tuning_hps(
    selected: &NvidiaDeviceInfo,
    tuning: NvidiaKernelTuning,
    secs: u64,
    max_lanes_override: Option<usize>,
    hashes_per_launch_per_lane: u32,
) -> Result<f64> {
    let mut engine = CudaArgon2Engine::new(
        selected.index,
        selected.memory_total_mib,
        selected.memory_free_mib,
        tuning,
        max_lanes_override,
        hashes_per_launch_per_lane,
    )?;
    let header = [0u8; POW_HEADER_BASE_LEN];
    let start = Instant::now();
    let deadline = start + Duration::from_secs(secs.max(1));
    let mut nonce_cursor = 0u64;
    let mut total = 0u64;
    let mut nonces = vec![0u64; engine.max_hashes_per_launch()];

    while Instant::now() < deadline {
        let batch_hashes = engine.max_hashes_per_launch().max(1);
        if nonces.len() < batch_hashes {
            nonces.resize(batch_hashes, 0);
        }
        for nonce in nonces.iter_mut().take(batch_hashes) {
            *nonce = nonce_cursor;
            nonce_cursor = nonce_cursor.wrapping_add(1);
        }
        let done = engine.run_fill_batch(&header, &nonces[..batch_hashes])?;
        total = total.saturating_add(done as u64);
    }

    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    Ok(total as f64 / elapsed)
}

fn autotune_nvidia_kernel_tuning(
    selected: &NvidiaDeviceInfo,
    cache_path: &Path,
    autotune_secs: u64,
    autotune_samples: u32,
    max_lanes_override: Option<usize>,
    hashes_per_launch_per_lane: u32,
) -> Result<NvidiaKernelTuning> {
    let mut best: Option<(NvidiaKernelTuning, f64, f64)> = None;
    let sample_count = autotune_samples.max(1);

    for &max_rregcount in NVIDIA_AUTOTUNE_REGCAP_CANDIDATES {
        for &block_loop_unroll in NVIDIA_AUTOTUNE_LOOP_UNROLL_CANDIDATES {
            let candidate = NvidiaKernelTuning {
                max_rregcount,
                block_loop_unroll,
            };
            let mut samples = Vec::with_capacity(sample_count as usize);
            for _ in 0..sample_count {
                let measured = match measure_nvidia_kernel_tuning_hps(
                    selected,
                    candidate,
                    autotune_secs,
                    max_lanes_override,
                    hashes_per_launch_per_lane,
                ) {
                    Ok(hps) if hps.is_finite() => hps,
                    _ => continue,
                };
                samples.push(measured);
            }
            if samples.is_empty() {
                continue;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if samples.len().is_multiple_of(2) {
                let mid = samples.len() / 2;
                (samples[mid - 1] + samples[mid]) / 2.0
            } else {
                samples[samples.len() / 2]
            };
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            match best {
                Some((_, best_median, best_mean))
                    if median < best_median
                        || (median == best_median && mean <= best_mean) => {}
                _ => best = Some((candidate, median, mean)),
            }
        }
    }

    let (selected_tuning, measured_hps, _) = best.ok_or_else(|| {
        anyhow!(
            "NVIDIA autotune failed for device {} (index {})",
            selected.name,
            selected.index
        )
    })?;

    let key = build_nvidia_autotune_key(selected);
    let _ = persist_nvidia_autotune_record(
        cache_path,
        key,
        selected_tuning,
        measured_hps,
        autotune_secs,
        sample_count,
    );
    Ok(selected_tuning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be >= unix epoch")
            .as_nanos();
        path.push(format!(
            "seine-nvidia-autotune-{}-{}-{name}.json",
            std::process::id(),
            now
        ));
        path
    }

    fn test_template(work_id: u64) -> Arc<WorkTemplate> {
        Arc::new(WorkTemplate {
            work_id,
            epoch: 7,
            header_base: Arc::<[u8]>::from(vec![1u8; 92]),
            target: [0xff; 32],
            stop_at: Instant::now() + Duration::from_secs(30),
        })
    }

    #[test]
    fn parse_nvidia_smi_query_output_parses_multiple_rows() {
        let parsed = parse_nvidia_smi_query_output(
            "0, NVIDIA GeForce RTX 3080, 10240\n1, NVIDIA RTX A4000, 16384\n",
        )
        .expect("query output should parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].index, 0);
        assert_eq!(parsed[0].name, "NVIDIA GeForce RTX 3080");
        assert_eq!(parsed[0].memory_total_mib, 10_240);
        assert_eq!(parsed[1].index, 1);
        assert_eq!(parsed[1].name, "NVIDIA RTX A4000");
        assert_eq!(parsed[1].memory_total_mib, 16_384);
    }

    #[test]
    fn parse_nvidia_smi_query_output_rejects_invalid_rows() {
        let err =
            parse_nvidia_smi_query_output("abc, RTX, 8192").expect_err("invalid index should fail");
        assert!(format!("{err:#}").contains("invalid GPU index"));
    }

    #[test]
    fn nvidia_autotune_cache_round_trip_loads_latest_record() {
        let path = unique_temp_file("roundtrip");
        let key = NvidiaAutotuneKey {
            device_name: "NVIDIA GeForce RTX 3080".to_string(),
            memory_total_mib: 10_240,
            memory_budget_mib: 9_600,
            compute_cap_major: 8,
            compute_cap_minor: 6,
            m_cost_kib: 2_097_152,
            t_cost: 1,
            kernel_threads: KERNEL_THREADS,
        };
        persist_nvidia_autotune_record(
            &path,
            key.clone(),
            NvidiaKernelTuning {
                max_rregcount: 160,
                block_loop_unroll: false,
            },
            0.8,
            2,
            2,
        )
        .expect("first record should persist");
        persist_nvidia_autotune_record(
            &path,
            key.clone(),
            NvidiaKernelTuning {
                max_rregcount: 224,
                block_loop_unroll: true,
            },
            1.0,
            2,
            2,
        )
        .expect("second record should persist");

        let loaded =
            load_nvidia_cached_tuning(&path, &key).expect("cached tuning should be available");
        assert_eq!(loaded.max_rregcount, 224);
        assert!(loaded.block_loop_unroll);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn normalize_assignment_batch_preserves_contiguous_chunks() {
        let template = test_template(11);
        let assignments = vec![
            WorkAssignment {
                template: Arc::clone(&template),
                nonce_chunk: NonceChunk {
                    start_nonce: 100,
                    nonce_count: 4,
                },
            },
            WorkAssignment {
                template,
                nonce_chunk: NonceChunk {
                    start_nonce: 104,
                    nonce_count: 6,
                },
            },
        ];

        let normalized =
            normalize_assignment_batch(&assignments).expect("batch normalization should succeed");
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].nonce_chunk.start_nonce, 100);
        assert_eq!(normalized[0].nonce_chunk.nonce_count, 4);
        assert_eq!(normalized[1].nonce_chunk.start_nonce, 104);
        assert_eq!(normalized[1].nonce_chunk.nonce_count, 6);
    }

    #[test]
    fn normalize_assignment_batch_rejects_non_contiguous_chunks() {
        let template = test_template(11);
        let assignments = vec![
            WorkAssignment {
                template: Arc::clone(&template),
                nonce_chunk: NonceChunk {
                    start_nonce: 100,
                    nonce_count: 4,
                },
            },
            WorkAssignment {
                template,
                nonce_chunk: NonceChunk {
                    start_nonce: 105,
                    nonce_count: 6,
                },
            },
        ];

        let err = normalize_assignment_batch(&assignments)
            .expect_err("non-contiguous chunks should fail");
        assert!(format!("{err:#}").contains("not contiguous"));
    }

    #[test]
    fn derive_memory_budget_uses_free_vram_with_headroom() {
        let budget = derive_memory_budget_mib(10_240, Some(9_800));
        assert_eq!(budget, 9_640);
    }

    #[test]
    fn blake2b_long_matches_expected_len_behavior() {
        let mut short = [0u8; 32];
        blake2b_long(&[b"seine"], &mut short).expect("short output should hash");

        let mut long = [0u8; 96];
        blake2b_long(&[b"seine"], &mut long).expect("long output should hash");

        assert_ne!(short, [0u8; 32]);
        assert_ne!(long, [0u8; 96]);
        assert_ne!(&long[..32], &short);
    }
}
