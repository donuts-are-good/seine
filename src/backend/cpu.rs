use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use argon2::{Algorithm, Argon2, Version};
use blocknet_pow_spec::{pow_params, POW_OUTPUT_LEN};
use crossbeam_channel::Sender;

use crate::backend::{BackendEvent, MiningSolution, MiningWork, PowBackend};
use crate::types::hash_meets_target;

const IDLE_SLEEP: Duration = Duration::from_millis(2);
const STALE_SLEEP: Duration = Duration::from_millis(1);
const HASH_EVENT_BATCH: u64 = 64;

struct Shared {
    started: AtomicBool,
    shutdown: AtomicBool,
    current_epoch: AtomicU64,
    solved_epoch: AtomicU64,
    work: RwLock<Option<Arc<MiningWork>>>,
    event_sink: RwLock<Option<Sender<BackendEvent>>>,
}

pub struct CpuBackend {
    threads: usize,
    shared: Arc<Shared>,
    worker_handles: Vec<JoinHandle<()>>,
}

impl CpuBackend {
    pub fn new(threads: usize) -> Self {
        Self {
            threads,
            shared: Arc::new(Shared {
                started: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                current_epoch: AtomicU64::new(0),
                solved_epoch: AtomicU64::new(0),
                work: RwLock::new(None),
                event_sink: RwLock::new(None),
            }),
            worker_handles: Vec::new(),
        }
    }
}

impl PowBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn lanes(&self) -> usize {
        self.threads.max(1)
    }

    fn set_event_sink(&mut self, sink: Sender<BackendEvent>) {
        if let Ok(mut slot) = self.shared.event_sink.write() {
            *slot = Some(sink);
        }
    }

    fn start(&mut self) -> Result<()> {
        if self.shared.started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        self.shared.shutdown.store(false, Ordering::SeqCst);
        self.shared.current_epoch.store(0, Ordering::SeqCst);
        self.shared.solved_epoch.store(0, Ordering::SeqCst);

        if let Ok(mut work) = self.shared.work.write() {
            *work = None;
        }

        for thread_idx in 0..self.threads.max(1) {
            let shared = Arc::clone(&self.shared);
            self.worker_handles
                .push(thread::spawn(move || cpu_worker_loop(shared, thread_idx)));
        }

        Ok(())
    }

    fn stop(&mut self) {
        if !self.shared.started.swap(false, Ordering::SeqCst) {
            return;
        }

        self.shared.shutdown.store(true, Ordering::SeqCst);

        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }

        if let Ok(mut work) = self.shared.work.write() {
            *work = None;
        }

        self.shared.current_epoch.store(0, Ordering::SeqCst);
        self.shared.solved_epoch.store(0, Ordering::SeqCst);
    }

    fn set_work(&self, work: MiningWork) -> Result<()> {
        if !self.shared.started.load(Ordering::SeqCst) {
            return Err(anyhow!("CPU backend is not started"));
        }

        self.shared.solved_epoch.store(0, Ordering::SeqCst);
        let epoch = work.epoch;
        let shared_work = Arc::new(work);
        {
            let mut slot = self
                .shared
                .work
                .write()
                .map_err(|_| anyhow!("CPU work lock poisoned"))?;
            *slot = Some(shared_work);
        }
        self.shared.current_epoch.store(epoch, Ordering::SeqCst);
        Ok(())
    }

    fn kernel_bench(&self, seconds: u64, shutdown: &AtomicBool) -> Result<u64> {
        let lanes = self.threads.max(1);
        let stop_at = Instant::now() + Duration::from_secs(seconds.max(1));
        let total_hashes = AtomicU64::new(0);

        thread::scope(|scope| {
            for lane in 0..lanes {
                let total_hashes = &total_hashes;
                scope.spawn(move || {
                    let params = match pow_params() {
                        Ok(p) => p,
                        Err(_) => return,
                    };
                    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

                    let mut header_base = [0u8; blocknet_pow_spec::POW_HEADER_BASE_LEN];
                    for (i, byte) in header_base.iter_mut().enumerate() {
                        *byte = (i as u8).wrapping_mul(31).wrapping_add(7);
                    }

                    let mut output = [0u8; POW_OUTPUT_LEN];
                    let mut nonce = lane as u64;
                    let mut local_hashes = 0u64;
                    let stride = lanes as u64;

                    while Instant::now() < stop_at && !shutdown.load(Ordering::Relaxed) {
                        let nonce_bytes = nonce.to_le_bytes();
                        if argon2
                            .hash_password_into(&nonce_bytes, &header_base, &mut output)
                            .is_err()
                        {
                            break;
                        }
                        nonce = nonce.wrapping_add(stride);
                        local_hashes = local_hashes.saturating_add(1);
                    }

                    total_hashes.fetch_add(local_hashes, Ordering::Relaxed);
                });
            }
        });

        Ok(total_hashes.load(Ordering::Relaxed))
    }
}

fn cpu_worker_loop(shared: Arc<Shared>, thread_idx: usize) {
    let params = match pow_params() {
        Ok(p) => p,
        Err(_) => {
            emit_error(
                &shared,
                format!("cpu thread {thread_idx}: invalid Argon2 params"),
            );
            shared.shutdown.store(true, Ordering::SeqCst);
            return;
        }
    };

    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut output = [0u8; POW_OUTPUT_LEN];
    let mut local_epoch = 0u64;
    let mut local_work: Option<Arc<MiningWork>> = None;
    let mut nonce = 0u64;
    let mut hash_batch = 0u64;

    loop {
        if shared.shutdown.load(Ordering::Relaxed) {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
            break;
        }

        let current_epoch = shared.current_epoch.load(Ordering::Acquire);
        if current_epoch == 0 {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
            thread::sleep(IDLE_SLEEP);
            continue;
        }

        if current_epoch != local_epoch {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
            let next_work = match shared.work.read() {
                Ok(slot) => slot.clone(),
                Err(_) => {
                    emit_error(
                        &shared,
                        format!("cpu thread {thread_idx}: work lock poisoned"),
                    );
                    shared.shutdown.store(true, Ordering::SeqCst);
                    break;
                }
            };

            let Some(work) = next_work else {
                thread::sleep(IDLE_SLEEP);
                continue;
            };
            if work.epoch != current_epoch {
                thread::yield_now();
                continue;
            }

            nonce = work
                .start_nonce
                .wrapping_add(work.lane_offset)
                .wrapping_add(thread_idx as u64);
            local_epoch = current_epoch;
            local_work = Some(work);
        }

        let Some(work) = local_work.as_ref() else {
            thread::sleep(IDLE_SLEEP);
            continue;
        };

        if shared.solved_epoch.load(Ordering::Relaxed) == local_epoch
            || Instant::now() >= work.stop_at
        {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
            thread::sleep(STALE_SLEEP);
            continue;
        }

        let nonce_bytes = nonce.to_le_bytes();
        if argon2
            .hash_password_into(&nonce_bytes, &work.header_base, &mut output)
            .is_err()
        {
            emit_error(
                &shared,
                format!("cpu thread {thread_idx}: hash_password_into failed"),
            );
            shared.shutdown.store(true, Ordering::SeqCst);
            break;
        }

        hash_batch = hash_batch.saturating_add(1);
        if hash_batch >= HASH_EVENT_BATCH {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
        }

        if hash_meets_target(&output, &work.target)
            && shared
                .solved_epoch
                .compare_exchange(0, local_epoch, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            flush_hash_batch(&shared, local_epoch, &mut hash_batch);
            emit_event(
                &shared,
                BackendEvent::Solution(MiningSolution {
                    epoch: local_epoch,
                    nonce,
                    backend: "cpu".to_string(),
                }),
            );
        }

        nonce = nonce.wrapping_add(work.global_stride.max(1));
    }
}

fn flush_hash_batch(shared: &Shared, epoch: u64, hash_batch: &mut u64) {
    if epoch == 0 || *hash_batch == 0 {
        return;
    }
    let count = *hash_batch;
    *hash_batch = 0;
    emit_event(
        shared,
        BackendEvent::Hashes {
            backend: "cpu",
            epoch,
            count,
        },
    );
}

fn emit_error(shared: &Shared, message: String) {
    emit_event(
        shared,
        BackendEvent::Error {
            backend: "cpu",
            message,
        },
    );
}

fn emit_event(shared: &Shared, event: BackendEvent) {
    let tx = match shared.event_sink.read() {
        Ok(slot) => slot.clone(),
        Err(_) => None,
    };
    if let Some(tx) = tx {
        let _ = tx.send(event);
    }
}
