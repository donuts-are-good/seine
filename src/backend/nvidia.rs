use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use blocknet_pow_spec::CPU_LANE_MEMORY_BYTES;
use crossbeam_channel::{bounded, Receiver, Sender};

use crate::backend::cpu::{CpuBackend, CpuBackendTuning};
use crate::backend::{
    AssignmentSemantics, BackendCallStatus, BackendCapabilities, BackendEvent,
    BackendExecutionModel, BackendInstanceId, BackendTelemetry, BenchBackend, DeadlineSupport,
    PowBackend, PreemptionGranularity, WorkAssignment,
};
use crate::config::CpuAffinityMode;

const BACKEND_NAME: &str = "nvidia";
const EVENT_FORWARD_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
struct NvidiaDeviceInfo {
    index: u32,
    name: String,
    memory_total_mib: u64,
}

pub struct NvidiaBackend {
    instance_id: AtomicU64,
    requested_device_index: Option<u32>,
    resolved_device: RwLock<Option<NvidiaDeviceInfo>>,
    event_sink: Arc<RwLock<Option<Sender<BackendEvent>>>>,
    event_forward_handle: Mutex<Option<JoinHandle<()>>>,
    inner: CpuBackend,
}

impl NvidiaBackend {
    pub fn new(device_index: Option<u32>) -> Self {
        let inner = CpuBackend::with_tuning(
            1,
            CpuAffinityMode::Off,
            CpuBackendTuning {
                hash_batch_size: 64,
                control_check_interval_hashes: 1,
                hash_flush_interval: Duration::from_millis(50),
                event_dispatch_capacity: EVENT_FORWARD_CAPACITY,
            },
        );

        let event_sink = Arc::new(RwLock::new(None));
        let (inner_event_tx, inner_event_rx) = bounded::<BackendEvent>(EVENT_FORWARD_CAPACITY);
        inner.set_event_sink(inner_event_tx);
        let event_forward_handle =
            spawn_event_forwarder(Arc::clone(&event_sink), inner_event_rx, BACKEND_NAME);

        Self {
            instance_id: AtomicU64::new(0),
            requested_device_index: device_index,
            resolved_device: RwLock::new(None),
            event_sink,
            event_forward_handle: Mutex::new(Some(event_forward_handle)),
            inner,
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
}

impl Drop for NvidiaBackend {
    fn drop(&mut self) {
        let (tx, _rx) = bounded::<BackendEvent>(1);
        self.inner.set_event_sink(tx);
        if let Ok(mut slot) = self.event_forward_handle.lock() {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }
    }
}

impl PowBackend for NvidiaBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn lanes(&self) -> usize {
        self.inner.lanes()
    }

    fn set_instance_id(&self, id: BackendInstanceId) {
        self.instance_id.store(id, Ordering::Release);
        self.inner.set_instance_id(id);
    }

    fn set_event_sink(&self, sink: Sender<BackendEvent>) {
        if let Ok(mut slot) = self.event_sink.write() {
            *slot = Some(sink);
        }
    }

    fn start(&self) -> Result<()> {
        let selected = self.validate_or_select_device()?;
        self.inner.start().with_context(|| {
            format!(
                "failed to start NVIDIA backend for device {} ({})",
                selected.index, selected.name
            )
        })
    }

    fn stop(&self) {
        self.inner.stop();
    }

    fn assign_work(&self, work: WorkAssignment) -> Result<()> {
        self.inner.assign_work(work)
    }

    fn assign_work_with_deadline(&self, work: &WorkAssignment, deadline: Instant) -> Result<()> {
        self.inner.assign_work_with_deadline(work, deadline)
    }

    fn supports_assignment_batching(&self) -> bool {
        self.inner.supports_assignment_batching()
    }

    fn supports_true_nonblocking(&self) -> bool {
        self.inner.supports_true_nonblocking()
    }

    fn assign_work_batch(&self, work: &[WorkAssignment]) -> Result<()> {
        self.inner.assign_work_batch(work)
    }

    fn assign_work_batch_with_deadline(
        &self,
        work: &[WorkAssignment],
        deadline: Instant,
    ) -> Result<()> {
        self.inner.assign_work_batch_with_deadline(work, deadline)
    }

    fn assign_work_batch_nonblocking(&self, work: &[WorkAssignment]) -> Result<BackendCallStatus> {
        self.inner.assign_work_batch_nonblocking(work)
    }

    fn wait_for_nonblocking_progress(&self, wait_for: Duration) -> Result<()> {
        self.inner.wait_for_nonblocking_progress(wait_for)
    }

    fn cancel_work_nonblocking(&self) -> Result<BackendCallStatus> {
        self.inner.cancel_work_nonblocking()
    }

    fn fence_nonblocking(&self) -> Result<BackendCallStatus> {
        self.inner.fence_nonblocking()
    }

    fn cancel_work(&self) -> Result<()> {
        self.inner.cancel_work()
    }

    fn request_timeout_interrupt(&self) -> Result<()> {
        self.inner.request_timeout_interrupt()
    }

    fn cancel_work_with_deadline(&self, deadline: Instant) -> Result<()> {
        self.inner.cancel_work_with_deadline(deadline)
    }

    fn fence(&self) -> Result<()> {
        self.inner.fence()
    }

    fn fence_with_deadline(&self, deadline: Instant) -> Result<()> {
        self.inner.fence_with_deadline(deadline)
    }

    fn take_hashes(&self) -> u64 {
        self.inner.take_hashes()
    }

    fn take_telemetry(&self) -> BackendTelemetry {
        self.inner.take_telemetry()
    }

    fn preemption_granularity(&self) -> PreemptionGranularity {
        self.inner.preemption_granularity()
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            preferred_iters_per_lane: Some(1),
            preferred_allocation_iters_per_lane: Some(1),
            preferred_hash_poll_interval: Some(Duration::from_millis(50)),
            preferred_assignment_timeout: None,
            preferred_control_timeout: None,
            preferred_assignment_timeout_strikes: None,
            preferred_worker_queue_depth: None,
            max_inflight_assignments: 1,
            deadline_support: DeadlineSupport::Cooperative,
            assignment_semantics: AssignmentSemantics::Replace,
            execution_model: BackendExecutionModel::Blocking,
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
        self.validate_or_select_device()?;
        self.inner.kernel_bench(seconds, shutdown)
    }
}

fn spawn_event_forwarder(
    sink: Arc<RwLock<Option<Sender<BackendEvent>>>>,
    rx: Receiver<BackendEvent>,
    backend_name: &'static str,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            let translated = match event {
                BackendEvent::Solution(mut solution) => {
                    solution.backend = backend_name;
                    BackendEvent::Solution(solution)
                }
                BackendEvent::Error {
                    backend_id,
                    message,
                    ..
                } => BackendEvent::Error {
                    backend_id,
                    backend: backend_name,
                    message,
                },
            };

            let outbound = match sink.read() {
                Ok(slot) => slot.clone(),
                Err(_) => None,
            };
            let Some(outbound) = outbound else {
                continue;
            };
            if outbound.send(translated).is_err() {
                break;
            }
        }
    })
}

fn query_nvidia_devices() -> Result<Vec<NvidiaDeviceInfo>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
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
        let memory_total_mib = columns[columns.len() - 1].parse::<u64>().with_context(|| {
            format!(
                "invalid GPU memory value '{}' at line {}",
                columns[columns.len() - 1],
                line_idx + 1
            )
        })?;
        let name = columns[1..columns.len() - 1].join(",");
        let name = name.trim().to_string();
        if name.is_empty() {
            bail!("missing GPU name at line {}", line_idx + 1);
        }

        devices.push(NvidiaDeviceInfo {
            index,
            name,
            memory_total_mib,
        });
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
