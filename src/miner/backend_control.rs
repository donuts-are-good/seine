use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use crossbeam_channel::{bounded, unbounded, Receiver, RecvTimeoutError, Sender};

use crate::backend::{BackendEvent, BackendInstanceId, PowBackend};

use super::ui::{error, info, warn};
use super::{
    backend_names, remove_backend_by_id, BackendSlot, RuntimeBackendEventAction, RuntimeMode,
};

type BackendFailureMap = BTreeMap<BackendInstanceId, (&'static str, Vec<String>)>;
type BackendControlDispatchError = (usize, BackendInstanceId, &'static str, Arc<dyn PowBackend>);

pub(super) fn cancel_backend_slots(
    backends: &mut Vec<BackendSlot>,
    mode: RuntimeMode,
    control_timeout: Duration,
) -> Result<RuntimeBackendEventAction> {
    control_backend_slots(backends, mode, false, control_timeout)
}

pub(super) fn quiesce_backend_slots(
    backends: &mut Vec<BackendSlot>,
    mode: RuntimeMode,
    control_timeout: Duration,
) -> Result<RuntimeBackendEventAction> {
    control_backend_slots(backends, mode, true, control_timeout)
}

pub(super) fn handle_runtime_backend_event(
    event: BackendEvent,
    epoch: u64,
    backends: &mut Vec<BackendSlot>,
    mode: RuntimeMode,
) -> Result<(
    RuntimeBackendEventAction,
    Option<crate::backend::MiningSolution>,
)> {
    match event {
        BackendEvent::Solution(solution) => {
            let backend_active = backends.iter().any(|slot| slot.id == solution.backend_id);
            if solution.epoch == epoch && backend_active {
                match mode {
                    RuntimeMode::Mining => {
                        return Ok((RuntimeBackendEventAction::None, Some(solution)));
                    }
                    RuntimeMode::Bench => {
                        info(
                            "BENCH",
                            format!(
                                "unexpected solution from {}#{} at nonce={}",
                                solution.backend, solution.backend_id, solution.nonce
                            ),
                        );
                    }
                }
            }
            Ok((RuntimeBackendEventAction::None, None))
        }
        BackendEvent::Error {
            backend_id,
            backend,
            message,
        } => {
            match mode {
                RuntimeMode::Mining => {
                    error(
                        "BACKEND",
                        format!("{backend}#{backend_id} runtime error: {message}"),
                    );
                }
                RuntimeMode::Bench => {
                    error(
                        "BENCH",
                        format!("backend '{backend}#{backend_id}' runtime error: {message}"),
                    );
                }
            }

            let removed = remove_backend_by_id(backends, backend_id);
            if removed {
                if backends.is_empty() {
                    match mode {
                        RuntimeMode::Mining => {
                            bail!(
                                "all mining backends are unavailable after failure in '{backend}#{backend_id}'"
                            );
                        }
                        RuntimeMode::Bench => {
                            bail!(
                                "all benchmark backends are unavailable after failure in '{backend}#{backend_id}'"
                            );
                        }
                    }
                }

                match mode {
                    RuntimeMode::Mining => {
                        warn(
                            "BACKEND",
                            format!(
                                "quarantined {backend}#{backend_id}; continuing with {}",
                                backend_names(backends)
                            ),
                        );
                    }
                    RuntimeMode::Bench => {
                        warn(
                            "BENCH",
                            format!(
                                "quarantined {backend}#{backend_id}; remaining backends={}",
                                backend_names(backends)
                            ),
                        );
                    }
                }
                Ok((RuntimeBackendEventAction::TopologyChanged, None))
            } else {
                if mode == RuntimeMode::Mining {
                    warn(
                        "BACKEND",
                        format!("ignoring error from unavailable backend '{backend}#{backend_id}'"),
                    );
                }
                Ok((RuntimeBackendEventAction::None, None))
            }
        }
    }
}

pub(super) fn drain_runtime_backend_events(
    backend_events: &Receiver<BackendEvent>,
    epoch: u64,
    backends: &mut Vec<BackendSlot>,
    mode: RuntimeMode,
) -> Result<(
    RuntimeBackendEventAction,
    Option<crate::backend::MiningSolution>,
)> {
    let mut action = RuntimeBackendEventAction::None;
    let mut solution = None;
    while let Ok(event) = backend_events.try_recv() {
        let (event_action, maybe_solution) =
            handle_runtime_backend_event(event, epoch, backends, mode)?;
        if event_action == RuntimeBackendEventAction::TopologyChanged {
            action = RuntimeBackendEventAction::TopologyChanged;
        }
        if solution.is_none() {
            solution = maybe_solution;
        }
    }
    Ok((action, solution))
}

fn control_backend_slots(
    backends: &mut Vec<BackendSlot>,
    mode: RuntimeMode,
    include_fence: bool,
    control_timeout: Duration,
) -> Result<RuntimeBackendEventAction> {
    if backends.is_empty() {
        return Ok(RuntimeBackendEventAction::None);
    }

    let timeout = control_timeout.max(Duration::from_millis(1));
    let mut failures = BackendFailureMap::new();

    let (mut survivors, cancel_failures) = run_backend_control_phase(
        std::mem::take(backends),
        BackendControlPhase::Cancel,
        timeout,
    );
    merge_backend_failures(&mut failures, cancel_failures);

    if include_fence && !survivors.is_empty() {
        let (after_fence, fence_failures) =
            run_backend_control_phase(survivors, BackendControlPhase::Fence, timeout);
        survivors = after_fence;
        merge_backend_failures(&mut failures, fence_failures);
    }

    *backends = survivors;
    prune_backend_control_workers(backends);

    if failures.is_empty() {
        return Ok(RuntimeBackendEventAction::None);
    }

    for (backend_id, (backend, messages)) in &failures {
        let details = messages.join(" | ");
        match mode {
            RuntimeMode::Mining => {
                error(
                    "BACKEND",
                    format!("{backend}#{backend_id} control error: {details}"),
                );
            }
            RuntimeMode::Bench => {
                error(
                    "BENCH",
                    format!("backend '{backend}#{backend_id}' control error: {details}"),
                );
            }
        }
    }

    if backends.is_empty() {
        let failed = failures
            .iter()
            .map(|(backend_id, (backend, _))| format!("{backend}#{backend_id}"))
            .collect::<Vec<_>>()
            .join(",");
        match mode {
            RuntimeMode::Mining => {
                bail!("all mining backends are unavailable after control failure in {failed}");
            }
            RuntimeMode::Bench => {
                bail!("all benchmark backends are unavailable after control failure in {failed}");
            }
        }
    }

    let remaining = backend_names(backends);
    for (backend_id, (backend, _)) in &failures {
        match mode {
            RuntimeMode::Mining => {
                warn(
                    "BACKEND",
                    format!(
                        "quarantined {backend}#{backend_id} after control failure; continuing with {remaining}"
                    ),
                );
            }
            RuntimeMode::Bench => {
                warn(
                    "BENCH",
                    format!(
                        "quarantined {backend}#{backend_id} after control failure; remaining backends={remaining}"
                    ),
                );
            }
        }
    }

    Ok(RuntimeBackendEventAction::TopologyChanged)
}

#[derive(Debug, Clone, Copy)]
enum BackendControlPhase {
    Cancel,
    Fence,
}

impl BackendControlPhase {
    fn action_label(self) -> &'static str {
        match self {
            Self::Cancel => "cancel",
            Self::Fence => "fence",
        }
    }
}

struct BackendControlOutcome {
    idx: usize,
    elapsed: Duration,
    result: Result<()>,
}

enum BackendControlCommand {
    Run {
        idx: usize,
        backend_id: BackendInstanceId,
        backend: &'static str,
        backend_handle: Arc<dyn PowBackend>,
        phase: BackendControlPhase,
        timeout: Duration,
        outcome_tx: Sender<BackendControlOutcome>,
    },
}

#[derive(Clone)]
struct BackendControlWorker {
    tx: Sender<BackendControlCommand>,
}

static BACKEND_CONTROL_WORKERS: OnceLock<Mutex<BTreeMap<BackendInstanceId, BackendControlWorker>>> =
    OnceLock::new();

fn backend_control_worker_registry(
) -> &'static Mutex<BTreeMap<BackendInstanceId, BackendControlWorker>> {
    BACKEND_CONTROL_WORKERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn spawn_backend_control_worker(
    backend_id: BackendInstanceId,
    backend: &'static str,
) -> Option<BackendControlWorker> {
    let (cmd_tx, cmd_rx) = unbounded::<BackendControlCommand>();
    let thread_name = format!("seine-control-{backend}-{backend_id}");
    let spawn_result = thread::Builder::new().name(thread_name).spawn(move || {
        while let Ok(command) = cmd_rx.recv() {
            match command {
                BackendControlCommand::Run {
                    idx,
                    backend_id,
                    backend,
                    backend_handle,
                    phase,
                    timeout,
                    outcome_tx,
                } => run_backend_control_task(
                    idx,
                    backend_id,
                    backend,
                    backend_handle,
                    phase,
                    timeout,
                    outcome_tx,
                ),
            }
        }
    });

    if spawn_result.is_err() {
        return None;
    }
    Some(BackendControlWorker { tx: cmd_tx })
}

fn worker_sender_for_backend_control(
    backend_id: BackendInstanceId,
    backend: &'static str,
) -> Option<Sender<BackendControlCommand>> {
    let mut registry = backend_control_worker_registry().lock().ok()?;
    if let Some(worker) = registry.get(&backend_id) {
        return Some(worker.tx.clone());
    }
    let worker = spawn_backend_control_worker(backend_id, backend)?;
    let sender = worker.tx.clone();
    registry.insert(backend_id, worker);
    Some(sender)
}

pub(super) fn clear_backend_control_workers() {
    if let Ok(mut registry) = backend_control_worker_registry().lock() {
        registry.clear();
    }
}

pub(super) fn prune_backend_control_workers(backends: &[BackendSlot]) {
    let active_ids = backends
        .iter()
        .map(|slot| slot.id)
        .collect::<std::collections::BTreeSet<_>>();
    if let Ok(mut registry) = backend_control_worker_registry().lock() {
        registry.retain(|backend_id, _| active_ids.contains(backend_id));
    }
}

fn remove_backend_control_worker(backend_id: BackendInstanceId) {
    if let Ok(mut registry) = backend_control_worker_registry().lock() {
        registry.remove(&backend_id);
    }
}

fn quarantine_backend(backend: Arc<dyn PowBackend>) {
    let detached = Arc::clone(&backend);
    if thread::Builder::new()
        .name("seine-backend-quarantine".to_string())
        .spawn(move || detached.stop())
        .is_err()
    {
        warn(
            "BACKEND",
            "failed to spawn backend quarantine worker; skipping synchronous stop to avoid runtime stall",
        );
    }
}

fn dispatch_backend_control_task(
    idx: usize,
    backend_id: BackendInstanceId,
    backend: &'static str,
    backend_handle: Arc<dyn PowBackend>,
    phase: BackendControlPhase,
    timeout: Duration,
    outcome_tx: Sender<BackendControlOutcome>,
) -> std::result::Result<(), BackendControlDispatchError> {
    let Some(worker_tx) = worker_sender_for_backend_control(backend_id, backend) else {
        return Err((idx, backend_id, backend, backend_handle));
    };

    let command = BackendControlCommand::Run {
        idx,
        backend_id,
        backend,
        backend_handle,
        phase,
        timeout,
        outcome_tx,
    };
    match worker_tx.send(command) {
        Ok(()) => Ok(()),
        Err(send_err) => {
            if let Ok(mut registry) = backend_control_worker_registry().lock() {
                registry.remove(&backend_id);
            }
            match send_err.0 {
                BackendControlCommand::Run {
                    idx,
                    backend_id,
                    backend,
                    backend_handle,
                    ..
                } => Err((idx, backend_id, backend, backend_handle)),
            }
        }
    }
}

fn run_backend_control_phase(
    slots: Vec<BackendSlot>,
    phase: BackendControlPhase,
    timeout: Duration,
) -> (Vec<BackendSlot>, BackendFailureMap) {
    if slots.is_empty() {
        return (Vec::new(), BackendFailureMap::new());
    }

    let timeout = timeout.max(Duration::from_millis(1));
    let expected = slots.len();
    let mut slots_by_idx: Vec<Option<BackendSlot>> = slots.into_iter().map(Some).collect();
    let mut outcomes: Vec<Option<BackendControlOutcome>> =
        std::iter::repeat_with(|| None).take(expected).collect();
    let (outcome_tx, outcome_rx) = bounded::<BackendControlOutcome>(expected.max(1));

    for (idx, slot_opt) in slots_by_idx.iter().enumerate().take(expected) {
        let Some(slot) = slot_opt.as_ref() else {
            continue;
        };
        let backend_id = slot.id;
        let backend = slot.backend.name();
        let backend_handle = Arc::clone(&slot.backend);
        if let Err((idx, backend_id, backend, backend_handle)) = dispatch_backend_control_task(
            idx,
            backend_id,
            backend,
            Arc::clone(&backend_handle),
            phase,
            timeout,
            outcome_tx.clone(),
        ) {
            run_backend_control_task(
                idx,
                backend_id,
                backend,
                backend_handle,
                phase,
                timeout,
                outcome_tx.clone(),
            );
        }
    }
    drop(outcome_tx);

    let deadline = Instant::now() + timeout;
    let mut received = 0usize;
    while received < expected {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        match outcome_rx.recv_timeout(deadline.saturating_duration_since(now)) {
            Ok(outcome) => {
                let outcome_idx = outcome.idx;
                if outcomes[outcome_idx].is_none() {
                    received = received.saturating_add(1);
                }
                outcomes[outcome_idx] = Some(outcome);
            }
            Err(RecvTimeoutError::Timeout) => break,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    while let Ok(outcome) = outcome_rx.try_recv() {
        let outcome_idx = outcome.idx;
        if outcomes[outcome_idx].is_none() {
            received = received.saturating_add(1);
        }
        outcomes[outcome_idx] = Some(outcome);
    }

    let mut survivors = Vec::new();
    let mut failures = BackendFailureMap::new();
    let action_label = phase.action_label();

    for (idx, outcome_slot) in outcomes.iter_mut().enumerate().take(expected) {
        let Some(slot) = slots_by_idx.get_mut(idx).and_then(Option::take) else {
            continue;
        };
        let backend_id = slot.id;
        let backend = slot.backend.name();
        match outcome_slot.take() {
            Some(outcome) => {
                if outcome.elapsed > timeout {
                    quarantine_backend(Arc::clone(&slot.backend));
                    remove_backend_control_worker(backend_id);
                    failures
                        .entry(backend_id)
                        .or_insert_with(|| (backend, Vec::new()))
                        .1
                        .push(format!(
                            "{action_label} timed out after {}ms (limit={}ms)",
                            outcome.elapsed.as_millis(),
                            timeout.as_millis()
                        ));
                    continue;
                }

                match outcome.result {
                    Ok(()) => survivors.push((idx, slot)),
                    Err(err) => {
                        quarantine_backend(Arc::clone(&slot.backend));
                        remove_backend_control_worker(backend_id);
                        failures
                            .entry(backend_id)
                            .or_insert_with(|| (backend, Vec::new()))
                            .1
                            .push(format!("{action_label} failed: {err:#}"));
                    }
                }
            }
            None => {
                quarantine_backend(Arc::clone(&slot.backend));
                remove_backend_control_worker(backend_id);
                failures
                    .entry(backend_id)
                    .or_insert_with(|| (backend, Vec::new()))
                    .1
                    .push(format!(
                        "{action_label} timed out after {}ms; backend quarantined",
                        timeout.as_millis()
                    ));
            }
        }
    }

    survivors.sort_by_key(|(idx, _)| *idx);
    (
        survivors
            .into_iter()
            .map(|(_, slot)| slot)
            .collect::<Vec<_>>(),
        failures,
    )
}

fn run_backend_control_task(
    idx: usize,
    _backend_id: BackendInstanceId,
    _backend: &'static str,
    backend_handle: Arc<dyn PowBackend>,
    phase: BackendControlPhase,
    timeout: Duration,
    outcome_tx: crossbeam_channel::Sender<BackendControlOutcome>,
) {
    let started = Instant::now();
    let deadline = started + timeout;
    let result = match panic::catch_unwind(AssertUnwindSafe(|| match phase {
        BackendControlPhase::Cancel => backend_handle.cancel_work_with_deadline(deadline),
        BackendControlPhase::Fence => backend_handle.fence_with_deadline(deadline),
    })) {
        Ok(result) => result,
        Err(_) => Err(anyhow!("{} task panicked", phase.action_label())),
    };
    let elapsed = started.elapsed();
    let send_result = outcome_tx.send(BackendControlOutcome {
        idx,
        elapsed,
        result,
    });
    if send_result.is_err() {
        quarantine_backend(backend_handle);
    }
}

fn merge_backend_failures(failures: &mut BackendFailureMap, additional: BackendFailureMap) {
    for (backend_id, (backend, messages)) in additional {
        failures
            .entry(backend_id)
            .or_insert_with(|| (backend, Vec::new()))
            .1
            .extend(messages);
    }
}
