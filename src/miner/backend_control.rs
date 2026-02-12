use std::collections::BTreeMap;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError};

use crate::backend::{BackendEvent, BackendInstanceId};

use super::ui::{error, info, warn};
use super::{
    backend_names, remove_backend_by_id, BackendSlot, RuntimeBackendEventAction, RuntimeMode,
};

type BackendFailureMap = BTreeMap<BackendInstanceId, (&'static str, Vec<String>)>;

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
    slot: BackendSlot,
    elapsed: Duration,
    result: Result<()>,
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
    let mut metadata = Vec::with_capacity(expected);
    let (outcome_tx, outcome_rx) = bounded::<BackendControlOutcome>(expected.max(1));

    for (idx, slot) in slots.into_iter().enumerate() {
        let backend_id = slot.id;
        let backend = slot.backend.name();
        metadata.push((backend_id, backend));

        let outcome_tx = outcome_tx.clone();
        thread::spawn(move || {
            let started = Instant::now();
            let deadline = started + timeout;
            let result = match phase {
                BackendControlPhase::Cancel => slot.backend.cancel_work_with_deadline(deadline),
                BackendControlPhase::Fence => slot.backend.fence_with_deadline(deadline),
            };
            let elapsed = started.elapsed();
            let _ = outcome_tx.send(BackendControlOutcome {
                idx,
                slot,
                elapsed,
                result,
            });
        });
    }
    drop(outcome_tx);

    let deadline = Instant::now() + timeout;
    let mut outcomes: Vec<Option<BackendControlOutcome>> =
        std::iter::repeat_with(|| None).take(expected).collect();
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

    let mut survivors = Vec::new();
    let mut failures = BackendFailureMap::new();
    let action_label = phase.action_label();

    for idx in 0..expected {
        let (backend_id, backend) = metadata[idx];
        match outcomes[idx].take() {
            Some(mut outcome) => {
                if outcome.elapsed > timeout {
                    outcome.slot.backend.stop();
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
                    Ok(()) => survivors.push((idx, outcome.slot)),
                    Err(err) => {
                        outcome.slot.backend.stop();
                        failures
                            .entry(backend_id)
                            .or_insert_with(|| (backend, Vec::new()))
                            .1
                            .push(format!("{action_label} failed: {err:#}"));
                    }
                }
            }
            None => {
                failures
                    .entry(backend_id)
                    .or_insert_with(|| (backend, Vec::new()))
                    .1
                    .push(format!(
                        "{action_label} timed out after {}ms; backend detached",
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

fn merge_backend_failures(failures: &mut BackendFailureMap, additional: BackendFailureMap) {
    for (backend_id, (backend, messages)) in additional {
        failures
            .entry(backend_id)
            .or_insert_with(|| (backend, Vec::new()))
            .1
            .extend(messages);
    }
}
