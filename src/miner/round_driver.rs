use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{anyhow, Result};
use crossbeam_channel::Receiver;

use crate::backend::{BackendEvent, BackendInstanceId};

use super::backend_executor::BackendExecutor;
use super::hash_poll::{next_backend_poll_deadline, BackendPollState};
use super::{collect_round_backend_samples, BackendRoundTelemetry, BackendSlot, MIN_EVENT_WAIT};

pub(super) struct RoundDriverStep {
    pub collected_hashes: u64,
    pub event: Option<BackendEvent>,
}

pub(super) fn drive_round_step(
    backends: &[BackendSlot],
    backend_events: &Receiver<BackendEvent>,
    backend_executor: &BackendExecutor,
    configured_hash_poll_interval: std::time::Duration,
    poll_state: &mut BackendPollState,
    round_backend_hashes: &mut BTreeMap<BackendInstanceId, u64>,
    round_backend_telemetry: &mut BTreeMap<BackendInstanceId, BackendRoundTelemetry>,
    stop_at: Instant,
    extra_deadline: Option<Instant>,
) -> Result<RoundDriverStep> {
    let collected_hashes = collect_round_backend_samples(
        backends,
        backend_executor,
        configured_hash_poll_interval,
        poll_state,
        round_backend_hashes,
        round_backend_telemetry,
    );

    let now = Instant::now();
    let next_hash_poll_at = next_backend_poll_deadline(poll_state);
    let mut wait_until = stop_at.min(next_hash_poll_at);
    if let Some(extra_deadline) = extra_deadline {
        wait_until = wait_until.min(extra_deadline);
    }
    let wait_for = wait_until
        .saturating_duration_since(now)
        .max(MIN_EVENT_WAIT);

    let mut event = None;
    crossbeam_channel::select! {
        recv(backend_events) -> backend_event => {
            event = Some(backend_event.map_err(|_| anyhow!("backend event channel closed"))?);
        }
        default(wait_for) => {}
    }

    Ok(RoundDriverStep {
        collected_hashes,
        event,
    })
}
