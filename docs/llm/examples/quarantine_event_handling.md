# Example: Quarantine Event Handling

1. Subscribe to `GET /v1/events/stream`.
2. Detect `backend.phase` events with `phase=quarantined`.
3. Read `GET /v1/runtime/state` and `GET /v1/backends`.
4. Decide whether to continue or restart with revised config.
