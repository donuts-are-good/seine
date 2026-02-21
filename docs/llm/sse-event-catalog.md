# SSE Event Catalog

Endpoint: `GET /v1/events/stream`

Control-plane endpoints are open by default; no API key is required.

Each SSE frame includes:
- `id`: monotonic event sequence
- `event`: event type
- `data`: JSON payload

## Event Types

- `state.changed`
  - Payload: `{ previous, current, reason }`
- `miner.log`
  - Payload: `{ elapsed_secs, level, tag, message }`
- `backend.phase`
  - Payload: `{ backend, phase }`
  - Typical phases: `initializing`, `initialized`, `ready`, `quarantined`
- `nvidia.init.progress`
  - Payload: `{ pending_count, message }`
- `autotune.progress`
  - Payload: `{ tag, message }`
- `autotune.completed`
  - Payload: `{ tag, message }`
- `wallet.required`
  - Payload: `{ message }`
- `wallet.loaded`
  - Payload: `{ source? | message? }`
- `solution.found`
  - Payload: `{ message }`
- `submit.result`
  - Payload: `{ message, accepted? }`
- `devfee.mode`
  - Payload: `{ mode }` where mode is `user` or `dev`

## Delivery Notes

- Events are best-effort in-memory broadcast for active subscribers.
- Slow subscribers can be lagged and skip older events.
- Clients should reconnect and refresh snapshot state from `GET /v1/runtime/state`.
- Keepalive frames are emitted periodically.
