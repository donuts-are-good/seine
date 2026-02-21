# Seine API + LLM Docs Implementation Plan

## Summary
Implement a first-class local control API exposing nearly all miner frontend/runtime capabilities with sparse optional configuration, support both embedded and service modes, keep delayed NVIDIA/autotune observable, and preserve non-removable dev fee behavior.

## Core API Direction
- Protocol: REST + SSE
- Versioning: `/v1` path versioning
- Modes: embedded API server + explicit `--service` mode
- Service startup: idle by default; mining starts via API command
- Config input: sparse patch over defaults
- Runtime config changes: safe live toggles only
- Auth default: control API is open (no API key required)
- Bind default: loopback only; non-loopback requires explicit unsafe flag
- CORS: configurable allow-origin value
- Metrics: include `/metrics`

## Required Runtime/API Capabilities
- Start/stop/restart mining lifecycle
- Runtime state snapshots
- Effective/default config inspection
- Backend status/telemetry visibility
- Structured event streaming (SSE)
- Wallet unlock flow for API mode
- NVIDIA delayed init progress visibility
- CPU/NVIDIA autotune progress + completion visibility
- Dev fee visibility as read-only state (immutable)

## Planned Endpoints (v1)
- `GET /v1/health`
- `GET /v1/runtime/state`
- `GET /v1/runtime/config/defaults`
- `GET /v1/runtime/config/effective`
- `POST /v1/miner/start`
- `POST /v1/miner/stop`
- `POST /v1/miner/restart`
- `PATCH /v1/miner/live-config`
- `GET /v1/backends`
- `POST /v1/wallet/unlock`
- `GET /v1/events/stream`
- `GET /metrics`

## Required SSE Event Types
- `state.changed`
- `miner.log`
- `backend.phase`
- `nvidia.init.progress`
- `autotune.progress`
- `autotune.completed`
- `round.summary`
- `solution.found`
- `submit.result`
- `devfee.mode`
- `wallet.required`
- `wallet.loaded`

## Dev Fee Policy (Immutable)
- Preserve existing hardcoded dev fee percent/address and scheduling behavior
- Expose dev fee state as read-only in API + SSE
- Reject any config attempts to alter or disable dev fee

## Internal Architecture Changes
- Rename daemon HTTP client module for clarity (`src/api.rs` -> `src/daemon_api.rs`)
- Add API server module (`src/control_api/`) using `axum + tokio`
- Add miner supervisor/session controller for lifecycle control
- Add runtime state store + telemetry/event bus for API/SSE/metrics
- Keep hashing backend execution logic unchanged

## Sparse Config + Live Patch Rules
- Start request accepts optional overrides only
- Omitted fields resolve from current defaults/autodetection
- Live-patch only for safe toggles
- Live-patch queues for next start when runtime is active
- Return `requires_restart` for backend topology/autotune structural changes while running

## LLM Docs Deliverables (Required)
- `docs/openapi/seine-api-v1.yaml` (canonical source of truth)
- `docs/schemas/*.json` (request/response/event schemas)
- `docs/llm/sse-event-catalog.md`
- `docs/llm/integration-playbook.md`
- `docs/llm/prompt-safety-guide.md`
- `docs/llm/examples/` comprehensive workflows

## LLM Example Coverage
- Service boot idle -> start mining
- Sparse config start
- Mixed CPU/NVIDIA with progress stream
- Wallet unlock and resume
- Live-safe patch flow
- Restart-required change flow
- Backend quarantine handling
- Graceful stop verification

## Validation + CI
- OpenAPI/schema lint and validation
- Example payload conformance checks
- API-doc drift guard
- Fail CI on undocumented public endpoint/event additions

## Implementation Phases
1. Supervisor + runtime state + event bus scaffolding
2. API server skeleton + health/state/SSE
3. Service mode idle start + lifecycle endpoints
4. Sparse config resolver + effective/default config endpoints
5. Wallet unlock endpoint + wallet-required signaling
6. Structured NVIDIA/autotune/backend phase events
7. Metrics endpoint
8. LLM docs/openapi/schemas/examples + CI checks
