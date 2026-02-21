# Prompt Safety Guide (LLM Integrators)

## Immutable Dev Fee Rule

- Dev fee behavior is mandatory and enforced in runtime.
- API clients must treat dev fee settings as read-only.
- Any attempt to remove/disable dev fee is unsupported.

## Unsafe Bind Rule

- Default API bind must remain loopback unless operator explicitly opts into unsafe binding.
- Agents should never auto-switch to non-loopback binding without explicit user confirmation.

## API Key Rule

- Write endpoints are open by default; no API key is required.

## Runtime Config Mutability

- `PATCH /v1/miner/live-config` may return a restart requirement depending on active runtime state.
- Agents should handle `requires_restart=true` and ask permission before restart.

## Recommended Operational Sequence

1. Read `GET /v1/runtime/state`.
2. Read/compare `GET /v1/runtime/config/effective`.
3. Apply sparse patch.
4. Subscribe to SSE and verify `state.changed`.
5. Re-read snapshot to confirm final state.

## Retry Policy

- Backoff on non-2xx errors.
- Reconnect SSE and resync via snapshot endpoints.
- Treat 409 conflicts as state coordination issues, not transport failures.
