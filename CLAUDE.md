# seine

External Blocknet miner with pluggable CPU/GPU backends.

## Developer & Agent Reference

See [AGENTS.md](AGENTS.md) for architecture details, runtime tuning knobs, benchmarking harness, and optimization history.

## Key Paths

- `src/miner/` — runtime orchestration (mining, scheduler, backend control, stats)
- `src/backend/` — PoW backend implementations (CPU, NVIDIA, Metal)
- `scripts/` — build and benchmark helpers
- `CPU_OPTIMIZATION_LOG.md` — measured CPU tuning history
- `NVIDIA_OPTIMIZATION_LOG.md` — measured NVIDIA tuning history
