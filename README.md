# seine

External miner for Blocknet with pluggable CPU and NVIDIA GPU backends.

![seine TUI](screenshot.png)

## Quick Start

**Option A — Pre-built binary** (from [Releases](../../releases)):

```bash
# Download the binary for your platform, then:
./seine
```

**Option B — Build from source:**

```bash
cargo build --release
./target/release/seine
```

Zero-argument mode is the default path. Seine auto-resolves:
- API URL from running daemon args when available (`--api`), else `http://127.0.0.1:8332`
- auth token from `api.cookie` (auto-discovered from daemon data dir)
- backends (CPU + NVIDIA when available)
- CPU threads from available cores and RAM

If your daemon is not running yet, or you want to override detection, set these explicitly:

```bash
./seine --api-url http://127.0.0.1:8332 --cookie /path/to/data/api.cookie
```

## Requirements

| Platform | CPU | GPU (NVIDIA) |
|----------|-----|--------------|
| Linux x86_64 | works out of the box | CUDA driver + NVRTC libs |
| macOS x86_64 | works out of the box | — |
| macOS ARM | works out of the box (Metal experimental) | — |
| Windows x86_64 | works out of the box | CUDA driver + NVRTC libs |

Each CPU thread needs ~2 GB RAM (Argon2id parameters). Seine auto-sizes thread count from available cores and memory.

## Configuration

```bash
# Set CPU thread count explicitly
./seine --threads 4

# Force a specific backend (auto-detects by default)
./seine --backend cpu
./seine --backend nvidia
./seine --backend cpu,nvidia

# Wallet password (if wallet is encrypted)
./seine --wallet-password-file /path/to/wallet.pass

# Plain log output instead of TUI
./seine --ui plain
```

Password sources (checked in order): `--wallet-password`, `--wallet-password-file`, `SEINE_WALLET_PASSWORD` env var, interactive prompt.

## GPU Mining

### NVIDIA

Requires CUDA driver and NVRTC libraries on the host. Seine compiles kernels at startup via NVRTC.

```bash
# Auto-detect all GPUs
./seine --backend nvidia

# Select specific devices
./seine --backend nvidia --nvidia-devices 0,1
```

If CUDA initialization fails, NVIDIA backends are quarantined and CPU mining continues.

### Metal (macOS ARM)

Metal support is experimental. Pre-built macOS ARM binaries include it. To build from source:

```bash
cargo build --release --no-default-features --features metal
```

## Building from Source

```bash
# Default build (includes NVIDIA support)
cargo build --release

# CPU-only build (no CUDA dependency)
cargo build --release --no-default-features

# Host-native CPU build (optimized for your specific CPU)
./scripts/build_cpu_native.sh --cpu-only
```

## Benchmarking

Run offline benchmarks without a daemon connection:

```bash
./seine --bench --bench-kind backend --backend cpu --threads 1 --bench-secs 20 --bench-rounds 3
```

## Further Reading

- [AGENTS.md](AGENTS.md) — Architecture details, tuning knobs, benchmarking harness
- [CPU_OPTIMIZATION_LOG.md](CPU_OPTIMIZATION_LOG.md) — CPU backend tuning history
- [NVIDIA_OPTIMIZATION_LOG.md](NVIDIA_OPTIMIZATION_LOG.md) — NVIDIA backend tuning history
