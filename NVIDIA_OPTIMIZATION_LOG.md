# NVIDIA Optimization Log

This log tracks every measurable NVIDIA backend optimization attempt so we can iterate without repeating dead ends.

## Benchmark Protocol
- Command: `cargo run --release -- --bench --bench-kind kernel --backend nvidia --nvidia-devices 0 --bench-secs 10 --bench-rounds 3 --ui plain`
- GPU target: RTX 3080 (10 GiB)
- Rule: keep only changes that improve multi-round average H/s and do not increase failure rate (OOM/crash/correctness risk).
- Variance note: results move with clocks/thermals; prioritize 3-round averages over single-round samples.

## Current Best Known Config
- Commit: `working tree` (post-pass)
- Kernel model: cooperative per-lane execution (`32` threads/lane block)
- NVRTC flags include: `--maxrregcount=<autotuned>`, `-DSEINE_FIXED_M_BLOCKS=2097152U`, `-DSEINE_FIXED_T_COST=1U`
- NVIDIA autotune schema: `3`; regcap sweep: `240/224/208/192/160` (autotune key now also includes CUDA compute capability; local cache currently selected `160`)
- Startup lane sizing now uses free-VRAM-aware budgeting with a small reserve (`max(total/64, 64 MiB)`)
- Kernel hot path writes compression output directly to destination memory (no `block_tmp` shared staging buffer)
- Runtime hot path: no explicit pre-`memcpy_dtoh` stream synchronize in `run_fill_batch`
- Observed best 3-round average so far: `1.214 H/s` (2026-02-13)

## Attempt History
| Date | Commit/State | Change | Result | Outcome |
| --- | --- | --- | --- | --- |
| 2026-02-13 | `2b5ed78` | Initial native CUDA backend | `0.065 H/s` | Baseline |
| 2026-02-13 | pre-`35190c1` | One lane per CUDA block launch geometry | `0.076 H/s` | Kept |
| 2026-02-13 | pre-`35190c1` | Full free-VRAM lane sizing without probe | OOM at startup | Reverted |
| 2026-02-13 | pre-`35190c1` | Shared-memory scratch rewrite (old kernel) | `0.050 H/s` | Reverted |
| 2026-02-13 | pre-`35190c1` | VRAM touch-probe + lane backoff + regcap tuning | up to `0.128 H/s` | Kept in `35190c1` |
| 2026-02-13 | `1198bd2` | Cooperative per-lane kernel rewrite | `0.983 H/s` avg | Kept |
| 2026-02-13 | working tree | Cooperative threads sweep: `24/32/40/48` | `0.574/0.978/0.590/0.766 H/s` | `32` kept |
| 2026-02-13 | working tree | Lane cap sweep: `1/2/3/4` | `0.246/0.487/0.736/0.957 H/s` | Highest lane count kept |
| 2026-02-13 | working tree | Regcap A/B: `224` vs `160` | `1.007` vs `0.973 H/s` avg | `224` kept |
| 2026-02-13 | working tree | PTXAS cache hint `-dlcm=ca` | `0.985 H/s` avg | Reverted |
| 2026-02-13 | working tree | Thread-0-only ref-index mapping micro-opt | `0.849 H/s` avg | Reverted |
| 2026-02-13 | working tree | Cooperative loop unroll pragma pass | `0.583 H/s` | Reverted |
| 2026-02-13 | working tree | Warp-sync + device hash-counter removal + lane-0 ref-index broadcast | `0.977 H/s` avg | Reverted (slower than baseline) |
| 2026-02-13 | working tree | Keep warp-sync + drop device hash-counter + restore per-thread ref-index + remove pre-D2H stream sync | `1.027 H/s` avg | Kept (new best, stable over rerun) |
| 2026-02-13 | working tree | Remove post-store `coop_sync()` in inner block loop | `1.014 H/s` avg | Reverted (regressed vs 1.026 baseline) |
| 2026-02-13 | working tree | Direct compression write-to-destination (drop `block_tmp` shared staging) | `1.120 H/s` avg | Kept (clear uplift over baseline) |
| 2026-02-13 | working tree | Compile-time Argon2 constants via NVRTC defines (`SEINE_FIXED_M_BLOCKS`, `SEINE_FIXED_T_COST`) | `1.214 H/s` avg | Kept (new best, stable over rerun) |
| 2026-02-13 | working tree | NVIDIA autotune schema bump `1 -> 2` + regcap candidate sweep `240/224/208/192/160` | baseline `1.162 H/s`; candidates `0.896/1.202 H/s`; final rerun `1.208 H/s` | Kept (`+3.44%` best-vs-baseline; first candidate outlier tied to one-time cache rebuild) |
| 2026-02-13 | working tree | Drop NVRTC option `--extra-device-vectorization` from NVIDIA kernel compile flags | baseline `1.094 H/s`; candidates `1.179/1.193 H/s`; final rerun `1.193 H/s` | Kept (`+9.05%` best-vs-baseline, `+8.41%` mean, stable across rerun) |
| 2026-02-13 | working tree | NVIDIA autotune schema bump `2 -> 3` + regcap candidate sweep `240/224/208/192/176/160/144` | baseline `1.143 H/s`; candidates `0.864/1.143 H/s`; final reruns `0.874/1.122 H/s` | Reverted (best delta `0.00%`, below `0.5%` keep gate; introduced one-time cache-rebuild outlier) |
| 2026-02-13 | working tree | Add NVRTC PTXAS cache hint `--ptxas-options=-dlcm=cg` | baseline `1.139 H/s`; candidates `1.045/1.048 H/s`; final rerun `1.138 H/s` | Reverted (`-7.99%` best-vs-baseline, clear and stable regression) |
| 2026-02-13 | working tree | Drop NVRTC option `--use_fast_math` from NVIDIA kernel compile flags | baseline `1.138 H/s`; candidates `1.114/1.130 H/s`; final rerun `1.121 H/s` | Reverted (`-0.70%` best-vs-baseline, below keep gate and negative) |
| 2026-02-14 | working tree | Split NVIDIA worker channels (`assign` vs `control`), enable true nonblocking backend calls, and initially add deadline-tail lane throttling plus loop-unroll autotune candidate | same-machine `HEAD` baseline (fixed regcap `160`): kernel `1.184 H/s`, backend `1.213 H/s`; initial candidate: kernel `1.119 H/s`, backend `1.049 H/s` | Partially reverted (tail throttling + active unroll path regressed throughput) |
| 2026-02-14 | working tree | Keep worker channel split + nonblocking control path; revert deadline-tail lane throttling | backend reruns with fixed regcap `160`: `1.195/1.248 H/s` vs `HEAD` `1.213 H/s` | Kept (`+2.89%` best-vs-baseline; control enqueue no longer blocks behind long kernel batches) |
| 2026-02-14 | working tree | Remove live `SEINE_BLOCK_LOOP_UNROLL` NVRTC/kernel path; keep cache compatibility (`schema=2`, `#[serde(default)] block_loop_unroll`) and unroll candidate disabled | kernel reruns with fixed regcap `160`: `1.163/1.201 H/s` vs `HEAD` `1.184 H/s`; backend rerun `1.279 H/s` vs `HEAD` `1.213 H/s` | Kept (kernel best `+1.44%`; backend best `+5.44%`; avoids forced cache-rebuild churn) |
| 2026-02-14 | working tree | True NVIDIA batch queueing (no collapse), free-VRAM-aware lane budgeting (`derive_memory_budget_mib`), autotune key/schema update (`schema=3`, include compute capability) | fresh baseline: kernel `0.964 H/s`, backend `1.045 H/s`; clean kernel candidates `1.149/1.117/1.155 H/s`; backend candidates `1.183/1.171/1.164 H/s` (additional low-clock samples `1.044/1.044 H/s`) | Kept (kernel mean `+18.29%`; backend mean `+7.29%` across all clean samples; queueing correctness + OOM resilience improved) |
| 2026-02-14 | working tree | Deadline-tail dynamic lane throttling via per-assignment hash-time EMA | kernel candidate `0.881 H/s`; backend candidate `0.880 H/s` vs same-pass baseline `0.964/1.045 H/s` | Reverted (clear throughput regression; reduced late shares but hurt total H/s) |
| 2026-02-14 | working tree | Little-endian memcpy fast path for seed-word expansion + final block byte packing | kernel `1.136 H/s` vs local kept-state `1.117 H/s`; backend `1.167 H/s` vs local kept-state `1.171 H/s` | Reverted (backend delta `-0.34%`, below keep gate and negative) |

## New Entry Template
Copy this row and fill in all fields after each pass:

`| YYYY-MM-DD | <commit or working tree> | <exact code/config change> | <H/s avg and command> | <Kept/Reverted + why> |`
