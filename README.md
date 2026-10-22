# CPU benchmark for decision models

*Does a decision model run usefully on a CPU? Measured, not estimated.*

Does a **decision model** (System One / Jev-class: typed, calibrated answers, no
generated text) run usefully on a CPU, and which CPUs are worth using?

Everything here is reproducible and self-contained. Background and the model
class itself: [`DECISION_MODELS.md`](DECISION_MODELS.md).

## Run it

```bash
curl -fLO https://raw.githubusercontent.com/sku10/cpu-decision-bench/main/cpu_decision_bench.sh
chmod +x cpu_decision_bench.sh

./cpu_decision_bench.sh --check     # hardware verdict only, no download, ~1 s
./cpu_decision_bench.sh --quick     # ~5 min
./cpu_decision_bench.sh             # full profile, ~20 min
```

Needs `gcc g++ cmake git python3 curl`, ~6 GB disk, no root and no GPU. It
builds a CPU-only llama.cpp at a pinned commit, fetches a 4B decision model,
generates its own benchmark items, and writes everything under `./cpudecbench/`.
Delete that directory to remove all of it.

**Run `--check` first on every machine.** It prints the CPU, usable L3, memory
and SIMD support, says whether a model layer fits in cache, and exits non-zero
on a CPU that cannot run this at all — no download, no build.

## What it measures

1. **Thread scaling** — one request at a time at `-t 1,2,4,<all>`. Shows whether a
   single decision parallelises.
2. **Concurrency** — 1/2/4/8 simultaneous requests, `-t <all>` with 8 slots. Shows
   whether throughput scales with load.
3. **Concurrency, 1 thread per slot** — the other deployment shape.
4. **Package watts** via RAPL where readable (Intel, and AMD with
   `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`).

## Reading the result

| pattern | meaning |
|---|---|
| req/s flat as concurrency rises, CPU well under 100% of all cores | **memory-bound** — more cores won't help; memory channels and L3 will |
| req/s rises with concurrency | this CPU has headroom; serve several in parallel |
| p50 flat from `-t 1` to `-t <cores>` | a single decision doesn't parallelise; prefer many slots × few threads |
| big regression at `-t <all SMT threads>` | SMT contention — set `-t` to **physical** cores |

## Why L3 is the number to look at

The model is 32 layers. Each decision is one forward pass that reads every
layer's weights exactly once, with no reuse — so the floor is
`model_bytes ÷ memory_bandwidth`, and cache only helps if a **layer** fits.

Per-layer footprint, measured from the GGUF files with `layer_size.py`:

| quant | per layer | total |
|---|---|---|
| Q4_K_M | **64 MiB** | 2.51 GiB |
| Q5_K_M | 74 MiB | 2.85 GiB |
| Q8_0 | 115 MiB | 4.16 GiB |
| BF16 | 216 MiB | 7.84 GiB |

So ~64 MiB of **usable** L3 is the threshold for Q4.

**"Usable" is per CCD/die, not the total `lscpu` prints.** A 5950X reports
"64 MiB (2 instances)" but it is 2×32 MiB and a thread only reaches its own —
it behaves like 32 MiB. The script accounts for this; read its "usable L3" line,
not the raw lscpu figure.

A CPU with ≥64 MiB usable L3 (X3D parts at 96 MiB, EPYC X-series at 768+ MiB) is
the only place where keeping a layer resident is possible, which is the
precondition for the layer-fused batching described in
[`DECISION_MODELS.md`](DECISION_MODELS.md).

## Measured so far

4B decision model, Q8_0 unless noted, 111-item hard split, `--kv-unified`:

| CPU | usable L3 | RAM | 1 req p50 | req/s @ conc 1→8 |
|---|---|---|---|---|
| Ryzen 7 5800X (8C/16T, X570) | 32 MiB | 64 GB DDR4 dual-ch | 820 ms | 0.99 → 0.95 (flat) |
| Core i5-7600K (4C/4T, Z270) | 6 MiB | 16 GB DDR4-2667 dual-ch | ~3900 ms | — |

Both are **memory-bound**: on the 5800X, 8 concurrent requests raised CPU use
from 335% to 347% of 1600% available — i.e. ~78% of the chip sits idle waiting
on RAM, and throughput does not improve. Adding cores cannot fix this.

Other findings worth knowing before you benchmark:

- **`-t 16` on an 8-core is a 2.9× regression** (p50 819 ms → 2387 ms). Set `-t`
  to physical cores. Several of our earlier numbers were measured at this bad
  setting before we caught it.
- **`--kv-unified` matters.** Without it, throughput *degrades* under load
  (0.66 → 0.34 req/s) and p95 blows out to 91 s. With it, throughput stays flat
  and p95 drops to 13 s. The script always sets it.
- **Q4_K_M is ~18% faster than Q8_0** (1.27 vs 1.08 req/s) for a 2-of-231
  accuracy cost, where every flipped answer was already low-confidence (≤0.578).
  On memory-bound hardware, prefer Q4.
- **No AVX2 means don't bother.** A KVM guest exposing "QEMU Virtual CPU" has no
  AVX2/FMA and falls back to scalar code; the script refuses to benchmark it.

## Which server CPUs to test (estimates)

**These are projections, not measurements.** Our two data points are both
desktop parts, so anything below is an extrapolation and the point of running
the script is to confirm or kill it. Two independent questions:

### (a) Single-decision latency — set by per-core speed, not core count

Threads 1→8 gave **no** speedup, so one decision runs at roughly one core's pace.
What matters is clock × IPC × SIMD width. AVX-512 is worth ~1.5–1.8× per core on
GEMM in practice (not the 2× on paper).

| CPU | all-core GHz | ISA | est. single p50 |
|---|---|---|---|
| Ryzen 7 5800X *(measured)* | 4.5 | AVX2 | **819 ms** |
| EPYC 9354 (Zen4, 32c) | 3.6 | AVX-512 | ~620 ms |
| EPYC 9684X (Zen4-X, 96c) | 3.0 | AVX-512 | ~745 ms |
| Xeon 6342 (ICX, 24c) | 3.0 | AVX-512 | ~745 ms |
| Xeon 8480+ (SPR, 56c) | 2.6 | AVX-512 | ~860 ms |
| EPYC 7763 (Zen3, 64c) | 2.9 | AVX2 | ~1270 ms |
| Xeon 6980P (GNR, 128c) | 2.4 | AVX-512 | ~930 ms |

**A 64-core EPYC is slower per decision than this desktop chip.** Server parts
trade clock for core count, and single-stream latency follows the clock. Do not
expect a big server to feel snappier on one request.

### (b) Aggregate throughput — set by memory bandwidth

Each concurrent decision streams the whole model once, so the ceiling is
`bandwidth × efficiency ÷ model_bytes`. We measured **~13% of peak bandwidth**
on the 5800X; these hold that efficiency constant and scale the channels.

| CPU | channels | GB/s peak | usable L3 | est. Q4 req/s | vs 5800X |
|---|---|---|---|---|---|
| Ryzen 7 5800X *(measured ~1.3)* | 2 × DDR4-3200 | 51 | 32 MiB | ~2.5 | 1× |
| EPYC 7763 (Zen3, 64c) | 8 × DDR4-3200 | 205 | 32 MiB/CCD | ~10 | 4× |
| **EPYC 7773X (Zen3-X, 64c)** | 8 × DDR4-3200 | 205 | **96 MiB/CCD** | ~10 | 4× |
| Xeon 8480+ (SPR, 56c) | 8 × DDR5-4800 | 307 | 105 MiB | ~15 | 6× |
| **EPYC 9654 / 9684X (Zen4, 96c)** | 12 × DDR5-4800 | 461 | 32 / **96 MiB/CCD** | ~22 | 9× |
| Xeon 6980P (GNR, 128c) | 12 × DDR5-6400 | 614 | 504 MiB | ~30 | 12× |
| **Xeon Max 9480 (HBM2e)** | 64 GB HBM | **~1200** | 112 MiB | **~58** | **~45×** |

Multi-socket roughly doubles channels but adds NUMA — assume ~1.7×, not 2×, and
pin one server process per socket.

### Best test candidates, in order

1. **Any EPYC 9004 (Genoa)** — 12 channels of DDR5 is the single biggest lever
   (~9×). The 9354 (32c, 3.6 GHz) is the sweet spot: highest clock in the family,
   so it should win on *both* latency and throughput.
2. **An X-series EPYC (7773X / 9684X)** — 96 MiB per CCD is the only widely
   available part where a **64 MiB Q4 layer actually fits in cache**. This is the
   one configuration that could break the memory wall rather than just widen it.
   If throughput scales with concurrency anywhere, it is here.
3. **Xeon Max 9480** if reachable — HBM2e is ~24× the bandwidth of desktop DDR4.
   By far the largest predicted gain, and the cleanest test of whether bandwidth
   is really the limit.
4. **Sapphire Rapids Xeon (8480+)** — AVX-512 plus DDR5, and `amx_bf16`/`amx_int8`
   may give a step change that none of this modelling captures. Worth running for
   that reason alone.

Lower priority: Zen 3 EPYC (AVX2 only, DDR4) and Ice Lake Xeon — real gains, but
they confirm the bandwidth story rather than test the cache hypothesis.

Skip: anything without AVX2, and any VM exposing a generic CPU model.

### What would falsify the model

If an X-series EPYC still shows **flat req/s as concurrency rises**, then cache
residency is not the limiter and the bottleneck is inside llama.cpp's scheduling,
not the hardware. That result is as useful as the positive one, so report it
either way. Check the `amx` flags in `--check` output on Sapphire Rapids and
later — if present, llama.cpp may already be using a path our estimates ignore.

## Files

| file | purpose |
|---|---|
| `cpu_decision_bench.sh` | the self-contained profiler described above |
| `conc_bench.py` | concurrency sweep against a running server |
| `layer_size.py` | per-layer GGUF footprint (standalone parser, no deps) |
| `run_jevbench.py` | accuracy + calibration over a JevBench-format dataset |
| `compare_cpu_gpu.py` | item-by-item diff of two runs: answers, drift, threshold crossings |

`run_jevbench.py` needs a JevBench-format dataset, which is not included here;
`cpu_decision_bench.sh` generates equivalent synthetic items so it needs nothing.
