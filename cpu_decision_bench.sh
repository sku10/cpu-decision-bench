#!/usr/bin/env bash
# cpu_decision_bench.sh -- profile a CPU for decision-model (System One) serving.
#
# Self-contained: checks the CPU, builds llama.cpp (CPU-only), fetches a 4B
# decision model, and reports single-thread latency, thread scaling, concurrency
# throughput and (where readable) package watts.
#
#   ./cpu_decision_bench.sh                 # full profile, Q4_K_M
#   ./cpu_decision_bench.sh --quant Q8_0    # heavier quant
#   ./cpu_decision_bench.sh --quick         # ~4 min instead of ~20
#   ./cpu_decision_bench.sh --check         # just print hardware verdict, run nothing
#
# Needs: gcc/g++, cmake, git, python3, curl, ~6 GB disk. No root, no GPU.
# Everything lands in ./cpudecbench/ under the current directory.
set -uo pipefail

QUANT=Q4_K_M; QUICK=0; CHECK_ONLY=0; PORT=8199
REPO=https://github.com/ggml-org/llama.cpp.git
COMMIT=0a1ef309cdbb9f2024f58a38fb4c42352c5db569   # first release with /v1/systemone
HF_REPO=h2oai/h2o-lightning-4b-gguf
ROOT="$PWD/cpudecbench"

while [ $# -gt 0 ]; do
  case "$1" in
    --quant) QUANT="$2"; shift 2;;
    --quick) QUICK=1; shift;;
    --check) CHECK_ONLY=1; shift;;
    --port)  PORT="$2"; shift 2;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

say() { printf '\n== %s ==\n' "$*"; }

# ---------------------------------------------------------------- hardware
say "hardware"
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
CORES=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{print $2}' | xargs)
TPC=$(lscpu 2>/dev/null | awk -F: '/^Thread\(s\) per core/{print $2}' | xargs)
SOCKETS=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/{print $2}' | xargs)
NPROC=$(nproc)
L3=$(lscpu 2>/dev/null | awk -F: '/^L3 cache/{print $2}' | xargs)
L3MIB=$(echo "$L3" | grep -oE '^[0-9.]+' | cut -d. -f1)
case "$L3" in *GiB*) L3MIB=$(( ${L3MIB:-0} * 1024 ));; esac
# lscpu prints the SUM across L3 instances (CCDs/dies). L3 is not shared between
# them, so the figure that matters for holding a layer is per-instance.
L3INST=$(echo "$L3" | grep -oE '\(([0-9]+) instance' | grep -oE '[0-9]+')
L3INST=${L3INST:-1}
L3PER=$(( ${L3MIB:-0} / L3INST ))
MEMGB=$(free -g | awk '/^Mem:/{print $2}')
BOARD="$(cat /sys/devices/virtual/dmi/id/board_vendor 2>/dev/null) $(cat /sys/devices/virtual/dmi/id/board_name 2>/dev/null)"
VIRT=$(systemd-detect-virt 2>/dev/null | head -1); VIRT=${VIRT:-unknown}
printf 'cpu          : %s\n' "$CPU"
printf 'topology     : %s socket(s) x %s core(s) x %s thread(s) = %s\n' "${SOCKETS:-?}" "${CORES:-?}" "${TPC:-?}" "$NPROC"
if [ "${L3INST:-1}" -gt 1 ]; then
  printf 'L3 cache     : %s  -> %s MiB per instance (NOT shared between them)\n' "${L3:-unknown}" "$L3PER"
else
  printf 'L3 cache     : %s  (~%s MiB)\n' "${L3:-unknown}" "${L3MIB:-?}"
fi
printf 'memory       : %s GB\n' "${MEMGB:-?}"
printf 'board        : %s\n' "${BOARD:- unknown}"
printf 'virtualised  : %s\n' "$VIRT"

ISA=$(grep -oE '\b(avx512[a-z_0-9]*|amx_[a-z0-9]+|avx_vnni|avx2|avx|fma|f16c)\b' /proc/cpuinfo | sort -u | tr '\n' ' ')
printf 'simd         : %s\n' "${ISA:-none}"
FATAL=0
case "$ISA" in
  *avx2*) ;;
  *) echo "VERDICT: NOT SUPPORTED -- no AVX2. llama.cpp falls back to a scalar path;"
     echo "         any number measured here describes the missing instruction set, not the CPU."
     echo "         (Common on KVM guests exposing a generic 'QEMU Virtual CPU'.)"
     FATAL=1;;
esac
case "$ISA" in *fma*) ;; *) echo "WARNING: no FMA -- expect roughly 2x slower than a comparable FMA chip.";; esac
case "$ISA" in *amx_*) echo "NOTE   : AMX present (Sapphire Rapids or later). This is a matrix-multiply"
                           echo "         accelerator that may give a step change our DDR-bandwidth estimates"
                           echo "         do not model. A result here is especially interesting.";; esac
case "$ISA" in *avx512*) echo "NOTE   : AVX-512 present -- expect ~1.5-1.8x per core over AVX2 on this workload.";; esac

# per-layer footprint of this 4B model, measured from the GGUF (see layer_size.py)
case "$QUANT" in
  Q4_K_M) LAYER_MIB=64;;  Q5_K_M) LAYER_MIB=74;;
  Q8_0)   LAYER_MIB=115;; BF16|F16) LAYER_MIB=216;;
  *) LAYER_MIB=64;;
esac
if [ "${L3PER:-0}" -gt 0 ]; then
  if [ "$L3INST" -gt 1 ]; then
    echo "cache check  : one $QUANT layer is ~${LAYER_MIB} MiB vs ${L3PER} MiB usable L3"
    echo "               (${L3MIB} MiB total / $L3INST instances -- a thread only reaches its own)"
  else
    echo "cache check  : one $QUANT layer is ~${LAYER_MIB} MiB vs ${L3PER} MiB L3"
  fi
  if [ "$L3PER" -ge "$LAYER_MIB" ]; then
    echo "               -> a full layer FITS in L3. This CPU can benefit from layer-fused"
    echo "                  batching, and should scale with concurrency better than most."
  else
    PCT=$(( L3PER * 100 / LAYER_MIB ))
    echo "               -> holds ~${PCT}% of a layer; weights stream from DRAM every layer."
    echo "                  Expect concurrency to saturate early (memory-bound)."
    if [ "$LAYER_MIB" -gt 64 ]; then
      echo "               -> try --quant Q4_K_M (~64 MiB/layer) to cut memory traffic."
    fi
  fi
fi
[ "$CHECK_ONLY" = 1 ] && exit $FATAL
[ "$FATAL" = 1 ] && { echo; echo "Refusing to benchmark without AVX2. Re-run with --check to see this verdict only."; exit 1; }

for t in gcc g++ cmake git python3 curl; do
  command -v "$t" >/dev/null || { echo "MISSING TOOL: $t"; exit 1; }
done

mkdir -p "$ROOT"/{models,data} && cd "$ROOT" || exit 1

# ---------------------------------------------------------------- build
say "build (CPU-only llama.cpp @ ${COMMIT:0:9})"
if [ ! -x llama.cpp/build/bin/llama-server ]; then
  [ -d llama.cpp ] || git clone -q --progress=false --filter=blob:none "$REPO" llama.cpp 2>/dev/null || git clone -q --filter=blob:none "$REPO" llama.cpp || exit 1
  ( cd llama.cpp \
    && git fetch --depth=1 origin "$COMMIT" 2>/dev/null \
    && git checkout -q "$COMMIT" \
    && cmake -B build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=ON \
             -DCMAKE_BUILD_TYPE=Release > /tmp/cdb-cmake.log 2>&1 \
    && cmake --build build --target llama-server -j"$NPROC" > /tmp/cdb-build.log 2>&1 ) \
    || { echo "build failed; see /tmp/cdb-build.log"; tail -20 /tmp/cdb-build.log; exit 1; }
fi
BIN="$ROOT/llama.cpp/build/bin/llama-server"
"$BIN" --version 2>&1 | head -2

# ---------------------------------------------------------------- model
say "model ($QUANT)"
MODEL="$ROOT/models/h2o-lightning-4b-$QUANT.gguf"
if [ ! -s "$MODEL" ]; then
  URL="https://huggingface.co/$HF_REPO/resolve/main/h2o-lightning-4b-$QUANT.gguf?download=true"
  echo "downloading $QUANT (2.5-4.5 GB) ..."
  curl -fL --retry 3 -o "$MODEL.part" "$URL" || { echo "download failed"; exit 1; }
  mv "$MODEL.part" "$MODEL"
fi
ls -la "$MODEL" | awk '{printf "%s  %.2f GiB\n", $9, $5/1073741824}'

# ---------------------------------------------------------------- data
say "benchmark items"
# 24 synthetic decisions with the same shape/size as the public JevBench hard split
python3 - <<'PY'
import json, random, os
random.seed(7)
POLICY = ("Refunds require an unused item, a receipt, and a request within 30 days of "
          "delivery. Items marked final-sale are excluded. Shipping damage is handled "
          "under the carrier claim process, not the refund policy. Store credit may be "
          "offered at the agent's discretion when the receipt is missing but the order "
          "is found in the system. ")
rows = []
for i in range(24):
    days = random.choice([5, 12, 28, 34, 61])
    receipt = random.choice([True, False])
    used = random.choice([True, False])
    state = {
        "policy": POLICY * 3,
        "case": {"days_since_delivery": days, "receipt_present": receipt,
                 "item_used": used, "final_sale": False,
                 "notes": "Customer contacted support by email. " * 8},
    }
    ok = (days <= 30) and receipt and not used
    rows.append({"id": f"synth-{i:02d}", "family": "policy",
                 "state": state, "expected": ok,
                 "question": {"type": "noul",
                              "instructions": "Is the refund permitted under `policy`?",
                              "criteria": {"true": "All stated conditions are met",
                                           "false": "Any required condition is unmet"}}})
os.makedirs("data", exist_ok=True)
with open("data/hard.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
print(f"wrote data/hard.jsonl ({len(rows)} items, "
      f"~{len(json.dumps(rows[0]['state']))} chars of state each)")
PY

# ---------------------------------------------------------------- harness
cat > conc_bench.py <<'PY'
import argparse, json, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
def post(url, payload, timeout=600):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--data", default="data")
ap.add_argument("--n", type=int, default=24); ap.add_argument("--levels", default="1")
a = ap.parse_args()
items = [json.loads(l) for l in open(f"{a.data}/hard.jsonl") if l.strip()]
items = (items * (a.n // len(items) + 1))[:a.n]
for conc in [int(x) for x in a.levels.split(",")]:
    lat, bad = [], 0
    def one(it):
        p = {"state": it["state"], "questions": {"q": it["question"]}}
        t0 = time.perf_counter()
        try: post(a.url, p)
        except Exception: return None
        return time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for d in ex.map(one, items):
            if d is None: bad += 1
            else: lat.append(d)
    wall = time.perf_counter() - t0
    if not lat: print(f"{conc:>5}  ALL FAILED"); continue
    lat.sort()
    print(f"{conc:>5} {wall:>8.1f} {len(lat)/wall:>8.2f} {lat[len(lat)//2]*1000:>8.0f} "
          f"{lat[min(len(lat)-1,int(len(lat)*0.95))]*1000:>8.0f} "
          f"{statistics.mean(lat)*1000:>8.0f}" + (f"  ({bad} failed)" if bad else ""))
PY

rapl() { cat /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
HAVE_RAPL=$([ -n "$(rapl)" ] && echo yes || echo no)
[ "$HAVE_RAPL" = no ] && echo "(package watts unavailable: RAPL not readable -- try 'sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj')"

start() { "$BIN" -m "$MODEL" -ngl 0 -t "$1" -tb "$1" --parallel "$2" -c "$3" \
            -b 8192 -ub 2048 --kv-unified --host 127.0.0.1 --port "$PORT" \
            > /tmp/cdb-srv.log 2>&1 &
          for _ in $(seq 1 180); do
            curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && return 0
            sleep 1; done; return 1; }
stop()  { local p; p=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
          [ -n "$p" ] && kill "$p" 2>/dev/null; sleep 2; }
trap stop EXIT

row() { # label conc n
  local e0 t0 e1 t1 secs w out
  e0=$(rapl); t0=$(date +%s.%N)
  out=$(python3 conc_bench.py --url "http://127.0.0.1:$PORT/v1/systemone" --n "$3" --levels "$2" 2>/dev/null | tail -1)
  t1=$(date +%s.%N); e1=$(rapl)
  w="n/a"
  if [ "$HAVE_RAPL" = yes ]; then
    secs=$(echo "$t1 - $t0" | bc)
    w=$(echo "scale=1; ($e1 - $e0)/1000000/$secs" | bc 2>/dev/null)
  fi
  printf '%-10s %s  pkg_W=%s\n' "$1" "$out" "$w"
}

HDR=$(printf '%-10s %5s %8s %8s %8s %8s %8s' cfg conc wall_s req/s p50_ms p95_ms mean_ms)
N=24; LEVELS="1 2 4 8"; THREADS="1 2 4 $NPROC"
[ "$QUICK" = 1 ] && { N=12; LEVELS="1 4"; THREADS="1 $NPROC"; }

say "idle power"
if [ "$HAVE_RAPL" = yes ]; then
  E0=$(rapl); sleep 5; E1=$(rapl)
  echo "idle package: $(echo "scale=1; ($E1-$E0)/1000000/5" | bc) W"
else echo "idle package: n/a"; fi

say "thread scaling (one request at a time)"
echo "$HDR"
for T in $THREADS; do
  [ "$T" -gt "$NPROC" ] && continue
  start "$T" 1 8192 && { row "t$T/c1" 1 "$N"; stop; }
done

say "concurrency (threads = $NPROC, 8 slots)"
echo "$HDR"
if start "$NPROC" 8 65536; then
  for C in $LEVELS; do row "t$NPROC/c$C" "$C" "$((N > C ? N : C))"; done
  stop
fi

say "concurrency (1 thread per slot, 8 slots)"
echo "$HDR"
if start 1 8 65536; then
  for C in $LEVELS; do row "t1/c$C" "$C" "$((N > C ? N : C))"; done
  stop
fi

cat <<TXT

== how to read this ==
* Flat req/s as conc rises, with CPU well under 100% of all cores
    -> memory-bound. More cores will not help; more memory channels and
       larger L3 will. This is the usual outcome on desktop parts.
* req/s rising roughly with conc
    -> this CPU has headroom; serve several requests in parallel.
* p50 flat from -t 1 to -t <cores>
    -> a single decision does not parallelise; prefer many slots with
       few threads each over one wide slot.
* A big regression at -t <all SMT threads> (e.g. -t 16 on an 8-core)
    -> SMT contention. Set -t to PHYSICAL cores, not threads.

Report: CPU model, L3, memory config, and the three tables above.
TXT
