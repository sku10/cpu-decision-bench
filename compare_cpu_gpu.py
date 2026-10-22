#!/usr/bin/env python3
"""Compare CPU vs GPU JevBench runs: accuracy, answer agreement, latency, calibration."""
import json, sys, glob, os

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def ece(rows, bins=10):
    buckets = {}
    for r in rows:
        b = min(int(r["confidence"] * bins), bins - 1)
        buckets.setdefault(b, []).append(r)
    n = len(rows); tot = 0.0
    for rs in buckets.values():
        acc = sum(1 for r in rs if r["correct"]) / len(rs)
        cf  = sum(r["confidence"] for r in rs) / len(rs)
        tot += (len(rs) / n) * abs(acc - cf)
    return tot

def summarise(path):
    rows = load(path)
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "file": os.path.basename(path), "n": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "ece": ece(rows),
        "p50": lat[len(lat)//2]*1000,
        "p95": lat[min(len(lat)-1, int(len(lat)*0.95))]*1000,
        "mean": sum(lat)/len(lat)*1000,
        "total_s": sum(lat),
        "rows": {r["id"]: r for r in rows},
    }

if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    A, B = summarise(a), summarise(b)
    for S in (A, B):
        print(f"{S['file']:<34} {S['correct']}/{S['n']}  ECE {S['ece']:.4f}  "
              f"p50 {S['p50']:.0f}ms  p95 {S['p95']:.0f}ms  mean {S['mean']:.0f}ms  "
              f"wall {S['total_s']:.0f}s")
    common = set(A["rows"]) & set(B["rows"])
    diff = [i for i in common if A["rows"][i]["predicted"] != B["rows"][i]["predicted"]]
    print(f"\ncommon items      : {len(common)}")
    print(f"answers differing : {len(diff)}")
    drift = max(abs(A["rows"][i]["confidence"] - B["rows"][i]["confidence"]) for i in common)
    print(f"max conf drift    : {drift:.3e}")
    if diff:
        for i in sorted(diff)[:15]:
            ra, rb = A["rows"][i], B["rows"][i]
            print(f"  {i:<28} A={ra['predicted']!r}({ra['confidence']:.3f}) "
                  f"B={rb['predicted']!r}({rb['confidence']:.3f}) exp={ra['expected']!r}")
    if A["mean"] and B["mean"]:
        print(f"\nspeed ratio (mean): {max(A['mean'],B['mean'])/min(A['mean'],B['mean']):.1f}x")
