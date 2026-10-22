#!/usr/bin/env python3
"""Run the 231 public JevBench decisions against a llama-server /v1/systemone endpoint.

Scores accuracy (vs the gold hard label) and calibration (ECE, 10 equal-width bins,
empty bins absent -- the JevBench convention). Reports per-family and per-type too.

Target to reproduce: H2O's published 204/231 for Q8_0 (BF16 reference 205/231).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

QID = "jevbench"


def post(url: str, payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def build_question(item: dict) -> dict:
    """JevBench item -> the /v1/systemone question object.

    choice: criteria is {label: description}; keep the dataset's own label set.
    score:  criteria is an ordered list of level names.
    noul:   no criteria needed.
    """
    q = item["question"]
    qtype = q["type"]
    out = {"type": qtype, "instructions": q["instructions"]}
    crit = q.get("criteria")
    if qtype == "choice":
        labels = item["labels"]
        if isinstance(crit, dict):
            out["criteria"] = {str(l): crit.get(str(l)) for l in labels}
        else:
            out["criteria"] = {str(l): None for l in labels}
    elif qtype == "score":
        if isinstance(crit, dict):
            # dict keyed by level -> ordered list by numeric key
            out["criteria"] = [crit[k] for k in sorted(crit, key=lambda x: int(x))]
        elif isinstance(crit, list):
            out["criteria"] = crit
        else:
            out["criteria"] = [str(l) for l in item["labels"]]
    elif qtype == "noul":
        if isinstance(crit, dict):
            out["criteria"] = {str(k): v for k, v in crit.items()}
    return out


def predicted(ans: dict, item: dict):
    """The model's top answer, in the same space as item['expected']."""
    t = ans.get("type")
    if t == "choice":
        return ans.get("choice")
    if t == "score":
        # expected is an int level; take the argmax level, not the weighted mean
        probs = ans.get("probabilities") or {}
        if not probs:
            return None
        return int(max(probs, key=probs.get))
    if t == "noul":
        return bool(ans.get("noul", 0) >= 0.5)
    return None


def confidence(ans: dict) -> float:
    """P(the answer we actually gave) -- the number calibration is about."""
    t = ans.get("type")
    if t == "noul":
        p = float(ans.get("noul", 0.0))
        return p if p >= 0.5 else 1.0 - p
    probs = ans.get("probabilities") or {}
    return max(probs.values()) if probs else 0.0


def normalise_expected(exp, ans_type):
    if ans_type == "noul":
        if isinstance(exp, bool):
            return exp
        if isinstance(exp, str):
            return exp.strip().lower() in ("true", "yes", "1")
        if isinstance(exp, (int, float)):
            return bool(exp)
    if ans_type == "score":
        try:
            return int(exp)
        except (TypeError, ValueError):
            return exp
    return exp


def ece(records, bins: int = 10) -> float:
    """Expected Calibration Error, 10 equal-width bins; empty bins absent."""
    if not records:
        return float("nan")
    buckets: dict[int, list] = {}
    for conf, correct in records:
        b = min(int(conf * bins), bins - 1)
        buckets.setdefault(b, []).append((conf, correct))
    n = len(records)
    total = 0.0
    for rows in buckets.values():
        acc = sum(1 for _, c in rows if c) / len(rows)
        avg_conf = sum(cf for cf, _ in rows) / len(rows)
        total += (len(rows) / n) * abs(acc - avg_conf)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/systemone")
    ap.add_argument("--data", required=True, help="dir with easy/original/hard .jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke test)")
    ap.add_argument("--out", default="", help="write per-item JSONL here")
    ap.add_argument("--label", default="", help="run label for the summary line")
    args = ap.parse_args()

    items = []
    for split in ("easy", "original", "hard"):
        path = f"{args.data}/{split}.jsonl"
        try:
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        items.append(json.loads(line))
        except FileNotFoundError:
            print(f"missing {path}", file=sys.stderr)
            return 2
    if args.limit:
        items = items[: args.limit]

    print(f"{len(items)} decisions -> {args.url}")
    results, lat = [], []
    out_fh = open(args.out, "w") if args.out else None

    for i, item in enumerate(items, 1):
        q = build_question(item)
        payload = {"state": item["state"], "questions": {QID: q}}
        t0 = time.perf_counter()
        try:
            resp = post(args.url, payload)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  [{i}] {item['id']}: REQUEST FAILED {e}", file=sys.stderr)
            continue
        dt = time.perf_counter() - t0
        lat.append(dt)

        ans = (resp.get("answers") or {}).get(QID) or {}
        pred = predicted(ans, item)
        exp = normalise_expected(item["expected"], ans.get("type"))
        ok = (pred == exp)
        conf = confidence(ans)
        results.append({
            "id": item["id"], "family": item["family"], "type": ans.get("type") or item["question"]["type"],
            "expected": exp, "predicted": pred, "correct": ok, "confidence": conf,
            "latency_s": round(dt, 4),
            "usage": resp.get("usage") or {},
        })
        if out_fh:
            out_fh.write(json.dumps(results[-1]) + "\n")
        if args.limit or i % 25 == 0:
            mark = "ok " if ok else "WRONG"
            print(f"  [{i}/{len(items)}] {item['id']:<28} {mark} pred={pred!r} exp={exp!r} conf={conf:.3f} {dt*1000:.0f}ms")

    if out_fh:
        out_fh.close()
    if not results:
        print("no results", file=sys.stderr)
        return 1

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    cal = ece([(r["confidence"], r["correct"]) for r in results])
    lat.sort()
    p50 = lat[len(lat) // 2]
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]

    print("\n" + "=" * 62)
    if args.label:
        print(f"run: {args.label}")
    print(f"accuracy : {correct}/{n} = {100*correct/n:.2f}%")
    print(f"ECE      : {cal:.4f}  (10 bins; lower is better)")
    print(f"latency  : p50 {p50*1000:.0f} ms   p95 {p95*1000:.0f} ms")
    out_tokens = sum((r['usage'] or {}).get('output_tokens', 0) for r in results)
    print(f"output_tokens total: {out_tokens}  (expect 0 - no text generated)")

    print("\nby question type:")
    for t in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == t]
        c = sum(1 for r in rows if r["correct"])
        print(f"  {t:<8} {c:>3}/{len(rows):<3} {100*c/len(rows):6.2f}%   ECE {ece([(r['confidence'], r['correct']) for r in rows]):.4f}")

    print("\nby family:")
    for f in sorted({r["family"] for r in results}):
        rows = [r for r in results if r["family"] == f]
        c = sum(1 for r in rows if r["correct"])
        print(f"  {f:<18} {c:>3}/{len(rows):<3} {100*c/len(rows):6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
