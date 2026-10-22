#!/usr/bin/env python3
"""Concurrency sweep: fixed item set, N parallel workers, measure throughput + latency.

Answers: does a decision model server scale with concurrent requests when a
single request barely uses more than one core?
"""
import argparse, json, statistics, sys, time
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def build_question(item):
    q = item["question"]; t = q["type"]
    out = {"type": t, "instructions": q["instructions"]}
    crit = q.get("criteria")
    if t == "choice":
        labels = item["labels"]
        out["criteria"] = ({str(l): crit.get(str(l)) for l in labels}
                           if isinstance(crit, dict) else {str(l): None for l in labels})
    elif t == "score":
        if isinstance(crit, dict):
            out["criteria"] = [crit[k] for k in sorted(crit, key=lambda x: int(x))]
        elif isinstance(crit, list):
            out["criteria"] = crit
        else:
            out["criteria"] = [str(l) for l in item["labels"]]
    elif t == "noul" and isinstance(crit, dict):
        out["criteria"] = {str(k): v for k, v in crit.items()}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="hard")
    ap.add_argument("--n", type=int, default=64, help="items per level")
    ap.add_argument("--levels", default="1,2,4,8,16")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(f"{args.data}/{args.split}.jsonl") if l.strip()]
    items = (items * ((args.n // len(items)) + 1))[:args.n]

    print(f"{len(items)} requests per level -> {args.url}")
    print(f"{'conc':>5} {'wall_s':>8} {'req/s':>8} {'p50_ms':>8} {'p95_ms':>8} {'mean_ms':>8} {'speedup':>8}")
    base_rps = None
    for conc in [int(x) for x in args.levels.split(",")]:
        lat = []
        def one(it):
            p = {"state": it["state"], "questions": {"q": build_question(it)}}
            t0 = time.perf_counter()
            try:
                post(args.url, p)
            except Exception as e:
                return None
            return time.perf_counter() - t0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for d in ex.map(one, items):
                if d is not None:
                    lat.append(d)
        wall = time.perf_counter() - t0
        if not lat:
            print(f"{conc:>5}  ALL FAILED"); continue
        lat.sort()
        rps = len(lat) / wall
        if base_rps is None:
            base_rps = rps
        print(f"{conc:>5} {wall:>8.1f} {rps:>8.2f} {lat[len(lat)//2]*1000:>8.0f} "
              f"{lat[min(len(lat)-1,int(len(lat)*0.95))]*1000:>8.0f} "
              f"{statistics.mean(lat)*1000:>8.0f} {rps/base_rps:>7.2f}x")
    return 0

if __name__ == "__main__":
    sys.exit(main())
