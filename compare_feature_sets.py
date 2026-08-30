#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_feature_sets.py — paired per-storm comparison of two battery runs.

Two independent AUC means with overlapping CIs can still hide a consistent
per-storm improvement (or hide the fact that a "gain" comes from one storm).
Because both runs score the SAME storms under the same CV blocking, the honest
test is paired: difference each storm's held-out AUC, then bootstrap the mean
difference by resampling storms.

USAGE
    python compare_feature_sets.py \\
        --baseline results/metrics/multiseason_strict_genesis_battery.json \\
        --candidate scratch/shearlow/multiseason_strict_genesis_battery.json \\
        --label-a "curated(4)" --label-b "curated+shear_low(5)"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(p: str):
    d = json.loads(Path(p).read_text())
    return d, d["discriminator"]["by_storm"]


def main() -> int:
    a = parse_args()
    da, base = load(a.baseline)
    db, cand = load(a.candidate)

    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit("[compare] no storms in common between the two runs.")
    x = np.array([base[k] for k in shared], float)
    y = np.array([cand[k] for k in shared], float)
    d = y - x
    ok = np.isfinite(d)
    x, y, d, shared = x[ok], y[ok], d[ok], [s for s, o in zip(shared, ok) if o]

    rng = np.random.default_rng(7)
    bm = [np.mean(d[rng.integers(0, len(d), size=len(d))]) for _ in range(a.n_boot)]
    lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
    wins = int((d > 0).sum())

    print(f"[compare] {len(shared)} shared storms "
          f"(cv={da.get('cv')}, label={da.get('label')})")
    print(f"  {a.label_a:26s} mean AUC = {x.mean():.4f}")
    print(f"  {a.label_b:26s} mean AUC = {y.mean():.4f}")
    print(f"  paired delta              = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  storms improved           = {wins}/{len(d)} ({wins/len(d):.0%})")
    verdict = ("CANDIDATE WINS" if lo > 0 else
               "BASELINE WINS" if hi < 0 else
               "no significant difference")
    print(f"  verdict                   = {verdict}")

    print(f"\n  {'storm':24s} {a.label_a:>12s} {a.label_b:>12s} {'delta':>8s}")
    for k, xi, yi, di in sorted(zip(shared, x, y, d), key=lambda t: -t[3]):
        print(f"  {k:24s} {xi:12.3f} {yi:12.3f} {di:+8.3f}")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "n_storms": len(d), "label_a": a.label_a, "label_b": a.label_b,
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "paired_delta": float(d.mean()), "ci": [lo, hi],
            "storms_improved": wins, "verdict": verdict,
            "by_storm": {k: {"a": float(xi), "b": float(yi), "delta": float(di)}
                         for k, xi, yi, di in zip(shared, x, y, d)},
        }, indent=2))
        print(f"\n[compare] wrote {a.out}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Paired per-storm comparison of two battery runs.")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="candidate")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
