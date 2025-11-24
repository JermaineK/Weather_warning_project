#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_storm_probability.py

Compute a storm-likelihood probability from a slim feature table.

This is the *prediction* companion to thresholds_and_alerts.py.
It performs NO thresholding. It simply computes:

    raw_score = sum_j w_j * x_j
    prob      = logistic( (raw_score - shift) / temp )

Why separate?
-------------
- thresholds_and_alerts.py is for TRAINING/TUNING thresholds.
- When you flip to PREDICTION, you want:
      alerts = P(storm | features)
  and let downstream users or ops decide thresholds.

Inputs
------
A CSV(.gz) with:
    row_id, feat1, feat2, ..., featN, [time,lat,lon]

Outputs
-------
A CSV(.gz) with columns:
    row_id, time, lat, lon, raw_score, prob_storm, [features if requested]

CLI knobs
---------
--feature-cols         explicit list, otherwise auto-detect numeric
--weights-json         per-feature weights JSON string or file
--default-weight       weight for any unlisted feature
--shift                center point before logistic
--temp                 temperature scaling (controls steepness)
--keep-features        include the input feature columns
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------- small utilities ----------------

def _print(*a, **k):
    print(*a, **k, flush=True)

def _normalize_explicit_cols(cols: Optional[List[str]]) -> Optional[List[str]]:
    if not cols:
        return None
    if len(cols) == 1 and isinstance(cols[0], str) and "," in cols[0]:
        return [c.strip() for c in cols[0].split(",") if c.strip()]
    return cols

def detect_feature_columns(df: pd.DataFrame, id_col: str,
                           explicit_cols: Optional[List[str]]) -> List[str]:
    explicit_cols = _normalize_explicit_cols(explicit_cols)
    cols = list(df.columns)

    if explicit_cols:
        feats = [c for c in explicit_cols if c in cols and c != id_col]
        if not feats:
            raise ValueError(
                f"None of requested columns found: {explicit_cols}"
            )
        return feats

    skip = {id_col, "time", "lat", "lon"}
    numeric = []
    for c in cols:
        if c in skip:
            continue
        try:
            v = pd.to_numeric(df[c].head(200), errors="coerce")
            if v.notna().any():
                numeric.append(c)
        except Exception:
            continue
    if not numeric:
        raise ValueError("Could not auto-detect numeric columns.")
    return numeric

def load_weights(spec: Optional[str], feat_cols: List[str], default_w: float) -> Dict[str, float]:
    if not spec:
        return {c: float(default_w) for c in feat_cols}

    try:
        obj = json.loads(spec)
    except json.JSONDecodeError:
        txt = Path(spec).read_text(encoding="utf-8")
        obj = json.loads(txt)

    out = {}
    for c in feat_cols:
        out[c] = float(obj.get(c, default_w))
    return out

def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------- prediction core ----------------

def predict_chunk(
    chunk: pd.DataFrame,
    id_col: str,
    feat_cols: List[str],
    weights: Dict[str, float],
    score_col: str,
    prob_col: str,
    shift: float,
    temp: float,
) -> pd.DataFrame:

    out = pd.DataFrame()
    out[id_col] = chunk[id_col].copy()

    raw = np.zeros(len(chunk), dtype="float64")

    # Linear score
    for c in feat_cols:
        if c not in chunk.columns:
            continue
        v = pd.to_numeric(chunk[c], errors="coerce").fillna(0).values
        w = float(weights.get(c, 0.0))
        raw += w * v

    out[score_col] = raw

    # Logistic transform
    z = (raw - shift) / (temp if temp != 0 else 1.0)
    out[prob_col] = logistic(z)

    return out


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Compute storm-likelihood probabilities.")
    ap.add_argument("--scoring-src", required=True,
                    help="Slim table: row_id + features (+time/lat/lon optional).")
    ap.add_argument("--out", default="results/alerts_predicted.csv.gz",
                    help="Output CSV(.gz) with probabilities.")

    ap.add_argument("--id-col", default="row_id")
    ap.add_argument("--feature-cols", nargs="*", default=None)
    ap.add_argument("--weights-json", default=None)
    ap.add_argument("--default-weight", type=float, default=40.0)

    ap.add_argument("--score-col", default="raw_score")
    ap.add_argument("--prob-col", default="prob_storm")

    # logistic controls
    ap.add_argument("--shift", type=float, default=0.0,
                    help="Center point before logistic.")
    ap.add_argument("--temp", type=float, default=200.0,
                    help="Temperature scaling (higher = flatter logistic).")

    ap.add_argument("--chunk-rows", type=int, default=400000)
    ap.add_argument("--keep-features", action="store_true")

    args = ap.parse_args()

    src = Path(args.scoring_src)
    if not src.exists():
        raise SystemExit(f"[fatal] missing scoring-src: {src}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if out.suffix.endswith("gz") else "infer"

    # sample
    sample = pd.read_csv(src, nrows=min(args.chunk_rows, 20000), compression="infer")
    feat_cols = detect_feature_columns(sample, id_col=args.id_col,
                                       explicit_cols=args.feature_cols)
    weights = load_weights(args.weights_json, feat_cols, args.default_weight)

    _print(f"[init] features={feat_cols}")
    _print(f"[init] weights default={args.default_weight}")
    _print(f"[init] shift={args.shift} temp={args.temp}")
    _print(f"[init] writing → {out}")

    rdr = pd.read_csv(src, chunksize=args.chunk_rows, compression="infer",
                      low_memory=False)
    if not hasattr(rdr, "__iter__"):
        rdr = [rdr]

    first = True
    total = 0

    for i, chunk in enumerate(rdr, start=1):
        if chunk is None or chunk.empty:
            continue

        pred = predict_chunk(
            chunk=chunk,
            id_col=args.id_col,
            feat_cols=feat_cols,
            weights=weights,
            score_col=args.score_col,
            prob_col=args.prob_col,
            shift=float(args.shift),
            temp=float(args.temp),
        )

        # carry geo/time
        for c in ("time", "lat", "lon"):
            if c in chunk.columns:
                pred[c] = chunk[c].values

        # optional: carry features
        if args.keep_features:
            for c in feat_cols:
                if c in chunk.columns and c not in pred.columns:
                    pred[c] = chunk[c].values

        pred.to_csv(out, index=False,
                    mode="w" if first else "a",
                    header=first,
                    compression=compression,
                    date_format="%Y-%m-%d %H:%M:%S")
        first = False
        total += len(pred)

        if i % 10 == 0:
            _print(f"[chunk {i}] total={total:,}")

    _print(f"[done] wrote {total:,} rows → {out}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass