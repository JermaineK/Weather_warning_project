#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_leadtime_grid.py

Evaluate a single trained grid model against a labelled grid using
strict-future lead windows.

For each point (lat, lon, time):

  • Coincident target: y(t) = target at time t
  • Future window for lead H: y_H(t) = max(target) over (t, t+H]

The script:
  1. Loads a model bundle: {'features', 'model', 'scaler' (optional)}.
  2. Predicts probabilities p(t) at each (lat, lon, time).
  3. Computes coincident metrics on target(t).
  4. For each lead horizon H in --lead-hours:
       - builds y_H(t) via strict future window
       - prints AUC, PRAUC, and Brier for (y_H, p).

This gives a quick sense of how far ahead the model carries useful signal.

Usage:
  python eval_leadtime_grid.py \
    --labelled data/grid_labelled.parquet \
    --model models/global_logit.pkl \
    --target storm \
    --lead-hours 6 12 24 48
"""

import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning,
)
pd.options.mode.copy_on_write = True


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate lead-time skill of a grid model using strict-future windows."
    )
    ap.add_argument(
        "--labelled",
        required=True,
        help="CSV(.gz)/Parquet with lat, lon, time, target + features.",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="joblib bundle: {'model', 'features', 'scaler'(optional)}.",
    )
    ap.add_argument(
        "--target",
        required=True,
        choices=["storm", "near_storm", "pregen"],
        help="Target column to treat as event indicator.",
    )
    ap.add_argument(
        "--lead-hours",
        nargs="+",
        type=int,
        default=[24, 48],
        help="Lead horizons in hours, e.g. 6 12 24 48.",
    )
    return ap.parse_args()


def load_any(path: str) -> pd.DataFrame:
    lower = str(path).lower()
    if lower.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["time"], low_memory=False)


def load_model(path):
    m = joblib.load(path)
    features = m["features"]
    scaler = m.get("scaler", None)
    clf = m["model"]
    return features, scaler, clf


def prep_df(labelled_path, use, target):
    df = load_any(labelled_path)

    # time → UTC tz-naive for consistent time-based rolling
    if "time" not in df.columns:
        raise ValueError("Missing 'time' column in labelled file.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)

    need = list({*use, "lat", "lon", "time", target})
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Labelled file missing columns: {missing}")

    df = df[need].copy()
    df = df.sort_values(["lat", "lon", "time"], kind="mergesort")
    df = df.dropna(subset=use + [target]).reset_index(drop=True)
    return df


def predict_probs(df, use, sc, clf):
    X = df[use].to_numpy(float)
    if sc is not None:
        X = sc.transform(X)
    return clf.predict_proba(X)[:, 1]


def future_max_within_hours(df, label_col, hours: int) -> np.ndarray:
    """
    For each (lat, lon), compute max(label) over the STRICT future window (t, t+hours].

    Implementation:
      • For each (lat, lon) group:
          - reverse time, so (t, t+H] becomes a past-looking window
          - shift by +1 to exclude current hour
          - use time-based rolling window of size H hours
      • Return a 1-D int array aligned to df.index.
    """
    out_parts = []
    for (_, _), g in df.groupby(["lat", "lon"], sort=False):
        y = (
            pd.to_numeric(g[label_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .reset_index(drop=True)
        )
        t = g["time"].reset_index(drop=True)

        # Reverse so a forward-looking window becomes a past-rolling;
        # shift(1) to EXCLUDE the current instant from the future window.
        y_rev = y.iloc[::-1]
        t_rev = t.iloc[::-1]
        y_rev.index = t_rev  # time-based index in hours
        fut_max_rev = y_rev.shift(1).rolling(f"{int(hours)}h", min_periods=1).max()

        fut_max = fut_max_rev.iloc[::-1].reset_index(drop=True).fillna(0).astype(int)
        out_parts.append(pd.Series(fut_max.values, index=g.index))

    out = pd.concat(out_parts).sort_index()
    return out.to_numpy(dtype=int)


def safe_metrics(y, p):
    """AUC/PRAUC/Brier that won't explode on degenerate y."""
    mets = {}
    try:
        mets["AUC"] = roc_auc_score(y, p)
    except Exception:
        mets["AUC"] = np.nan
    try:
        mets["PRAUC"] = average_precision_score(y, p)
    except Exception:
        mets["PRAUC"] = np.nan
    try:
        mets["Brier"] = brier_score_loss(y, p)
    except Exception:
        mets["Brier"] = np.nan
    return mets


def main():
    args = parse_args()
    use, sc, clf = load_model(args.model)
    df = prep_df(args.labelled, use, args.target)

    base = df[args.target].astype(int).to_numpy()
    print(
        f"Rows evaluated: {len(df):,}  "
        f"Positives (coincident {args.target}): {base.sum():,}"
    )

    p = predict_probs(df, use, sc, clf)

    # Coincident metrics (info only)
    m0 = safe_metrics(base, p)
    if not np.isnan(m0["AUC"]):
        print(
            f"[COINCIDENT] AUC={m0['AUC']:.3f}  "
            f"PRAUC={m0['PRAUC']:.3f}  "
            f"Brier={m0['Brier']:.3f}"
        )

    # Per-lead evaluation with strict-future windows
    for h in args.lead_hours:
        y = future_max_within_hours(df, args.target, hours=h)
        m = safe_metrics(y, p)
        pos = int(y.sum())
        print(
            f"Lead +{h:>3}h  →  "
            f"AUC={m['AUC']:.3f}  "
            f"PRAUC={m['PRAUC']:.3f}  "
            f"Brier={m['Brier']:.3f}  "
            f"Pos={pos:,}/{len(y):,}"
        )


if __name__ == "__main__":
    main()