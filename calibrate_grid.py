#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Robust probability calibration for your gridded models (isotonic or sigmoid),
# compatible with scikit-learn >= 1.6 where CalibratedClassifierCV uses `estimator=`.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit


# ---------------- I/O helpers ----------------

def read_any(path: str) -> pd.DataFrame:
    """
    Load labelled grid from CSV(.gz) or Parquet.
    Ensures a usable 'time' column and leaves other columns unchanged.
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df = pd.read_parquet(path)
        if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
        return df

    # CSV / gz
    df = pd.read_csv(path, low_memory=False, compression="infer")
    # Defensive: handle BOM header
    if "time" not in df.columns and "\ufefftime" in df.columns:
        df = df.rename(columns={"\ufefftime": "time"})
    if "time" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


# ---------------- utils ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate grid model probabilities (isotonic or sigmoid).")
    ap.add_argument("--labelled", required=True, help="CSV/Parquet with labelled grid data.")
    ap.add_argument("--model-in", required=True, help="Input .pkl from train_grid_logit_from_csv.py")
    ap.add_argument("--model-out", required=True, help="Output .pkl path for calibrated model")
    ap.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for calibration")
    ap.add_argument("--max-tries", type=int, default=10, help="Retry splits to ensure both classes present")
    return ap.parse_args()

def finite_mask(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.isfinite(a).all(axis=1)
    return mask


# ---------------- main ----------------

def main():
    args = parse_args()

    # Load data (no spatial restriction; all lat/lon rows are kept)
    df = read_any(args.labelled)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in {args.labelled}")

    # Load model bundle
    bundle = joblib.load(args.model_in)
    model  = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["features"]

    # Ensure features exist and are numeric
    present = [c for c in feats if c in df.columns]
    if not present:
        raise ValueError(f"None of model features {feats} found in input.")
    present = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    if not present:
        raise ValueError("No numeric features remain after filtering.")

    # Build matrices
    X = df[present].to_numpy(dtype=float, copy=False)
    y = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).to_numpy()

    # Drop rows with any NaN/Inf
    m = finite_mask(X, y)
    X, y = X[m], y[m]
    if len(X) == 0:
        raise ValueError("No finite rows remain after filtering. Check your features for NaN/Inf.")

    # Scale (match training)
    Xs = scaler.transform(X)

    # Ensure both classes in split
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise ValueError(f"Calibration needs both classes. Counts: pos={pos}, neg={neg}")

    sss = StratifiedShuffleSplit(n_splits=args.max_tries, test_size=args.test_size, random_state=42)
    split = None
    for tr, te in sss.split(Xs, y):
        ytr, yte = y[tr], y[te]
        if ytr.sum() > 0 and (len(ytr) - ytr.sum()) > 0 and yte.sum() > 0 and (len(yte) - yte.sum()) > 0:
            split = (tr, te); break
    if split is None:
        raise ValueError("Could not find a split with both classes in train and test. Try adjusting --test-size.")

    tr, te = split
    Xtr, Xte = Xs[tr], Xs[te]
    ytr, yte = y[tr], y[te]

    # Uncalibrated performance (model expects scaled features)
    prob_raw = np.clip(model.predict_proba(Xte)[:, 1], 1e-8, 1 - 1e-8)
    try:
        auc_before = roc_auc_score(yte, prob_raw)
    except ValueError:
        auc_before = float("nan")
    brier_before = brier_score_loss(yte, prob_raw)
    print(f"Brier (before): {brier_before:.6f} | AUC: {auc_before:.3f} | test N={len(yte)} (pos={int(yte.sum())})")

    # Calibrate (Note: estimator=, cv="prefit" stays for now despite sklearn 1.6 deprecation warning)
    calib = CalibratedClassifierCV(estimator=model, method=args.method, cv="prefit")
    calib.fit(Xtr, ytr)

    prob_after = np.clip(calib.predict_proba(Xte)[:, 1], 1e-8, 1 - 1e-8)
    try:
        auc_after = roc_auc_score(yte, prob_after)
    except ValueError:
        auc_after = float("nan")
    brier_after = brier_score_loss(yte, prob_after)
    print(f"Brier (after):  {brier_after:.6f} | AUC: {auc_after:.3f}")

    # Save calibrated bundle (overwrite the model key so downstream tools work)
    bundle["model"] = calib
    bundle.setdefault("meta", {})
    bundle["meta"]["calibration"] = dict(method=args.method, test_size=args.test_size)
    joblib.dump(bundle, args.model_out)
    print(f"Saved calibrated model → {args.model_out}")

if __name__ == "__main__":
    main()