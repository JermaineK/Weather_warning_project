#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate_grid.py — robust probability calibration for gridded models.

Upgrades:
  • Optional group-aware holdout (--group-col) to reduce time/space leakage
  • Safe pre-metrics when estimator lacks predict_proba (uses decision_function)
  • Optional sample weights (--weight-col)
  • Reliability curve CSV (+ optional PNG)
  • Calibration metrics persisted inside bundle['meta']['calibration']['metrics']
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_array


# ---------------- I/O helpers ----------------

def read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False, compression="infer")
        if "time" not in df.columns and "\ufefftime" in df.columns:
            df = df.rename(columns={"\ufefftime": "time"})
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


# ---------------- utils ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate grid model probabilities (isotonic or sigmoid).")
    ap.add_argument("--labelled", required=True, help="CSV/Parquet with labelled grid data.")
    ap.add_argument("--model-in", required=True, help="Input .pkl from training step")
    ap.add_argument("--model-out", required=True, help="Output .pkl path for calibrated model")
    ap.add_argument("--target", required=True, choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    ap.add_argument("--test-size", type=float, default=0.20, help="Holdout fraction (0 < test ≤ 0.5)")
    ap.add_argument("--max-tries", type=int, default=12, help="Retry splits to ensure both classes present")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    ap.add_argument("--group-col", default=None,
                    help="Optional grouping column for GroupShuffleSplit (e.g., 'day', 'storm_id'). "
                         "If omitted, will try day-blocks from 'time'.")
    ap.add_argument("--weight-col", default=None, help="Optional sample-weight column for calibration fit.")
    ap.add_argument("--reliability-out", default=None,
                    help="If set, write reliability CSV (and PNG if endswith .png).")
    return ap.parse_args()


def finite_mask(*arrays):
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 1:
            mask &= np.isfinite(a)
        else:
            mask &= np.isfinite(a).all(axis=1)
    return mask


def clip01(p, eps=1e-8):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def reliability_bins(y_true, y_prob, nbins=15):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.minimum(np.searchsorted(edges, y_prob, side="right") - 1, nbins - 1)
    rows = []
    for b in range(nbins):
        sel = (idx == b)
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(bin=b, left=edges[b], right=edges[b+1], n=0,
                             prob_mean=np.nan, pos_rate=np.nan))
            continue
        rows.append(dict(
            bin=b, left=float(edges[b]), right=float(edges[b+1]), n=n,
            prob_mean=float(y_prob[sel].mean()),
            pos_rate=float(y_true[sel].mean())
        ))
    return pd.DataFrame(rows)


def get_probs_before(model, X):
    # Try predict_proba; fallback to decision_function + logistic squash (for AUC/Brier baseline only)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is None:
            pass
        else:
            if proba.ndim == 2 and proba.shape[1] == 2:
                return clip01(proba[:, 1])
            if proba.ndim == 1:
                return clip01(proba)
    if hasattr(model, "decision_function"):
        m = np.asarray(model.decision_function(X), dtype=float)
        # approximate Platt squashing for baseline visualization
        return clip01(1.0 / (1.0 + np.exp(-m)))
    raise AttributeError("Estimator exposes neither predict_proba nor decision_function.")


# ---------------- main ----------------

def main():
    args = parse_args()

    if not (0.0 < args.test_size <= 0.5):
        raise ValueError("--test-size should be in (0, 0.5].")

    df = read_any(args.labelled)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in {args.labelled}")

    bundle = joblib.load(args.model_in)
    if not isinstance(bundle, dict):
        raise ValueError(f"{args.model_in}: expected a dict-like bundle.")

    try:
        base_model = bundle["model"]
        scaler = bundle["scaler"]
        feats = list(bundle["features"])
    except KeyError as e:
        raise ValueError(f"{args.model_in}: missing key in bundle: {e}")

    # Feature sanity
    missing = [c for c in feats if c not in df.columns]
    if missing:
        warnings.warn(f"{len(missing)} features missing from labelled data; ignoring: {missing[:8]}{'...' if len(missing)>8 else ''}")
    present = [c for c in feats if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not present:
        raise ValueError("No usable numeric features from bundle['features'] found in labelled data.")

    X = df[present].to_numpy(dtype=float, copy=False)
    y = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).to_numpy()

    m = finite_mask(X, y)
    X, y = X[m], y[m]
    if X.size == 0:
        raise ValueError("No finite rows remain after filtering.")

    # Optional weights
    sample_weight = None
    if args.weight-col if False else args.weight_col:  # keep linter happy
        wc = args.weight_col
        if wc not in df.columns:
            warnings.warn(f"--weight-col '{wc}' not found; proceeding unweighted.")
        else:
            w = pd.to_numeric(df.loc[m, wc], errors="coerce").to_numpy()
            sample_weight = np.where(np.isfinite(w) & (w >= 0), w, np.nan)
            if np.isnan(sample_weight).any():
                sample_weight = np.where(np.isnan(sample_weight), np.nanmedian(sample_weight), sample_weight)

    # Scale like training
    Xs = scaler.transform(X)
    Xs = check_array(Xs, accept_sparse=False, dtype=float)

    pos = int(y.sum()); neg = len(y) - pos
    prev = pos / max(1, len(y))
    if pos == 0 or neg == 0:
        raise ValueError(f"Calibration needs both classes. Counts: pos={pos}, neg={neg}")
    if prev < 0.01 or prev > 0.99:
        warnings.warn(f"Extreme class imbalance (prevalence={prev:.4f}); isotonic may overfit with tiny holdouts.")

    # Groups for leakage control
    groups = None
    if args.group_col and args.group_col in df.columns:
        groups = df.loc[m, args.group_col].to_numpy()
    elif "time" in df.columns:
        # default: day blocks from time
        t = pd.to_datetime(df.loc[m, "time"], errors="coerce")
        groups = t.dt.floor("D").astype(str).to_numpy()

    split = None
    if groups is not None and len(np.unique(groups)) > 1:
        gss = GroupShuffleSplit(n_splits=args.max_tries, test_size=args.test_size, random_state=args.random_state)
        for tr, te in gss.split(Xs, y, groups=groups):
            ytr, yte = y[tr], y[te]
            if ytr.sum() > 0 and (len(ytr) - ytr.sum()) > 0 and yte.sum() > 0 and (len(yte) - yte.sum()) > 0:
                split = (tr, te); break
    if split is None:
        sss = StratifiedShuffleSplit(n_splits=args.max_tries, test_size=args.test_size, random_state=args.random_state)
        for tr, te in sss.split(Xs, y):
            ytr, yte = y[tr], y[te]
            if ytr.sum() > 0 and (len(ytr) - ytr.sum()) > 0 and yte.sum() > 0 and (len(yte) - yte.sum()) > 0:
                split = (tr, te); break
    if split is None:
        raise ValueError("Could not find a split with both classes in train and test. Try adjusting --test-size.")

    tr, te = split
    Xtr, Xte = Xs[tr], Xs[te]
    ytr, yte = y[tr], y[te]
    wtr = None if sample_weight is None else sample_weight[tr]

    # Pre-metrics (uncalibrated)
    prob_raw = get_probs_before(base_model, Xte)
    try:
        auc_before = roc_auc_score(yte, prob_raw)
    except ValueError:
        auc_before = float("nan")
    brier_before = brier_score_loss(yte, clip01(prob_raw))

    print(f"Brier (before): {brier_before:.6f} | AUC: {auc_before:.3f} | test N={len(yte)} (pos={int(yte.sum())})")

    # Calibrate
    calib = CalibratedClassifierCV(estimator=base_model, method=args.method, cv="prefit")
    calib.fit(Xtr, ytr, sample_weight=wtr)

    prob_after = clip01(calib.predict_proba(Xte)[:, 1])
    try:
        auc_after = roc_auc_score(yte, prob_after)
    except ValueError:
        auc_after = float("nan")
    brier_after = brier_score_loss(yte, prob_after)
    print(f"Brier (after):  {brier_after:.6f} | AUC: {auc_after:.3f}")

    # Reliability diagnostics
    rel_before = reliability_bins(yte, prob_raw, nbins=15)
    rel_after  = reliability_bins(yte, prob_after, nbins=15)
    if args.reliability_out:
        out = Path(args.reliability_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rel_before.assign(which="before").to_csv(out.with_suffix(".before.csv"), index=False)
        rel_after.assign(which="after").to_csv(out.with_suffix(".after.csv"), index=False)
        if out.suffix.lower() == ".png":
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.6,label="perfect")
                ax.plot(rel_before["prob_mean"], rel_before["pos_rate"], "o-", label="before")
                ax.plot(rel_after["prob_mean"],  rel_after["pos_rate"],  "o-", label="after")
                ax.set_xlabel("Predicted probability (bin mean)")
                ax.set_ylabel("Empirical event rate")
                ax.set_title(f"Reliability ({args.method})")
                ax.grid(True, ls=":", alpha=0.5)
                ax.legend()
                fig.tight_layout()
                fig.savefig(out, dpi=150)
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Reliability plot failed: {e}")

    # Save calibrated bundle
    bundle["raw_model"] = bundle.get("raw_model", bundle["model"])
    bundle["model"] = calib
    meta = bundle.setdefault("meta", {})
    meta["calibration"] = {
        "method": args.method,
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "group_col": args.group_col or "time->day" if groups is not None else None,
        "per_lead": {},
        "metrics": {
            "prevalence": float(prev),
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "auc_before": float(auc_before) if np.isfinite(auc_before) else None,
            "auc_after": float(auc_after) if np.isfinite(auc_after) else None,
            "n_test": int(len(yte)),
            "n_pos_test": int(yte.sum()),
        }
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print(f"Saved calibrated model -> {args.model_out}")

if __name__ == "__main__":
    main()