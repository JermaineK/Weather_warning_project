#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a logistic regression storm/pregen classifier from a labelled grid CSV/Parquet.
- Keeps the classic "global" training (one target column).
- Optionally trains extra per-lead models if lead-specific target columns exist.
- Reuses the same imputer/clip/scaler stats for all models to avoid drift.

Outputs (joblib bundle):
  bundle = {
    "model": clf_global,                 # global (unchanged API)
    "scaler": StandardScaler,
    "features": [col, ...],
    "target": "pregen" | "storm" | "near_storm",
    "imputer_stats": {col: value, ...},
    "impute_kind": "median" | "mean" | "zero",
    "clip_stats": {"lo": np.ndarray, "hi": np.ndarray} or {},
    "clip_quantile": float,
    "class_weight": "none" | "balanced",
    "C": float,
    "meta": {...},
    # New, optional:
    "per_lead_models": {lead_h: clf, ...},   # only if --per-lead provided and labels found
  }
"""

import argparse
import re
import sys
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------- I/O helpers -----------------------

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
    df = pd.read_csv(path, parse_dates=["time"], compression="infer")
    # Defensive: handle BOM on header if present
    if "time" not in df.columns and "\ufefftime" in df.columns:
        df = df.rename(columns={"\ufefftime": "time"})
    return df

# ----------------------- metric helpers -----------------------

def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean((p - y_true) ** 2))

def nan_report(df: pd.DataFrame, top: int = 10) -> None:
    vals = df.isna().sum().sort_values(ascending=False)
    total = df.size
    cells_with_nan = int(df.isna().sum().sum())
    frac = cells_with_nan / total if total else 0.0
    print(f"[DATA] Non-finite -> NaN  | cells with NaN: {cells_with_nan:,}  ({frac:.3%} of all values)")
    if top > 0:
        print("       Top-NaN columns:")
        shown = 0
        for col, n in vals.items():
            if n > 0:
                print(f"         - {col:28s} NaNs={n:,}")
                shown += 1
                if shown >= top:
                    break

def quantile_clip_train(Xtr: np.ndarray, Xte: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Clip each feature by train-based quantiles to [1-q, q]."""
    if q is None or q <= 0 or q >= 1:
        return Xtr, Xte, {}
    lo = np.quantile(Xtr, 1 - q, axis=0)
    hi = np.quantile(Xtr, q, axis=0)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi), {"lo": lo, "hi": hi}

def apply_clip(arr: np.ndarray, clip_stats: Dict[str, np.ndarray]) -> np.ndarray:
    if not clip_stats:
        return arr
    lo, hi = clip_stats["lo"], clip_stats["hi"]
    return np.clip(arr, lo, hi)

def make_imputer(kind: str, fill_value: float = 0.0):
    """Return a simple columnwise imputer callable."""
    kind = (kind or "median").lower()
    def fit(X: pd.DataFrame) -> Dict[str, float]:
        if kind == "median":
            s = np.nanmedian(X, axis=0)
        elif kind == "mean":
            s = np.nanmean(X, axis=0)
        elif kind == "zero":
            s = np.zeros(X.shape[1], dtype=float)
        else:
            raise ValueError(f"Unsupported impute kind: {kind}")
        stats = {}
        for i, c in enumerate(X.columns):
            v = float(s[i]) if np.isfinite(s[i]) else fill_value
            stats[c] = v
        return stats
    def transform(X: pd.DataFrame, stats: Dict[str, float]) -> np.ndarray:
        arr = X.to_numpy(dtype=float, copy=True)
        for j, c in enumerate(X.columns):
            v = stats.get(c, fill_value)
            m = ~np.isfinite(arr[:, j])
            if m.any():
                arr[m, j] = v
        return arr
    return fit, transform

def auto_clean_columns(df: pd.DataFrame,
                       min_nonnull_frac: float = 0.98,
                       drop_regex: str = r"(?:_y$)|^(?:expver|number)$") -> Tuple[pd.DataFrame, List[str]]:
    """
    - Drops columns that match regex (dup/metadata).
    - Drops columns with too many NaNs (non-null fraction < threshold).
    - Removes perfectly constant columns.
    """
    dropped: List[str] = []

    if drop_regex:
        pat = re.compile(drop_regex)
        to_drop = [c for c in df.columns if pat.search(c)]
        if to_drop:
            df = df.drop(columns=to_drop, errors="ignore")
            dropped.extend(to_drop)

    if 0 < min_nonnull_frac < 1:
        nn_frac = df.notna().mean(axis=0)
        many_nan = nn_frac[nn_frac < min_nonnull_frac].index.tolist()
        if many_nan:
            df = df.drop(columns=many_nan, errors="ignore")
            dropped.extend(many_nan)

    nunique = df.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df = df.drop(columns=const_cols, errors="ignore")
        dropped.extend(const_cols)

    return df, dropped

def pick_feature_columns(df: pd.DataFrame, target: str, extra_ignores: Optional[List[str]] = None) -> List[str]:
    meta = {"time", "lat", "lon"}
    ignore = set([target]) | meta | set(extra_ignores or [])
    feats = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    return feats

# ----------------------- arg parsing -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train logistic regression from labelled grid CSV/Parquet (with auto-clean).")
    p.add_argument("--labelled", required=True, help="Input labelled CSV(.gz) or Parquet")
    p.add_argument("--target", choices=["pregen", "storm", "near_storm"], default="pregen")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--model-out", required=True, help="Output .pkl")

    # Per-lead (optional)
    p.add_argument("--per-lead", type=int, nargs="+", default=None,
                   help="Optional list of lead hours to train extra models for (requires lead-specific target columns).")

    # Preprocess
    p.add_argument("--impute", choices=["median", "mean", "zero"], default="median")
    p.add_argument("--clip-quantile", type=float, default=0.999,
                   help="Clip to [1-q, q] per feature based on train (default 0.999). Set outside (0,1) to disable.")

    # Cleaning
    p.add_argument("--auto-clean", action="store_true",
                   help="Enable automatic dropping of duplicate/meta/low-nonnull/constant columns.")
    p.add_argument("--min-nonnull-frac", type=float, default=0.98,
                   help="When --auto-clean, drop columns with non-null fraction lower than this (default 0.98).")
    p.add_argument("--drop-cols", default=None,
                   help="Optional regex for additional columns to drop AFTER auto-clean.")

    # Model
    p.add_argument("--warm", type=float, default=1.0, help="C inverse regularization scaling (1/C).")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="Optional class weighting for LogisticRegression.")
    return p.parse_args()

# ----------------------- training helpers -----------------------

def _fit_logit(Xtr_s, ytr, C, class_weight):
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=200,
        n_jobs=None,
        class_weight=class_weight,
        random_state=42,
    )
    clf.fit(Xtr_s, ytr)
    return clf

def _safe_auc(ytrue, p):
    try:
        return roc_auc_score(ytrue, p)
    except Exception:
        return float("nan")

def _safe_ap(ytrue, p):
    try:
        return average_precision_score(ytrue, p)
    except Exception:
        return float("nan")

def _find_lead_label_column(df: pd.DataFrame, base: str, lead: int) -> Optional[str]:
    """
    Try a few common patterns for lead-specific target columns.
    Returns the column name if found, else None.
    """
    cand = [
        f"{base}_lead{lead}",
        f"{base}_L{lead}",
        f"{base}_{lead}h",
        f"{base}{lead}",
        f"{base}_tplus{lead}",
    ]
    for c in cand:
        if c in df.columns:
            return c
    return None

# ----------------------- main -----------------------

def main():
    args = parse_args()

    # Read + normalize obvious bad values early
    df = read_any(args.labelled)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if args.target not in df.columns:
        print(f"[ERROR] target '{args.target}' not in dataframe.", file=sys.stderr)
        sys.exit(2)

    nan_report(df, top=10)

    # Auto-clean
    if args.auto_clean:
        df, dropped_auto = auto_clean_columns(
            df,
            min_nonnull_frac=args.min_nonnull_frac,
            drop_regex=r"(?:_y$)|^(?:expver|number)$"
        )
        removed = sorted(set(dropped_auto))
        if removed:
            head = ", ".join(removed[:12])
            print(f"[CLEAN] auto dropped {len(removed)} cols: {head}{' ...' if len(removed) > 12 else ''}")

    # Extra user regex drop AFTER auto-clean
    extra_ign = []
    if args.drop_cols:
        pat = re.compile(args.drop_cols)
        to_drop = [c for c in df.columns if pat.search(c)]
        if to_drop:
            df = df.drop(columns=to_drop, errors="ignore")
            head = ", ".join(to_drop[:12])
            print(f"[CLEAN] regex dropped {len(to_drop)} cols: {head}{' ...' if len(to_drop) > 12 else ''}")
        extra_ign.extend(to_drop)

    # Feature set (ignore all obvious meta + target + any extra drops)
    feats = pick_feature_columns(df, target=args.target, extra_ignores=extra_ign)
    if not feats:
        print("[ERROR] No numeric feature columns after cleaning.", file=sys.stderr)
        sys.exit(2)
    print(f"[FEATS] using {len(feats)} features (first 8): {feats[:8]}{' ...' if len(feats) > 8 else ''}")

    # Split GLOBAL
    X = df[feats]
    y = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).to_numpy()
    try:
        Xtr_df, Xte_df, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback: no stratify if class is degenerate
        Xtr_df, Xte_df, ytr, yte = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=None
        )

    # Impute (fit on GLOBAL train)
    fit_imp, tr_imp = make_imputer(args.impute)
    imp_stats = fit_imp(Xtr_df)
    Xtr = tr_imp(Xtr_df, imp_stats)
    Xte = tr_imp(Xte_df, imp_stats)

    # Clip (fit on GLOBAL train)
    q = args.clip_quantile
    Xtr, Xte, clip_stats = quantile_clip_train(Xtr, Xte, q) if (q and 0 < q < 1) else (Xtr, Xte, {})

    # Scale (fit on GLOBAL train)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Model hyperparams
    class_weight = None if args.class_weight == "none" else "balanced"
    C = 1.0 / max(args.warm, 1e-6)

    # ---- Train GLOBAL model
    clf_global = _fit_logit(Xtr_s, ytr, C, class_weight)
    ptr = clf_global.predict_proba(Xtr_s)[:, 1]
    pte = clf_global.predict_proba(Xte_s)[:, 1]
    mtr = dict(AUC=_safe_auc(ytr, ptr), PRAUC=_safe_ap(ytr, ptr), Brier=brier(ytr, ptr))
    mte = dict(AUC=_safe_auc(yte, pte), PRAUC=_safe_ap(yte, pte), Brier=brier(yte, pte))

    pos = int(y.sum())
    print(f"Rows used (GLOBAL): {len(X):,}  Positives: {pos:,}/{len(X):,}")
    print(f"[train][GLOBAL] AUC={mtr['AUC']:.3f}  PRAUC={mtr['PRAUC']:.3f}  Brier={mtr['Brier']:.3f}")
    print(f"[test] [GLOBAL] AUC={mte['AUC']:.3f}  PRAUC={mte['PRAUC']:.3f}  Brier={mte['Brier']:.3f}")

    # Top coefficients (GLOBAL)
    coef = clf_global.coef_.ravel()
    order = np.argsort(np.abs(coef))[::-1][:20]
    print("\nTop |coef| (GLOBAL, 20):")
    for idx in order:
        print(f"  {feats[idx]:28s} {coef[idx]:+0.4f}")

    # ---- Optional: train PER-LEAD models (if columns exist)
    per_lead_models: Dict[int, LogisticRegression] = {}
    if args.per_lead:
        print("\n[per-lead] attempting per-lead training using existing lead label columns…")
        for L in args.per_lead:
            y_col = _find_lead_label_column(df, args.target, L)
            if not y_col:
                print(f"  • lead {L:>3}h: skipped (no column like '{args.target}_lead{L}' / 'L{L}' / '{L}h').")
                continue

            yL = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).to_numpy()
            try:
                Xtr_dfL, Xte_dfL, ytrL, yteL = train_test_split(
                    X, yL, test_size=args.test_size, random_state=42, stratify=yL
                )
            except ValueError:
                Xtr_dfL, Xte_dfL, ytrL, yteL = train_test_split(
                    X, yL, test_size=args.test_size, random_state=42, stratify=None
                )

            # Reuse global imputer/clip/scaler stats so features live in same space
            XtrL = apply_clip(tr_imp(Xtr_dfL, imp_stats), clip_stats)
            XteL = apply_clip(tr_imp(Xte_dfL, imp_stats), clip_stats)
            XtrLs = scaler.transform(XtrL)
            XteLs = scaler.transform(XteL)

            # Train
            clfL = _fit_logit(XtrLs, ytrL, C, class_weight)
            ptrL = clfL.predict_proba(XtrLs)[:, 1]
            pteL = clfL.predict_proba(XteLs)[:, 1]
            mtrL = dict(AUC=_safe_auc(ytrL, ptrL), PRAUC=_safe_ap(ytrL, ptrL), Brier=brier(ytrL, ptrL))
            mteL = dict(AUC=_safe_auc(yteL, pteL), PRAUC=_safe_ap(yteL, pteL), Brier=brier(yteL, pteL))

            posL = int(yL.sum())
            print(f"[per-lead {L:>3}h] N={len(yL):,}  Pos={posL:,} | "
                  f"train: AUC={mtrL['AUC']:.3f} PRAUC={mtrL['PRAUC']:.3f} Brier={mtrL['Brier']:.3f} | "
                  f"test:  AUC={mteL['AUC']:.3f} PRAUC={mteL['PRAUC']:.3f} Brier={mteL['Brier']:.3f}")

            per_lead_models[int(L)] = clfL

    # Save bundle (GLOBAL, plus optional per-lead models)
    artifact = dict(
        model=clf_global,
        scaler=scaler,
        features=feats,
        target=args.target,
        imputer_stats=imp_stats,
        impute_kind=args.impute,
        clip_stats=clip_stats,
        clip_quantile=args.clip_quantile,
        class_weight=args.class_weight,
        C=C,
        meta=dict(
            test_size=args.test_size,
            auto_clean=args.auto_clean,
            min_nonnull_frac=args.min_nonnull_frac,
            drop_cols=args.drop_cols,
            per_lead_trained=sorted(per_lead_models.keys()),
        ),
    )
    if per_lead_models:
        artifact["per_lead_models"] = per_lead_models

    joblib.dump(artifact, args.model_out)
    print(f"\nSaved {args.model_out}")

if __name__ == "__main__":
    main()