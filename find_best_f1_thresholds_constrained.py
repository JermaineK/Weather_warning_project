#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import os

# --------- lightweight I/O (CSV/GZ + Parquet) ---------
def read_any(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet. If `columns` is provided, tries to read only those columns
    (fast for Parquet; CSV uses usecols filter).
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        # Parquet supports column projection natively
        return pd.read_parquet(path, columns=columns if columns else None)
    if columns:
        cols_set = set(columns)
        return pd.read_csv(path, compression="infer", low_memory=False,
                           usecols=lambda c: c in cols_set)
    return pd.read_csv(path, compression="infer", low_memory=False)

# --------- time/lon helpers ---------

def _try_parse_time_raw(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    # Be lenient: strip Z, try ISO, try provided fmt, try epoch s/ms, then generic
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (isinstance(mid, (int,float)) and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce")
    return t4.dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def _parse_area(aoi: Optional[str]):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")

# --------- utils ---------

def load_model_any(path: str):
    """
    Accept:
      - dict bundle with keys: model (or model-out/estimator), scaler, features[, imputer_stats]
      - plain estimator with optional attributes scaler_ / features_
    """
    m = joblib.load(path)
    if isinstance(m, dict):
        # Prefer 'model' if present, else support older key 'model-out' or 'estimator'
        model = m.get("model") or m.get("model-out") or m.get("estimator")
        scaler = m.get("scaler", None)
        feats  = m.get("features", m.get("feats", None))
        imp    = m.get("imputer_stats", None)
        if model is None:
            raise ValueError("Loaded bundle lacks a usable estimator ('model' / 'model-out' / 'estimator').")
        return model, scaler, feats, imp
    # plain estimator path
    return m, getattr(m, "scaler_", None), getattr(m, "features_", None), None

def build_feature_matrix(df: pd.DataFrame,
                         model,
                         scaler,
                         feats: Optional[List[str]],
                         imp_stats: Optional[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    # Feature selection / ordering
    if feats is not None:
        Xdf = df.reindex(columns=list(feats), fill_value=np.nan)
        feats = list(feats)
    else:
        drop = {"time","lat","lon","storm","near_storm","pregen"}
        num = df.select_dtypes(include=[np.number]).columns
        feats = [c for c in num if c not in drop]
        Xdf = df[feats].copy()

    # Replace ±inf -> NaN for consistent imputation
    Xnp = Xdf.to_numpy(dtype=float, copy=True)
    Xnp[~np.isfinite(Xnp)] = np.nan

    # Impute: prefer trainer stats when provided
    if imp_stats:
        filled = 0
        for j, c in enumerate(feats):
            fill = float(imp_stats.get(c, 0.0))
            m = ~np.isfinite(Xnp[:, j])
            if m.any():
                Xnp[m, j] = fill
                filled += int(m.sum())
        print(f"[IMPUTE] used trainer stats; filled NaNs/∞: {filled}")
    else:
        n0 = int(np.isnan(Xnp).sum())
        if n0 > 0:
            med = np.nanmedian(Xnp, axis=0)
            ridx, cidx = np.where(np.isnan(Xnp))
            Xnp[ridx, cidx] = med[cidx]
            print(f"[IMPUTE] median filled NaNs: {n0} (remaining: {int(np.isnan(Xnp).sum())})")
        else:
            print("[IMPUTE] no NaNs found.")

    # Optional scaler
    if scaler is not None:
        try:
            X = scaler.transform(Xnp)
        except Exception:
            X = Xnp
    else:
        X = Xnp
    return X, feats

def future_max_label_by_point(df: pd.DataFrame, target_col: str, hours: int) -> np.ndarray:
    """
    For each (lat,lon) time series, compute future max within `hours`.
    Aligned to the original row order; robust to duplicates and irregularities.
    """
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = _try_parse_time_raw(df["time"], None)

    def _per_point(g: pd.DataFrame) -> pd.Series:
        g = g.dropna(subset=["time"]).sort_values("time", kind="mergesort")
        vals = g[target_col].astype(int).to_numpy()[::-1]
        times = g["time"].to_numpy()[::-1]
        rev = pd.DataFrame({"x": vals, "time": times})
        win = rev.rolling(f"{hours}h", on="time", min_periods=1).max()["x"]
        out = win.iloc[::-1].to_numpy(dtype=int)
        return pd.Series(out, index=g.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        s = df.groupby(["lat","lon"], sort=False).apply(_per_point)
    s.index = s.index.droplevel([0,1])
    return s.to_numpy(dtype=int)

def sweep_thresholds(p: np.ndarray, y: np.ndarray, q_grid: Optional[int] = 1000) -> pd.DataFrame:
    """
    Build a metrics table across thresholds using a quantile grid for stability.
    """
    n = len(p)
    if q_grid is not None and q_grid > 1:
        qs = np.linspace(0, 1, q_grid + 1)
        thrs = np.unique(np.quantile(p, qs))
    else:
        thrs = np.unique(p)

    thrs = np.clip(thrs, 0.0, 1.0)

    # Sort by prob once for fast cumulative counts
    order = np.argsort(p)[::-1]
    p_sorted = p[order]
    y_sorted = y[order].astype(int)

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    Ptot = y_sorted.sum()
    idx = np.searchsorted(-p_sorted, -thrs, side="left")

    tp = np.where(idx > 0, tp_cum[idx - 1], 0)
    fp = np.where(idx > 0, fp_cum[idx - 1], 0)
    pred = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(Ptot > 0, tp / Ptot, 0.0)
        coverage = pred / n
        denom_f1 = precision + recall
        f1 = np.where(denom_f1 > 0, 2 * precision * recall / denom_f1, 0.0)

    return pd.DataFrame({
        "thr": thrs,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "F1": f1,
        "alerts": (coverage * n).astype(np.int64),
    })

def add_fbeta(table: pd.DataFrame, beta: float) -> pd.DataFrame:
    b2 = beta * beta
    pr, rc = table["precision"].to_numpy(), table["recall"].to_numpy()
    denom = (b2 * pr) + rc
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + b2) * pr * rc / denom, 0.0)
    t = table.copy()
    t["Fbeta"] = fbeta
    return t

def pick_best(table: pd.DataFrame, min_precision: float, max_coverage: float,
              metric: str) -> Dict[str, object]:
    """
    Pick best row under constraints; if none feasible, pick 'nearest' by relaxing precision.
    """
    feasible = table[(table["precision"] >= min_precision) & (table["coverage"] <= max_coverage)]
    if len(feasible):
        row = feasible.loc[feasible[metric].idxmax()]
        status = "feasible"
    else:
        under_cov = table[table["coverage"] <= max_coverage]
        if len(under_cov):
            row = under_cov.loc[under_cov[metric].idxmax()]
            status = "no-feasible (picked nearest)"
        else:
            k = max(int(round((1 - max_coverage) * (len(table) - 1))), 0)
            row = table.sort_values("thr").iloc[k]
            status = "no-feasible (quantile fallback)"

    return {
        "thr": float(row["thr"]),
        metric: float(row[metric]),
        "P": float(row["precision"]),
        "R": float(row["recall"]),
        "Cov": float(row["coverage"]),
        "Alerts": int(row["alerts"]),
        "status": status,
    }

def _fmt(res: dict, tag_label: str, score_key: str) -> None:
    thr   = res.get("thr", np.nan)
    scr   = res.get(score_key, 0.0)
    P     = res.get("P", 0.0)
    R     = res.get("R", 0.0)
    Cov   = res.get("Cov", 0.0)
    Alrt  = res.get("Alerts", None)
    stat  = res.get("status", "unknown")
    extra = f"  Alerts≈{Alrt:,}" if isinstance(Alrt, (int, np.integer)) else ""
    print(f"Lead +{res.get('lead_h','?')}h → [{stat}]  Best {tag_label} = {scr:.3f} @ thr={thr:.3f} "
          f"(P={P:.3f}, R={R:.3f})  Cov={Cov:.3f}{extra}")

# --------- main ---------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fast constrained F1 / Fβ threshold finder")
    ap.add_argument("--labelled", required=True, help="Labelled CSV(.gz) or Parquet")
    ap.add_argument("--model", required=True, help="Trained model (joblib)")
    ap.add_argument("--target", choices=["storm","near_storm","pregen"], default="pregen")
    ap.add_argument("--leads", nargs="+", type=int, required=True, help="Lead hours, e.g. 24 48 72")
    ap.add_argument("--min-precision", type=float, default=0.12)
    ap.add_argument("--max-coverage", type=float, default=0.25)
    ap.add_argument("--fbeta", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="CSV to save lead→thresholds")
    ap.add_argument("--save-table", default=None, help="Optional CSV with per-threshold metrics (last lead)")
    ap.add_argument("--verbose", action="store_true")
    # robustness flags:
    ap.add_argument("--time-format", default=None, help="Optional custom time format for parsing.")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none",
                    help="Normalize longitudes before processing.")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--prob-grid", type=int, default=1000, help="Threshold quantile grid size (default 1000).")
    return ap.parse_args()

def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("== Fast Fβ Threshold Finder (constrained) ==")
    print(f"Labelled : {args.labelled}")
    print(f"Model    : {args.model}")
    print(f"Target   : {args.target}")
    print(f"Leads    : {args.leads}")
    print(f"Constraints → min_precision={args.min_precision}  max_coverage={args.max_coverage}  β={args.fbeta}")
    print(f"Out      : {args.out}")

    # Load model to know feature set (so we can column-project on read)
    model, scaler, feats, imp_stats = load_model_any(args.model)

    # Decide which columns to read
    base_cols = ["time", "lat", "lon", args.target]
    columns = None
    if feats and len(feats) > 0:
        # Read exactly what's needed: meta + features
        columns = base_cols + list(dict.fromkeys(feats))  # de-dup while preserving order

    # Load & sanitize (CSV/GZ or Parquet)
    base = read_any(args.labelled, columns=columns)

    # Ensure required cols
    for c in ["time", "lat", "lon", args.target]:
        if c not in base.columns:
            raise ValueError(f"Labelled file must contain '{c}' column (missing after read).")

    # Basic sanitation
    base["time"] = _try_parse_time_raw(base["time"], args.time_format)
    base["lat"]  = pd.to_numeric(base["lat"], errors="coerce")
    base["lon"]  = _norm_lon(base["lon"], args.normalize_lon)
    base = base.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # Optional AOI crop (disabled unless explicitly set; default is global)
    a = _parse_area(args.area)
    if a:
        latN, lonW, latS, lonE = a
        base = base.loc[(base["lat"] <= latN) & (base["lat"] >= latS) &
                        (base["lon"] >= lonW) & (base["lon"] <= lonE)].reset_index(drop=True)

    if not base.empty:
        tmin = base["time"].min(); tmax = base["time"].max()
        print(f"[domain] time: {tmin} → {tmax}  | rows: {len(base):,}")
        print(f"[domain] lon:  {base['lon'].min():.3f} .. {base['lon'].max():.3f}  "
              f"| lat: {base['lat'].min():.3f} .. {base['lat'].max():.3f}")

    if args.target not in base.columns:
        raise ValueError(f"Target column '{args.target}' not found in labelled file.")

    # Features → probabilities
    X, feats_used = build_feature_matrix(base, model, scaler, feats, imp_stats)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1].astype(np.float64)
    else:
        from scipy.special import expit
        df_dec = model.decision_function(X).astype(np.float64)
        prob = expit(df_dec)

    y0 = pd.to_numeric(base[args.target], errors="coerce").fillna(0).astype(int).to_numpy()
    print(f"Rows: {len(base):,}  Pos(coincident {args.target}): {y0.sum():,}")

    results_rows = []
    last_table = None

    # We only need meta+target for label generation
    base_small = base[["time","lat","lon",args.target]].copy()

    for i, h in enumerate(args.leads, start=1):
        print(f"\nPreparing labels for lead +{h}h … ({i}/{len(args.leads)})")
        y = future_max_label_by_point(base_small, args.target, h)

        tbl = sweep_thresholds(prob, y, q_grid=args.prob_grid)
        tbl = add_fbeta(tbl, args.fbeta)
        if args.verbose:
            print(tbl.describe(percentiles=[0.1,0.5,0.9]).to_string())

        best_f1   = pick_best(tbl, args.min_precision, args.max_coverage, "F1")
        best_fbet = pick_best(tbl, args.min_precision, args.max_coverage, "Fbeta")
        best_f1["lead_h"] = h
        best_fbet["lead_h"] = h

        _fmt(best_f1, "F1", "F1")
        _fmt(best_fbet, f"Fβ={args.fbeta}", "Fbeta")

        results_rows.append({
            "lead_h": h,
            "thr_f1": best_f1["thr"],
            "F1": best_f1["F1"],
            "P_f1": best_f1["P"],
            "R_f1": best_f1["R"],
            "Cov_f1": best_f1["Cov"],
            "Alerts_f1": best_f1["Alerts"],
            "thr_Fbeta": best_fbet["thr"],
            "Fbeta": best_fbet["Fbeta"],
            "P_Fbeta": best_fbet["P"],
            "R_Fbeta": best_fbet["R"],
            "Cov_Fbeta": best_fbet["Cov"],
            "Alerts_Fbeta": best_fbet["Alerts"],
            "status_f1": best_f1["status"],
            "status_Fbeta": best_fbet["status"],
        })

        last_table = tbl

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved thresholds → {args.out}\n")
    print("== Summary (constrained) ==")
    with pd.option_context("display.max_columns", None):
        print(out_df.to_string(index=False))

    if args.save_table and last_table is not None:
        last_table.to_csv(args.save_table, index=False)
        print(f"\nSaved threshold curve (last lead) → {args.save_table}")

if __name__ == "__main__":
    main()