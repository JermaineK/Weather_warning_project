#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
find_best_f1_thresholds_constrained.py

Fast constrained per-lead threshold search directly from:
  - a labelled grid (CSV(.gz)/Parquet)
  - a trained model bundle (joblib)

It does NOT consume sweep summaries. Instead it:
  • loads the model bundle (global model and/or per-lead models)
  • scores probabilities on the labelled grid
  • builds time-aware labels per lead:
      - primary: success_col at t+lead (same grid)
      - fallback: future window of base target in (t, t+lead] per (lat,lon)
  • sweeps probability thresholds and maximises:
      - F1  (under constraints on precision / coverage)
      - Fβ (beta configurable)

Output:
  • CSV with one row per lead:
      lead_h, thr_f1, F1, P_f1, R_f1, Cov_f1, Alerts_f1,
      thr_Fbeta, Fbeta, P_Fbeta, R_Fbeta, Cov_Fbeta, Alerts_Fbeta,
      status_f1, status_Fbeta, label_mode

This CSV is compatible with grid_score.py when you set:
  thresholds_csv: results/best_fbeta_thresholds.csv
  thr_col: thr_Fbeta

Typical usage
-------------
python find_best_f1_thresholds_constrained.py \
  --labelled data/grid_labelled_FMA_gka_realthermo.parquet \
  --model models/grid_logit_perlead.pkl \
  --target storm \
  --success-col storm_window \
  --leads 24 48 72 120 \
  --min-precision 0.12 \
  --max-coverage 0.25 \
  --fbeta 0.5 \
  --out results/best_fbeta_thresholds.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib


# --------- lightweight I/O (CSV/GZ + Parquet) ---------

def read_any(path: str, columns: Optional[List[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read CSV(.gz) or Parquet. If `columns` is provided, tries to read only those columns
    (fast for Parquet; CSV uses usecols filter).
    """
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        try:
            df = pd.read_parquet(path, columns=columns if columns else None)
            return df.head(nrows) if nrows else df
        except Exception:
            df = pd.read_parquet(path)
            return df.head(nrows) if nrows else df
    if columns:
        cols_set = set(columns)
        return pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
            usecols=lambda c: c in cols_set,
            nrows=nrows,
        )
    return pd.read_csv(path, compression="infer", low_memory=False, nrows=nrows)


# --------- time/lon helpers ---------

def _try_parse_time_raw(s: pd.Series, fmt: Optional[str]) -> pd.Series:
    """
    Robust timestamp parser:
      - strip 'Z', try ISO
      - optional fixed format
      - epoch seconds/ms
      - fallback generic pd.to_datetime
    Always returns tz-naive UTC.
    """
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)

    # ISO-ish first
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5:
        return t1.dt.tz_localize(None)

    # Optional custom format
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5:
                return t2.dt.tz_localize(None)
        except Exception:
            pass

    # Epoch seconds / ms
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (isinstance(mid, (int, float)) and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5:
            return t3.dt.tz_localize(None)

    # Last resort
    t4 = pd.to_datetime(raw, utc=True, errors="coerce")
    return t4.dt.tz_localize(None)


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    # default: -180..180
    return ((x + 180) % 360) - 180


def _parse_area(aoi: Optional[str]):
    """
    Parse AOI string 'latN,lonW,latS,lonE' into floats.
    Example: '-5,125,-35,175'
    """
    if not aoi:
        return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")


# --------- model / feature utils ---------

def load_model_any(path: str):
    """
    Accept a dict bundle with keys:
      - model (global estimator) [optional if per_lead_models provided]
      - scaler
      - features
      - imputer_stats
      - clip_stats: {"lo": np.ndarray, "hi": np.ndarray}  (optional)
      - per_lead_models: {lead_h: estimator} (optional)

    Also supports plain estimators (limited).
    """
    m = joblib.load(path)
    if isinstance(m, dict):
        model = m.get("model") or m.get("model-out") or m.get("estimator")
        scaler = m.get("scaler", None)
        feats = list(m.get("features", m.get("feats", []) or []))
        imp = m.get("imputer_stats", None)
        clip = m.get("clip_stats", None)
        per_lead = m.get("per_lead_models", {}) or {}
        if model is None and not per_lead:
            raise ValueError("Bundle lacks a global 'model' and has no 'per_lead_models'.")
        if not feats:
            raise ValueError("Bundle lacks 'features'.")
        return model, scaler, feats, imp, clip, per_lead

    # plain estimator path (no per-lead support here)
    return (
        m,
        getattr(m, "scaler_", None),
        list(getattr(m, "features_", []) or []),
        None,
        None,
        {},
    )


def build_feature_matrix(
    df: pd.DataFrame,
    feats: List[str],
    imputer_stats: Optional[Dict[str, float]],
    clip_stats: Optional[Dict[str, np.ndarray]],
    scaler,
) -> np.ndarray:
    """
    Align columns to `feats`; impute with trainer stats (or zeros),
    optional clip, then scale.
    """
    Xdf = pd.DataFrame(index=df.index)
    for c in feats:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan

    X = Xdf.to_numpy(dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan

    # Impute
    if imputer_stats:
        for j, c in enumerate(feats):
            fill = float(imputer_stats.get(c, 0.0))
            m = ~np.isfinite(X[:, j])
            if m.any():
                X[m, j] = fill
    else:
        X = np.where(np.isfinite(X), X, 0.0)

    # Clip (if provided)
    if clip_stats and "lo" in clip_stats and "hi" in clip_stats:
        lo = np.asarray(clip_stats["lo"])
        hi = np.asarray(clip_stats["hi"])
        if lo.shape == X.shape[1:] and hi.shape == X.shape[1:]:
            X = np.clip(X, lo, hi)

    # Scale
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    return X


def future_max_label_by_point(
    df: pd.DataFrame,
    target_col: str,
    hours: int,
) -> np.ndarray:
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
        s = df.groupby(["lat", "lon"], sort=False).apply(_per_point)
    s.index = s.index.droplevel([0, 1])
    return s.to_numpy(dtype=int)


# --------- threshold sweeps ---------

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

    # Sort once for fast cumulative counts
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

    return pd.DataFrame(
        {
            "thr": thrs,
            "precision": precision,
            "recall": recall,
            "coverage": coverage,
            "F1": f1,
            "alerts": (coverage * n).astype(np.int64),
        }
    )


def add_fbeta(table: pd.DataFrame, beta: float) -> pd.DataFrame:
    b2 = beta * beta
    pr, rc = table["precision"].to_numpy(), table["recall"].to_numpy()
    denom = (b2 * pr) + rc
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = np.where(denom > 0, (1 + b2) * pr * rc / denom, 0.0)
    t = table.copy()
    t["Fbeta"] = fbeta
    return t


def pick_best(
    table: pd.DataFrame,
    min_precision: float,
    max_coverage: float,
    metric: str,
) -> Dict[str, object]:
    """
    Pick best row under constraints; if none feasible, pick 'nearest' by relaxing precision.
    """
    feasible = table[
        (table["precision"] >= min_precision) &
        (table["coverage"] <= max_coverage)
    ]
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
    thr = res.get("thr", np.nan)
    scr = res.get(score_key, 0.0)
    P = res.get("P", 0.0)
    R = res.get("R", 0.0)
    Cov = res.get("Cov", 0.0)
    Alrt = res.get("Alerts", None)
    stat = res.get("status", "unknown")
    extra = f"  Alerts≈{Alrt:,}" if isinstance(Alrt, (int, np.integer)) else ""
    print(
        f"Lead +{res.get('lead_h','?')}h -> [{stat}]  Best {tag_label} = {scr:.3f} @ thr={thr:.3f} "
        f"(P={P:.3f}, R={R:.3f})  Cov={Cov:.3f}{extra}"
    )


# --------- diagnostics helpers ---------

def _warn_if_flat(table: pd.DataFrame, y: np.ndarray, lead_h: int, tag: str = ""):
    uniq_y = int(np.unique(y).size)
    uniq_thr = int(np.unique(table["thr"]).size) if "thr" in table.columns else 0
    uniq_pts = int((table["precision"].round(6) + table["recall"].round(6)).nunique())
    if uniq_y <= 1:
        print(f"[warn] lead={lead_h}h{tag}: truth has a single class (uniq_y={uniq_y}).")
    if uniq_thr <= 5:
        print(f"[warn] lead={lead_h}h{tag}: threshold grid collapsed (uniq_thr={uniq_thr}).")
    if uniq_pts <= 3:
        print(f"[warn] lead={lead_h}h{tag}: PR curve has ≤3 distinct points (uniq_pts={uniq_pts}).")


# --------- main ---------

def _strip_choice(val: str) -> str:
    """Normalize choice strings to allow leading/trailing spaces in YAML/CLI."""
    return str(val).strip()


def _parse_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
        return [p.strip() for p in parts if p.strip()]
    out = []
    for tok in raw:
        out.extend(_parse_list(tok))
    # Remove dups, preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _infer_features_from_model(model) -> list[str]:
    """
    Try to recover feature names from sklearn estimators/pipelines.
    """
    candidates = []
    for attr in ("feature_names_in_",):
        val = getattr(model, attr, None)
        if val is not None:
            candidates = list(val)
            break
    if candidates:
        return candidates

    # Walk reversed pipeline steps if present
    if hasattr(model, "steps"):
        for _, step in reversed(getattr(model, "steps", [])):
            for attr in ("feature_names_in_",):
                val = getattr(step, attr, None)
                if val is not None:
                    return list(val)
    return []


def _load_metrics_features(path: Optional[str]) -> Tuple[list[str], Optional[str]]:
    """
    Load features/target from a metrics JSON if present.
    """
    if not path:
        return [], None
    p = Path(path)
    if not p.exists():
        return [], None
    try:
        import json

        meta = json.loads(p.read_text())
        feats = meta.get("features") or meta.get("overall", {}).get("features") or []
        tgt = meta.get("target") or meta.get("overall", {}).get("target")
        if feats is None:
            feats = []
        return list(feats), tgt
    except Exception:
        return [], None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fast constrained F1 / Fβ threshold finder (per-lead aware)"
    )
    ap.add_argument("--labelled", required=True, help="Labelled CSV(.gz) or Parquet")
    ap.add_argument("--model", required=True, help="Trained model bundle (joblib)")
    ap.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit feature list (space or comma separated). If omitted, uses bundle metadata.",
    )
    ap.add_argument(
        "--metrics-json",
        default=None,
        help="Optional metrics/metadata JSON containing features/target (used as fallback).",
    )
    ap.add_argument(
        "--target",
        default="y_viable",
        help="Base label used for training (default: y_viable).",
    )
    ap.add_argument(
        "--success-col",
        default="y_viable",
        help=(
            "Success metric column (evaluated at t+lead on the same grid). "
            "If missing, we fall back to future window of --target per lead."
        ),
    )
    ap.add_argument(
        "--lead-col",
        default=None,
        help=(
            "If set (e.g., t_to_storm_min_h), labels are built as (0 < lead_col <= lead_h) "
            "per lead, bypassing success/target windows."
        ),
    )
    ap.add_argument(
        "--leads",
        nargs="+",
        default=["24", "48", "72", "120", "240"],
        help="Lead hours, e.g. 24 48 72 120 240 (comma/space separated).",
    )
    ap.add_argument("--min-precision", type=float, default=0.12)
    ap.add_argument("--max-coverage", type=float, default=0.25)
    ap.add_argument("--fbeta", type=float, default=0.5)
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows for scoring; if set and table is larger, a random sample is used.",
    )
    ap.add_argument("--out", required=True, help="CSV to save lead->thresholds")
    ap.add_argument(
        "--save-table",
        default=None,
        help="Optional CSV with per-threshold metrics (last lead)",
    )
    ap.add_argument("--verbose", action="store_true")
    # robustness flags:
    ap.add_argument(
        "--time-format",
        default=None,
        help="Optional custom time format for parsing.",
    )
    ap.add_argument(
        "--normalize-lon",
        type=_strip_choice,
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Normalize longitudes before processing.",
    )
    # Chunking hints (accepted for pipeline compatibility; currently processed as best-effort)
    ap.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Optional chunk size hint; accepted for compatibility (full table is loaded).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Alias for --chunk-rows (accepted for compatibility).",
    )
    ap.add_argument(
        "--parquet-rows",
        type=int,
        default=None,
        help="Optional row-group hint for parquet readers (accepted for compatibility).",
    )
    ap.add_argument(
        "--area",
        default=None,
        help='Optional crop "latN,lonW,latS,lonE" after lon normalization.',
    )
    ap.add_argument(
        "--prob-grid",
        type=int,
        default=1000,
        help="Threshold quantile grid size (default 1000).",
    )
    return ap.parse_args()


def _parse_leads(raw) -> list[int]:
    leads: list[int] = []
    for tok in raw:
        for part in str(tok).replace(",", " ").split():
            part = part.strip()
            if part:
                leads.append(int(float(part)))
    return leads


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    leads = _parse_leads(args.leads)

    # Try to auto-locate metrics JSON if not provided
    auto_metrics = None
    if not args.metrics_json:
        mpath = Path(args.model)
        cand1 = mpath.with_name(mpath.stem + "_metrics.json")
        cand2 = mpath.with_suffix(".meta.json")
        for cand in (cand1, cand2):
            if cand.exists():
                auto_metrics = str(cand)
                break
        if auto_metrics:
            args.metrics_json = auto_metrics
            print(f"[info] using metrics file {auto_metrics} for feature fallback")

    # Compatibility logging for chunk hints
    if args.chunk_rows or args.chunksize or args.parquet_rows:
        hint = args.chunk_rows or args.chunksize or args.parquet_rows
        print(f"[info] chunking hints received (chunk_rows={hint}); processing full table in-memory for now.")

    print("== Fast Fβ Threshold Finder (constrained, per-lead aware) ==")
    print(f"Labelled : {args.labelled}")
    print(f"Model    : {args.model}")
    print(f"Target   : {args.target}  | Success={args.success_col} (lead-aware)")
    if args.lead_col:
        print(f"Lead-col : {args.lead_col} (mask 0<lead<=L per lead)")
    print(f"Leads    : {leads}")
    print(
        f"Constraints -> min_precision={args.min_precision}  "
        f"max_coverage={args.max_coverage}  β={args.fbeta}"
    )
    print(f"Out      : {args.out}")

    # Load bundle to know features and (optionally) per-lead models
    model_global, scaler, feats, imp_stats, clip_stats, per_lead = load_model_any(args.model)
    cli_feats = _parse_list(args.features)
    if cli_feats:
        feats = cli_feats
    if not feats:
        feats = _infer_features_from_model(model_global)
    if not feats and args.metrics_json:
        meta_feats, meta_target = _load_metrics_features(args.metrics_json)
        if meta_feats:
            feats = meta_feats
            if meta_target and not cli_feats and args.target == "y_viable":
                # only override if user did not set a different target explicitly
                args.target = meta_target
            print(f"[info] loaded {len(feats)} features from metrics JSON {args.metrics_json}")
    if not feats:
        # Last-resort: infer numerics from a small sample of the labelled table
        sample = read_any(args.labelled, nrows=2000)
        if sample is not None:
            drop = {
                "time",
                "lat",
                "lon",
                args.target,
                args.success_col,
                args.lead_col,
            }
            feats = [
                c
                for c in sample.columns
                if c not in drop and pd.api.types.is_numeric_dtype(sample[c])
            ]
            if feats:
                print(f"[info] inferred {len(feats)} numeric feature columns from sample because bundle lacked features.")
    if not feats:
        raise ValueError(
            "No feature list found in bundle. Provide --features or ensure feature_names_in_ is saved with the model."
        )

    # Decide columns to read
    base_cols = ["time", "lat", "lon", args.target, args.success_col]
    if args.lead_col:
        base_cols.append(args.lead_col)
    columns = list(dict.fromkeys(base_cols + list(feats)))

    # Load & sanitize
    base = read_any(args.labelled, columns=columns)
    if args.max_rows and len(base) > args.max_rows:
        base = base.sample(n=int(args.max_rows), random_state=42)
        print(f"[info] sampled down to {len(base):,} rows (max_rows={args.max_rows})")

    # Ensure core cols exist (target may be missing only if success exists—handled below)
    need = {"time", "lat", "lon"}
    missing = need - set(base.columns)
    if missing:
        raise ValueError(f"Labelled file missing columns: {sorted(missing)}")

    base["time"] = _try_parse_time_raw(base["time"], args.time_format)
    base["lat"] = pd.to_numeric(base["lat"], errors="coerce")
    base["lon"] = _norm_lon(base["lon"], args.normalize_lon)
    base = base.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    # Optional AOI crop
    a = _parse_area(args.area)
    if a:
        latN, lonW, latS, lonE = a
        base = base.loc[
            (base["lat"] <= latN)
            & (base["lat"] >= latS)
            & (base["lon"] >= lonW)
            & (base["lon"] <= lonE)
        ].reset_index(drop=True)

    if not base.empty:
        tmin = base["time"].min()
        tmax = base["time"].max()
        print(
            f"[domain] time: {tmin} -> {tmax}  | rows: {len(base):,}"
        )
        print(
            f"[domain] lon:  {base['lon'].min():.3f} .. {base['lon'].max():.3f}  "
            f"| lat: {base['lat'].min():.3f} .. {base['lat'].max():.3f}"
        )

    success_available = args.success_col in base.columns
    target_available = args.target in base.columns

    if not success_available and not target_available and not args.lead_col:
        raise ValueError(
            f"Neither success_col '{args.success_col}' nor target '{args.target}' are present."
        )

    # Build features once (aligned to trainer)
    X = build_feature_matrix(
        base,
        feats,
        imputer_stats=imp_stats,
        clip_stats=clip_stats,
        scaler=scaler,
    )

    # For diagnostics only
    if target_available:
        y0 = pd.to_numeric(base[args.target], errors="coerce").fillna(0).astype(int).to_numpy()
        print(f"Rows: {len(base):,}  Pos(coincident {args.target} at t): {y0.sum():,}")
    else:
        print(
            f"Rows: {len(base):,}  (no '{args.target}' present; using success-only for selection)"
        )

    results_rows = []
    last_table = None

    # Pre-build success lookup for lead-aware merging (same grid at t+lead)
    base_small = None
    if success_available:
        succ = base[["time", "lat", "lon", args.success_col]].copy()
        succ[args.success_col] = (
            pd.to_numeric(succ[args.success_col], errors="coerce")
            .fillna(0)
            .astype(np.int8)
        )
        succ = succ.dropna(subset=["time"]).reset_index(drop=True)
        succ = succ.sort_values("time", kind="mergesort")
    else:
        # we will compute per-lead future windows of target on the fly
        if not args.lead_col:
            base_small = base[["time", "lat", "lon", args.target]].copy()
            base_small[args.target] = (
                pd.to_numeric(base_small[args.target], errors="coerce")
                .fillna(0)
                .astype(np.int8)
            )

    # Also keep features for joins
    base_key = base[["time", "lat", "lon"]].copy()

    # For anti-flattening checks
    last_pos = None

    lead_vals = None
    if args.lead_col:
        if args.lead_col in base.columns:
            lead_vals = pd.to_numeric(base[args.lead_col], errors="coerce").to_numpy()
        else:
            print(f"[warn] lead_col={args.lead_col} not found; falling back to success/target logic")

    # helper for chunked scoring to avoid long silent runs
    def score_probs(estimator, X_mat, batch_rows: int = 1_000_000):
        n = len(X_mat)
        if n == 0:
            return np.array([], dtype=np.float64)
        # choose batch: respect hint if provided
        if args.chunk_rows or args.chunksize or args.parquet_rows:
            hint = args.chunk_rows or args.chunksize or args.parquet_rows
            if hint and hint > 0:
                batch_rows = max(100_000, int(hint))
        prob_out = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_rows):
            end = min(n, start + batch_rows)
            if hasattr(estimator, "predict_proba"):
                prob_out[start:end] = estimator.predict_proba(X_mat[start:end])[:, 1].astype(np.float64)
            else:
                from scipy.special import expit
                dec = estimator.decision_function(X_mat[start:end]).astype(np.float64)
                prob_out[start:end] = expit(dec)
            if n > batch_rows:
                pct = 100.0 * end / n
                print(f"    scored {end:,}/{n:,} rows ({pct:.1f}%)", flush=True)
        return prob_out

    for i, h in enumerate(leads, start=1):
        print(
            f"\nLead +{h}h - scoring probabilities and sweeping thresholds . "
            f"({i}/{len(leads)})"
        )

        # Pick estimator for this lead (fallback to global)
        est = per_lead.get(h, model_global)
        if h in per_lead:
            print("  • using per-lead estimator", flush=True)
        else:
            print("  • no per-lead estimator; using global", flush=True)

        # Score probs for this lead
        prob = score_probs(est, X)

        # Build success labels for this horizon
        if lead_vals is not None:
            y = ((lead_vals > 0) & (lead_vals <= float(h))).astype(np.int8)
            label_mode = f"{args.lead_col} in (0,{h}]"
        elif success_available:
            # Align by shifting the *features clock* forward by +h and joining success at that time
            shifted = base_key.copy()
            shifted["time"] = shifted["time"] + pd.to_timedelta(int(h), unit="h")
            y_df = shifted.merge(
                succ, on=["time", "lat", "lon"], how="left", validate="one_to_one"
            )
            y = (
                pd.to_numeric(y_df[args.success_col], errors="coerce")
                .fillna(0)
                .astype(np.int8)
                .to_numpy()
            )
            label_mode = f"success={args.success_col} @ t+{h}h"
        else:
            # Fallback: build future-of-target window per lead (old behavior)
            if base_small is None:
                raise ValueError(
                    f"Lead-aware labels require either lead_col '{args.lead_col}', "
                    f"success_col '{args.success_col}', or target '{args.target}'."
                )
            y = future_max_label_by_point(base_small.copy(), args.target, h)
            y = y.astype(np.int8)
            label_mode = f"future_of_{args.target} (window={h}h)"

        pos = int(y.sum())
        print(
            f"  • label mode: {label_mode} | positives={pos:,}  frac={pos/len(y):.4f}"
        )
        if last_pos is not None:
            d = pos - last_pos
            print(f"  • Δpositives vs prev lead: {d:+,}")
        last_pos = pos

        # Early degeneracy flags (warn, but still compute)
        if pos == 0 or pos == len(y):
            print(
                f"[warn] lead={h}h: label degeneracy (all zeros or all ones). "
                f"Threshold search will be uninformative."
            )

        # Sweep thresholds under constraints
        tbl = sweep_thresholds(prob, y, q_grid=args.prob_grid)
        tbl = add_fbeta(tbl, args.fbeta)

        # Anti-flattening diagnostics
        _warn_if_flat(tbl, y, h, tag="")

        if args.verbose:
            print(
                tbl.describe(percentiles=[0.1, 0.5, 0.9]).to_string()
            )

        best_f1 = pick_best(tbl, args.min_precision, args.max_coverage, "F1")
        best_fbet = pick_best(tbl, args.min_precision, args.max_coverage, "Fbeta")
        best_f1["lead_h"] = h
        best_fbet["lead_h"] = h

        _fmt(best_f1, "F1", "F1")
        _fmt(best_fbet, f"Fβ={args.fbeta}", "Fbeta")

        results_rows.append(
            {
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
                "label_mode": label_mode,
            }
        )

        last_table = tbl

    out_df = pd.DataFrame(results_rows).sort_values("lead_h")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".pq", ".pqt"}:
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    print(f"\nSaved thresholds -> {args.out}\n")
    print("== Summary (constrained) ==")
    with pd.option_context("display.max_columns", None):
        print(out_df.to_string(index=False))

    if args.save_table and last_table is not None:
        save_path = Path(args.save_table)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() in {".parquet", ".pq", ".pqt"}:
            last_table.to_parquet(save_path, index=False)
        else:
            last_table.to_csv(save_path, index=False)
        print(f"\nSaved threshold curve (last lead) -> {args.save_table}")


if __name__ == "__main__":
    main()
