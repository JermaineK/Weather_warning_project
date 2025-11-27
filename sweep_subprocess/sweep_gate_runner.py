#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_gate_runner.py — gate (build×relax) threshold sweep with time-aware windows

Gate logic per row:
    gate   = (past_max(p_build, build_window_h) >= tb)
    alerts = gate & (p_relax >= tr)

Optionally apply hourly throttling (keep only the top (1-q) fraction by score
within each hour).

Outputs a CSV grid with:
    lead, tb, tr,
    F1, Precision, Recall, Coverage,
    AUC, PRAUC, Brier

Where:
  - score = p_relax masked by the gate (0 outside gate).
  - Future labels use a strict future window (t, t+lead].
  - Past window uses time-based rolling per (lat,lon) and excludes the
    current hour via shift(1).

Model bundles
-------------
Expected model format for --build / --relax:

  • Recommended: dict bundle saved with joblib:
        {"model": <sk_model>, "scaler": <sk_scaler or None>, "features": [..]}

  • Also accepted: raw sklearn model with feature_names_in_ attribute.

Features are median-imputed per column, and ±Inf are coerced to NaN before scaling.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ---------------- parsing ----------------


def parse_float_list(s: str) -> List[float]:
    """
    Parse either:
      "a:b:c" → [a, a+c, ..., ≤ b]
      "x,y,z" → [x, y, z]
    """
    s = str(s).strip()
    if ":" in s:
        a, b, c = [float(x) for x in s.split(":")]
        if c <= 0:
            raise ValueError(f"step in range must be > 0, got {c}")
        out: List[float] = []
        x = a
        while x <= b + 1e-12:
            out.append(round(x, 12))
            x += c
        return out
    return [float(x) for x in s.split(",") if x.strip()]


# ---------------- I/O helpers ----------------


def read_any(path: Path | str, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Read CSV(.gz)/Parquet with optional parse_dates."""
    path = Path(path)
    p = path.suffix.lower()
    if p in {".parquet", ".pq", ".pqt"}:
        return pd.read_parquet(path)
    return pd.read_csv(
        path,
        low_memory=False,
        compression="infer",
        parse_dates=list(parse_dates) if parse_dates else None,
    )


def write_any(path: Path | str, df: pd.DataFrame) -> None:
    """Write DataFrame to CSV(.gz) based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = path.suffix.lower()
    if p in {".parquet", ".pq", ".pqt"}:
        df.to_parquet(path, index=False)
        return
    comp = "gzip" if str(path).lower().endswith(".gz") else "infer"
    df.to_csv(path, index=False, compression=comp)


# ---------------- model scoring ----------------


def _load_model_bundle(model_path: Path | str):
    """
    Load a model bundle that is either:
      • dict: {"model", "scaler"(opt), "features"}
      • raw sklearn estimator with feature_names_in_
    """
    m = joblib.load(model_path)

    if isinstance(m, dict):
        model = m["model"]
        scaler = m.get("scaler", None)
        feats = list(m["features"])
        return model, scaler, feats

    feats = getattr(m, "feature_names_in_", None)
    if feats is None:
        raise ValueError(
            "Model does not expose 'feature_names_in_' and no 'features' list "
            "was found in the bundle. Please save your bundle as "
            "{'model','scaler','features'}."
        )
    return m, None, list(feats)


def load_model_probs(df: pd.DataFrame, model_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load model from joblib path, median-impute numeric NaNs, and return:
      p   : predicted probabilities (shape [n_rows])
      feats: ordered feature list used
    """
    model, scaler, feats = _load_model_bundle(model_path)
    Xdf = df.loc[:, feats].astype(float).replace([np.inf, -np.inf], np.nan)

    rows_with_nan = int(Xdf.isna().any(axis=1).sum())
    if rows_with_nan:
        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med)

    Xv = Xdf.to_numpy()
    if scaler is not None:
        Xv = scaler.transform(Xv)

    p = model.predict_proba(Xv)[:, 1]
    if rows_with_nan:
        print(
            f"  • {Path(model_path).name}: imputed NaNs on {rows_with_nan:,} rows",
            flush=True,
        )
    return p, feats


# ---------------- time-aware windows ----------------


def time_rolling_max_per_point(
    df: pd.DataFrame, col: str, hours: int
) -> pd.Series:
    """
    Per (lat,lon), time-based rolling max of `col` over the past `hours`,
    EXCLUDING the current instant (via shift(1)).

    Returns a Series aligned to df.index.
    """
    win = f"{int(hours)}H"
    tmp = df.loc[:, ["time", "lat", "lon", col]].copy()

    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        s = (
            g.set_index("time")[col]
            .rolling(win, min_periods=1)
            .max()
            .shift(1)
            .fillna(0.0)
        )
        s = s.reindex(g["time"])
        s.index = g.index
        return s

    out = tmp.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_one)
    return out.reindex(df.index)


def future_rolling_max_per_point(
    df: pd.DataFrame, target_col: str, hours: int
) -> pd.Series:
    """
    Per (lat,lon), strict-future (t, t+hours] rolling max of binary `target_col`.

    Returns a Series aligned to df.index (values are 0/1 int).
    """
    win = f"{int(hours)}H"
    tmp = df.loc[:, ["time", "lat", "lon", target_col]].copy()

    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        rev = g.set_index("time")[target_col].iloc[::-1]
        fut = rev.rolling(win, min_periods=1).max().shift(1)  # exclude current
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        fut.index = g.index
        return fut

    out = tmp.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_one)
    return out.reindex(df.index).astype(int)


# ---------------- post-filters ----------------


def throttle_hourly(
    score_vec: np.ndarray, times: pd.Series, q: float | None
) -> np.ndarray:
    """
    Keep only the top (1-q) fraction by score within each hour.
    If q=None, keep all.

    Example: q = 0.90 → keep top 10% by score in each hour.
    """
    if q is None:
        return np.ones(len(score_vec), dtype=bool)

    s = pd.Series(score_vec, index=times.index)

    def _keep(group: pd.Series) -> pd.Series:
        if len(group) == 0:
            return pd.Series([], dtype=bool)
        k = int(np.ceil(len(group) * (1 - q)))
        if k <= 0:
            return pd.Series([False] * len(group), index=group.index)
        thr = group.nlargest(k).min()
        return group >= thr

    mask = s.groupby(times.dt.floor("H"), sort=False).apply(_keep)
    return (
        mask.reset_index(level=0, drop=True)
        .reindex(s.index)
        .fillna(False)
        .to_numpy()
    )


# ---------------- main ----------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep build×relax gate thresholds with time-aware windows."
    )
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument(
        "--target",
        required=True,
        choices=["storm", "near_storm", "pregen", "storm_hit"],
    )
    ap.add_argument(
        "--build-window",
        type=int,
        default=24,
        help="Past hours for build max (exclude current).",
    )
    ap.add_argument(
        "--tb",
        help="Build thresholds (e.g. 0.03:0.10:0.005 or 0.05,0.07,0.10).",
    )
    ap.add_argument(
        "--tr",
        help="Relax thresholds (e.g. 0.03:0.10:0.005 or 0.05,0.07,0.10).",
    )
    ap.add_argument(
        "--leads",
        required=True,
        help="Comma list of lead hours (e.g. 24,48,72).",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="Hourly throttle quantile (e.g. 0.90 keeps top 10%).",
    )
    ap.add_argument(
        "--subsample-hours",
        type=float,
        default=0.0,
        help="Fraction of hours to keep for a quick sweep (0 disables).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV(.gz)/Parquet with metric grid.",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Print running progress and ETA.",
    )
    args = ap.parse_args()

    tb_list = parse_float_list(args.tb) if args.tb else [0.05, 0.07, 0.10]
    tr_list = parse_float_list(args.tr) if args.tr else [0.05, 0.07, 0.10]
    leads = [int(x) for x in str(args.leads).split(",") if str(x).strip()]

    print("== Gate Sweep (build×relax) ==")
    print(f"File         : {args.labelled}")
    print(f"Build model  : {args.build}")
    print(f"Relax model  : {args.relax}")
    print(f"Target       : {args.target}")
    print(f"Build window : {args.build_window} h")
    print(f"tb           : {tb_list}")
    print(f"tr           : {tr_list}")
    print(f"Leads        : {leads}")
    if args.quantile is not None:
        print(f"Hourly throttle quantile: {args.quantile}")
    if args.subsample_hours and args.subsample_hours > 0:
        print(f"Subsample hours fraction: {args.subsample_hours}")

    # Read labelled grid (CSV or Parquet)
    df = read_any(args.labelled, parse_dates=["time"])

    # Normalize time / coords
    df["time"] = (
        pd.to_datetime(df["time"], utc=True, errors="coerce")
        .dt.tz_convert(None)
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    # Optional subsample by hour for quicker sweeps
    if args.subsample_hours and args.subsample_hours > 0:
        hrs = df["time"].dt.floor("H").drop_duplicates().sort_values()
        keep_hrs = hrs.sample(frac=args.subsample_hours, random_state=42)
        df = (
            df[df["time"].dt.floor("H").isin(keep_hrs)]
            .sort_values(["time", "lat", "lon"])
            .reset_index(drop=True)
        )
        print(f"Subsampled hours → rows: {len(df):,}")
    else:
        df = df.sort_values(["time", "lat", "lon"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows after cleaning time/lat/lon.")

    # Score models
    print("Scoring build model…", flush=True)
    p_build, feats_build = load_model_probs(df, args.build)
    print("Scoring relax model…", flush=True)
    p_relax, feats_relax = load_model_probs(df, args.relax)

    df["p_build"] = p_build
    df["p_relax"] = p_relax

    print(f"Build features: {len(feats_build)} columns")
    print(f"Relax features: {len(feats_relax)} columns")

    # Compute time-aware past max for build
    print("Computing build_recent window…", flush=True)
    t0 = time.time()
    df["build_recent"] = time_rolling_max_per_point(
        df, "p_build", hours=args.build_window
    )
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    rows: List[dict] = []
    combos = list(itertools.product(leads, tb_list, tr_list))
    total = len(combos)
    start = time.time()
    last = start

    for i, (lead_h, tb, tr) in enumerate(combos, start=1):
        # Strict future window labels
        y = future_rolling_max_per_point(
            df, args.target, hours=lead_h
        ).to_numpy()

        gate = df["build_recent"].to_numpy() >= tb
        p_relax_vec = df["p_relax"].to_numpy()
        score = np.where(gate, p_relax_vec, 0.0)
        alerts = gate & (p_relax_vec >= tr)

        if args.quantile is not None:
            keep_mask = throttle_hourly(score, df["time"], q=args.quantile)
            alerts = alerts & keep_mask

        cov = alerts.mean() if len(alerts) else 0.0

        # Confusion metrics on final alerts
        if alerts.any():
            pr, rc, f1, _ = precision_recall_fscore_support(
                y,
                alerts.astype(int),
                average="binary",
                zero_division=0,
            )
        else:
            pr = rc = f1 = 0.0

        # Score-based metrics
        try:
            auc = roc_auc_score(y, score)
        except ValueError:
            auc = np.nan
        try:
            prauc = average_precision_score(y, score)
        except ValueError:
            prauc = np.nan
        try:
            brier = brier_score_loss(y, score)
        except ValueError:
            brier = np.nan

        rows.append(
            {
                "lead": lead_h,
                "tb": tb,
                "tr": tr,
                "F1": float(f1),
                "Precision": float(pr),
                "Recall": float(rc),
                "Coverage": float(cov),
                "AUC": float(auc) if not np.isnan(auc) else np.nan,
                "PRAUC": float(prauc) if not np.isnan(prauc) else np.nan,
                "Brier": float(brier) if not np.isnan(brier) else np.nan,
            }
        )

        if args.progress and (time.time() - last >= 0.5 or i == total):
            pct = 100.0 * i / total
            elapsed = time.time() - start
            eta = elapsed * (total / i - 1) if i > 0 else float("nan")
            print(
                f"Progress: {pct:5.1f}%  ({i}/{total})  ETA {eta:6.1f}s",
                end="\r",
                flush=True,
            )
            last = time.time()

    if args.progress:
        print()

    out = pd.DataFrame(rows).sort_values(
        ["lead", "F1", "Recall", "Precision"],
        ascending=[True, False, False, False],
    )

    write_any(args.out, out)
    print(f"Wrote {args.out}  rows={len(out)}")

    # Per-lead “best line” for quick eyeballing
    for L in leads:
        best = out[out["lead"] == L].head(1)
        if len(best):
            r = best.iloc[0]
            print(
                f"[lead {L:>3}h]  F1={r.F1:.3f}  P={r.Precision:.3f}  R={r.Recall:.3f}  "
                f"Cov={r.Coverage:.3f}  tb={r.tb:.3f}  tr={r.tr:.3f}  "
                f"AUC={r.AUC:.3f}  PRAUC={r.PRAUC:.3f}  Brier={r.Brier:.3f}"
            )


if __name__ == "__main__":
    main()