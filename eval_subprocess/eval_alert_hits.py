#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_alert_hits.py
Compare grid alerts to truth with lead-aware timing on a fixed grid.

Defaults:
  - Truth column: storm_window  (success metric)
  - Alert flag:   alert_final
  - Lead hours:   required; we evaluate hits at T+lead_h against truth at T+lead_h.

Inputs:
  --labelled : labelled grid (must include time, lat, lon, truth col)
  --alerts   : alerts CSV/Parquet (must include time, lat, lon, flag col); location-preserving
  --lead-hours : integer lead (hours)
  --flag-col   : alerts flag column (default alert_final)
  --truth-col  : truth column (default storm_window)
  --time-col   : optional (default time)
  --normalize-lon : none | -180..180 | 0..360  (must match pipeline)
  --grid-decimals : optional int; if set, round lat/lon to this many decimals
                    in BOTH labelled and alerts before merging.

Outputs:
  - Prints precision/recall/F1/coverage and hourly diagnostics.
  - Writes a tiny CSV next to alerts with per-lead metrics.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_any(p, usecols=None):
    p_str = str(p)
    low = p_str.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return pd.read_parquet(p_str, columns=usecols if usecols is not None else None)
    return pd.read_csv(p_str, compression="infer", low_memory=False,
                       usecols=usecols if usecols is not None else None)


def to_utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def norm_lon(x, mode: str) -> pd.Series:
    xx = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return xx
    if mode == "0..360":
        return (xx % 360 + 360) % 360
    # default -180..180
    return ((xx + 180) % 360) - 180


def binarize(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int8)


def prf(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def by_hour_sum(t: pd.Series, v: np.ndarray) -> pd.Series:
    idx = to_utc_naive(t).dt.floor("H")
    return pd.Series(v, index=idx).groupby(level=0).sum()


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate grid alerts against truth (lead-aware).")
    ap.add_argument("--labelled",
                    default="data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet",
                    help="Labelled grid with time, lat, lon, truth column.")
    ap.add_argument("--alerts", required=True,
                    help="Alerts CSV/Parquet with time, lat, lon, flag column.")
    ap.add_argument("--lead-hours", type=int, required=True,
                    help="Lead time in hours (alerts at t evaluated vs truth at t+lead).")
    ap.add_argument("--flag-col", default="alert_final",
                    help="Alert flag column in alerts (default: alert_final).")
    ap.add_argument("--truth-col", default="storm_window",
                    help="Truth/target column in labelled grid (default: storm_window).")
    ap.add_argument("--truth-mode", choices=["storm_window", "t_to_storm_leq"], default="storm_window",
                    help="Truth definition: storm_window flag or t_to_storm_min_h <= lead-hours.")
    ap.add_argument("--lead-col", default="t_to_storm_min_h",
                    help="Lead column when using truth-mode=t_to_storm_leq.")
    ap.add_argument("--time-col", default="time",
                    help="Time column name shared by labelled/alerts (default: time).")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180",
                    help="Longitude frame (must match rest of pipeline).")
    ap.add_argument("--grid-decimals", type=int, default=None,
                    help="If set, round lat/lon to this many decimals before merging.")
    ap.add_argument("--out-csv", default=None,
                    help="Optional metrics CSV path (default: beside alerts, *_eval_lead{L}.csv).")
    ap.add_argument("--run-name", default=None,
                    help="Optional run name to stamp default output (results/metrics/<run>_alert_hits_lead{L}.csv).")
    return ap.parse_args()


def maybe_round_grid(df: pd.DataFrame, decimals: int | None, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    if decimals is None:
        return df
    out = df.copy()
    out[lat_col] = out[lat_col].round(decimals)
    out[lon_col] = out[lon_col].round(decimals)
    return out


def main():
    a = parse_args()
    lead_h = int(a.lead_hours)

    # ---- labelled (truth) ----
    lab = read_any(a.labelled)
    need_lab = {a.time_col, "lat", "lon", a.truth_col, a.lead_col}
    if not need_lab.issubset(lab.columns):
        raise ValueError(
            f"Labelled missing {sorted(need_lab)}; got {sorted(lab.columns)[:20]} ..."
        )

    lab = lab[[a.time_col, "lat", "lon", a.truth_col]].copy()
    lab[a.time_col] = to_utc_naive(lab[a.time_col])
    lab["lat"] = pd.to_numeric(lab["lat"], errors="coerce")
    lab["lon"] = norm_lon(lab["lon"], a.normalize_lon)
    lab[a.truth_col] = binarize(lab[a.truth_col])
    if a.truth_mode == "t_to_storm_leq":
        lead_vals = pd.to_numeric(lab[a.lead_col], errors="coerce")
        lab[a.truth_col] = ((lead_vals > 0) & (lead_vals <= lead_h)).astype(np.int8)
    lab = lab.dropna(subset=[a.time_col, "lat", "lon"]).reset_index(drop=True)
    lab = maybe_round_grid(lab, a.grid_decimals)

    # ---- alerts ----
    al = read_any(a.alerts)
    need_al = {a.time_col, "lat", "lon", a.flag_col}
    if not need_al.issubset(al.columns):
        raise ValueError(
            f"Alerts missing {sorted(need_al)}; got {sorted(al.columns)[:20]} ..."
        )

    al = al[[a.time_col, "lat", "lon", a.flag_col]].copy()
    al[a.time_col] = to_utc_naive(al[a.time_col])
    al["lat"] = pd.to_numeric(al["lat"], errors="coerce")
    al["lon"] = norm_lon(al["lon"], a.normalize_lon)
    al[a.flag_col] = binarize(al[a.flag_col])
    al = al.dropna(subset=[a.time_col, "lat", "lon"]).reset_index(drop=True)
    al = maybe_round_grid(al, a.grid_decimals)

    # ---- lead shift: alerts at t evaluate vs truth at t+lead ----
    al_eval = al.copy()
    al_eval[a.time_col] = al_eval[a.time_col] + pd.to_timedelta(lead_h, unit="h")

    # ---- exact (time,lat,lon) join ----
    key = [a.time_col, "lat", "lon"]
    df = al_eval.merge(
        lab,
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("_al", "_lab"),
    )

    default_out = (
        f"results/metrics/{a.run_name}_alert_hits_lead{lead_h}.csv"
        if a.run_name else None
    )

    if df.empty:
        print(
            f"[eval] No overlaps after lead shift (lead={lead_h}h). "
            f"Check lon frame, grid rounding, and that alerts preserve (time,lat,lon)."
        )
        outp = a.out_csv or default_out or (str(Path(a.alerts).with_suffix("")) + f"_eval_lead{lead_h}.csv")
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{
                "lead_h": lead_h,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "coverage": 0.0,
                "rows_joined": 0,
            }]
        ).to_csv(outp, index=False)
        print(f"[eval] wrote {outp}")
        return

    yhat = df[a.flag_col].to_numpy().astype(np.int8)
    ytru = df[a.truth_col].to_numpy().astype(np.int8)

    tp = int(((yhat == 1) & (ytru == 1)).sum())
    fp = int(((yhat == 1) & (ytru == 0)).sum())
    fn = int(((yhat == 0) & (ytru == 1)).sum())
    prec, rec, f1 = prf(tp, fp, fn)
    cov = float(yhat.mean())

    # ---- hourly diagnostics ----
    byh_alerts = by_hour_sum(df[a.time_col], yhat)
    byh_truth = by_hour_sum(df[a.time_col], ytru)
    same_cov = (cov == 0.0) or (cov == 1.0)

    print(
        f"[eval] lead={lead_h}h | n={len(df):,} | tp={tp:,} fp={fp:,} fn={fn:,} "
        f"| P={prec:.3f} R={rec:.3f} F1={f1:.3f} | cov={cov:.3f}",
        flush=True,
    )
    if same_cov:
        print(
            "[warn] alert coverage degenerate (0% or 100%). "
            "Check thresholding/throttle.",
            flush=True,
        )

    if len(byh_alerts):
        alerts_min = int(byh_alerts.min())
        alerts_max = int(byh_alerts.max())
    else:
        alerts_min = alerts_max = 0

    if len(byh_truth):
        truth_min = int(byh_truth.min())
        truth_max = int(byh_truth.max())
    else:
        truth_min = truth_max = 0

    print(
        f"[diag] hours={len(byh_alerts):,}  "
        f"alerts/h mean={byh_alerts.mean() if len(byh_alerts) else 0:.2f}  "
        f"truth/h mean={byh_truth.mean() if len(byh_truth) else 0:.2f} "
        f"min/max alerts/h=({alerts_min}, {alerts_max}) "
        f"min/max truth/h=({truth_min}, {truth_max})",
        flush=True,
    )

    # ---- write metrics CSV ----
    outp = a.out_csv or default_out or (str(Path(a.alerts).with_suffix("")) + f"_eval_lead{lead_h}.csv")
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "lead_h": lead_h,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "coverage": cov,
            "rows_joined": int(len(df)),
        }]
    ).to_csv(outp, index=False)
    print(f"[eval] wrote {outp}")


if __name__ == "__main__":
    main()
