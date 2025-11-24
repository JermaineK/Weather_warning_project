#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_runner.py

Grid alert sweep over thresholds / persistence / neighborhood / hourly-quantile.

- Reads labelled grid (CSV(.gz)/Parquet) and a sklearn joblib bundle
  containing {"model", "scaler"(opt), "features"} or a raw sklearn model.
- Builds time-aware future labels per (lat,lon) for each lead L.
- Applies:
    1) score threshold
    2) persistence ≥ P consecutive hours per cell
    3) neighborhood filter (≥ N neighbors in 8-neighborhood at same hour)
    4) hourly throttle: keep top q quantile by score within each hour
- Computes AUC/PRAUC/Brier on the full population for reference, and
  F1/Precision/Recall/Coverage on the final kept mask.

Usage (example)
--------------
python sweep_runner.py \
  --labelled data/grid_labelled_FMA_gka.csv.gz \
  --model models/grid_logit_cal.pkl \
  --target storm \
  --leads 24,48,72 \
  --thr-grid 0.04:0.10:0.002 \
  --persist 1,2 \
  --neighbors 0,3 \
  --quantiles 0.90 \
  --subsample-hours 1.0 \
  --out results/sweep_summary.csv
"""

import argparse, math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ---------------- I/O helpers ----------------

def read_any(path, parse_dates=None, usecols=None):
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path,
                       low_memory=False,
                       parse_dates=parse_dates if parse_dates else None,
                       usecols=usecols if usecols else None)

def write_any(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)

# ---------------- model loader ----------------

def load_model(model_path):
    m = joblib.load(model_path)
    if isinstance(m, dict):
        return m["model"], m.get("scaler", None), list(m["features"])
    # raw sklearn model
    feats = getattr(m, "feature_names_in_", None)
    return m, None, (list(feats) if feats is not None else None)

# ---------------- small utils ----------------

def parse_float_list(val):
    # "a:b:c" → a, a+c, ..., ≤b; or "x,y,z"
    if ":" in val:
        a,b,c = val.split(":")
        a,b,c = float(a), float(b), float(c)
        n = int(math.floor((b - a) / c + 1e-9)) + 1
        return [round(a + i*c, 10) for i in range(n)]
    return [float(x) for x in val.split(",") if str(x).strip()]

def future_max_timeaware(df: pd.DataFrame, target: str, hours: int) -> np.ndarray:
    """
    For each row (lat,lon,t), returns 1 if any target==1 occurs in (t, t+hours] for that cell.
    Handles missing hours and irregular cadence via timestamp searchsorted.
    """
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)

    for (_, _), g in df.groupby(["lat","lon"], sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y   = pd.to_numeric(g[target], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        t_ns = pd.to_datetime(g["time"], errors="coerce", utc=True).view("int64").to_numpy()

        order = np.argsort(t_ns, kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        t_sorted = t_ns[order]
        y_sorted = y[order]

        ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
        ps[1:] = np.cumsum(y_sorted)

        t_end_sorted = t_sorted + hour_ns
        end_pos = np.searchsorted(t_sorted, t_end_sorted, side="right")

        any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
        any_future = any_future_sorted[inv].astype(np.int8)
        out[idx] = any_future

    return out

def apply_persistence(df: pd.DataFrame, base_mask: np.ndarray, persist_h: int) -> np.ndarray:
    """Require ≥ persist_h consecutive kept hours within each (lat,lon)."""
    if persist_h <= 1:
        return base_mask
    out = np.zeros(len(df), dtype=bool)
    # Work per cell in time order
    for (_, _), g in df.assign(flag=base_mask.astype(np.int8)).groupby(["lat","lon"], sort=False):
        f = g["flag"].to_numpy()
        if f.sum() == 0:
            continue
        # rolling sum >= persist_h → keep
        # Use a cumulative trick (f is 0/1), but simple rolling is fine here:
        keep = pd.Series(f).rolling(persist_h, min_periods=persist_h).sum().to_numpy() >= persist_h
        out[g.index.to_numpy()] = keep
    return out

def indexize(df):
    ilat = pd.factorize(df["lat"].round(6))[0]
    ilon = pd.factorize(df["lon"].round(6))[0]
    return ilat, ilon

def neighborhood_filter(df: pd.DataFrame, flag_mask: np.ndarray, min_neighbors: int) -> np.ndarray:
    """Keep cells that have ≥ min_neighbors of 8-neighbors flagged at the same hour."""
    if min_neighbors <= 0:
        return flag_mask
    kept = np.zeros(len(df), dtype=bool)
    ilat, ilon = indexize(df)
    meta = pd.DataFrame({"time": df["time"], "ilat": ilat, "ilon": ilon, "flag": flag_mask.astype(np.int8)})

    for _, g in meta.groupby("time", sort=False):
        if g["flag"].sum() == 0:
            continue
        flagged = g.loc[g["flag"] == 1, ["ilat","ilon"]].to_numpy()
        # hash set of flagged coords
        present = set((int(i), int(j)) for i, j in flagged)
        nbr_ct = np.zeros(len(g), dtype=np.int16)
        IL = g["ilat"].to_numpy()
        JL = g["ilon"].to_numpy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nbr_ct += np.fromiter(((IL[k] + di, JL[k] + dj) in present for k in range(len(g))),
                                      count=len(g), dtype=np.int16)
        keep_h = (nbr_ct >= int(min_neighbors)) & (g["flag"].to_numpy().astype(bool))
        kept[g.index.to_numpy()] = keep_h
    return kept

def hourly_throttle(df: pd.DataFrame, score: np.ndarray, mask_in: np.ndarray, keep_quantile: float | None) -> np.ndarray:
    """Within each hour, among candidates mask_in, keep those with score ≥ quantile."""
    if keep_quantile is None:
        return mask_in
    out = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("time", sort=False).indices.items():
        sub = mask_in[idx]
        if not sub.any():
            continue
        s = score[idx][sub]
        q = np.quantile(s, keep_quantile) if s.size > 1 else s[0]
        keep = np.zeros_like(sub)
        keep[sub] = score[idx][sub] >= q
        out[idx] = keep
    return out

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Grid alert sweep runner")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="pregen", choices=["storm", "near_storm", "pregen"])
    ap.add_argument("--leads", default="24", help="Comma list, e.g. 24,48,72")
    ap.add_argument("--thr-grid", default="0.04:0.10:0.002", help="a:b:c or comma list")
    ap.add_argument("--persist", default="2", help="Comma list of consecutive-hour minima")
    ap.add_argument("--neighbors", default="3", help="Comma list of min 8-neighbors (0 disables)")
    ap.add_argument("--quantiles", default="0.90", help="Comma list of hourly keep quantiles (e.g. 0.90)")
    ap.add_argument("--subsample-hours", type=str, default="1.0",
                    help="Fraction (0<q≤1) or stride like 'every:3' to keep each n-th hour.")
    ap.add_argument("--out", default="results/sweep_summary.csv")
    args = ap.parse_args()

    leads     = [int(x) for x in str(args.leads).split(",")]
    thr_grid  = parse_float_list(args.thr_grid)
    persists  = [int(x) for x in str(args.persist).split(",")]
    neighbors = [int(x) for x in str(args.neighbors).split(",")]
    quantiles = [float(x) for x in str(args.quantiles).split(",")]

    print("== Sweep runner ==")
    print("Labelled :", args.labelled)
    print("Model    :", args.model)
    print("Target   :", args.target)
    print("Leads    :", leads)
    print("Thresh   :", f"{thr_grid[:3]} … {thr_grid[-3:]}" if len(thr_grid) > 6 else thr_grid)
    print("Persist  :", persists, "Neighbors:", neighbors, "Quantiles:", quantiles)
    print("Subsample:", args.subsample_hours)

    # Load model
    model, scaler, use_cols = load_model(args.model)
    if use_cols is None:
        raise ValueError("Model bundle missing 'features'; please save as {'model','scaler','features'}.")

    # Load labelled data with minimal columns
    need_cols = set(use_cols) | {"time","lat","lon", args.target}
    df = read_any(args.labelled, parse_dates=["time"])
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Labelled file missing required columns: {miss}")

    # Normalize and sort
    df = df.loc[:, list(need_cols)].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["time","lat","lon"]).sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    # Optional hour subsampling
    sub = str(args.subsample_hours).strip().lower()
    if sub.startswith("every:"):
        try:
            step = int(sub.split(":")[1])
            hours = df["time"].drop_duplicates().sort_values()
            keep_hours = set(hours.iloc[::max(1, step)].tolist())
            df = df[df["time"].isin(keep_hours)].copy()
        except Exception:
            pass
    else:
        frac = float(sub)
        if 0 < frac < 1.0:
            hours = df["time"].drop_duplicates().sort_values()
            keep_n = max(1, int(round(len(hours) * frac)))
            keep_hours = set(hours.sample(keep_n, random_state=42).tolist())
            df = df[df["time"].isin(keep_hours)].copy()

    df = df.sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)
    print(f"Rows after subsample: {len(df):,}")

    # Score once
    X = df[use_cols].to_numpy(float)
    if scaler is not None:
        X = scaler.transform(X)
    p = model.predict_proba(X)[:, 1]

    rows_out = []
    for lead in leads:
        # Build time-aware labels for this lead
        y = future_max_timeaware(df, args.target, lead)
        pos_total = int(y.sum())
        auc   = roc_auc_score(y, p) if (pos_total not in (0, len(y))) else np.nan
        prauc = average_precision_score(y, p) if not np.isnan(auc) else np.nan
        brier = brier_score_loss(y, p)

        for thr in thr_grid:
            base_mask = p >= thr
            for ph in persists:
                keep_persist = apply_persistence(df, base_mask, ph)
                for nb in neighbors:
                    keep_nb = neighborhood_filter(df, keep_persist, nb)
                    for q in quantiles:
                        keep_final = hourly_throttle(df, p, keep_nb, q)

                        pred_pos = int(keep_final.sum())
                        if pred_pos == 0:
                            rows_out.append(dict(
                                lead=lead, thr=thr, persist=ph, neighbors=nb, quantile=q,
                                AUC=float(auc) if not np.isnan(auc) else np.nan,
                                PRAUC=float(prauc) if not np.isnan(prauc) else np.nan,
                                Brier=float(brier), F1=np.nan, Precision=np.nan, Recall=np.nan,
                                Coverage=0.0, alerts=0, pos=pos_total, rows=len(df)
                            ))
                            continue

                        # Confusion numbers from the final mask
                        tp = int((keep_final & (y == 1)).sum())
                        fp = pred_pos - tp
                        fn = pos_total - tp

                        precision = tp / pred_pos if pred_pos > 0 else np.nan
                        recall    = tp / pos_total if pos_total > 0 else np.nan
                        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else np.nan
                        coverage = pred_pos / float(len(df))

                        rows_out.append(dict(
                            lead=lead, thr=thr, persist=ph, neighbors=nb, quantile=q,
                            AUC=float(auc) if not np.isnan(auc) else np.nan,
                            PRAUC=float(prauc) if not np.isnan(prauc) else np.nan,
                            Brier=float(brier),
                            F1=float(f1) if not np.isnan(f1) else np.nan,
                            Precision=float(precision) if not np.isnan(precision) else np.nan,
                            Recall=float(recall) if not np.isnan(recall) else np.nan,
                            Coverage=float(coverage),
                            alerts=pred_pos, pos=pos_total, rows=len(df)
                        ))

    out_path = Path(args.out)
    write_any(out_path, pd.DataFrame(rows_out))
    print(f"Wrote {out_path} rows: {len(rows_out)}")

if __name__ == "__main__":
    main()