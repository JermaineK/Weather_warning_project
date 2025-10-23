#!/usr/bin/env python
import argparse, math, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

def parse_float_list(val):
    if ":" in val:
        a,b,c = val.split(":")
        a,b,c = float(a), float(b), float(c)
        n = int(math.floor((b - a) / c + 1e-9)) + 1
        return [round(a + i*c, 6) for i in range(n)]
    return [float(x) for x in val.split(",") if x.strip()]

def load_model(model_path):
    m = joblib.load(model_path)
    if isinstance(m, dict):
        return m["model"], m.get("scaler", None), m["features"]
    return m, None, getattr(m, "feature_names_in_", None)

def future_max_window(series, hours):
    y = series.astype(int)
    f = y.iloc[::-1].shift(1).rolling(hours, min_periods=1).max()
    return f.iloc[::-1].fillna(0).astype(int)

def build_lead_labels(df, target_col, lead_h):
    return df.groupby(["lat","lon"], sort=False)[target_col].transform(
        lambda s: future_max_window(s, lead_h)
    ).to_numpy()

def apply_persistence_mask(df, base_mask, persist_h):
    """Require ≥persist_h consecutive flagged hours per (lat,lon)."""
    if persist_h <= 1:
        return base_mask
    out = np.zeros(len(df), dtype=bool)
    tmp = pd.DataFrame({"lat":df["lat"], "lon":df["lon"], "flag":base_mask.astype(int)})
    for _, idx in tmp.groupby(["lat","lon"], sort=False).indices.items():
        s = tmp.loc[idx, "flag"]
        keep = s.rolling(persist_h, min_periods=persist_h).sum().to_numpy() >= persist_h
        out[idx] = keep
    return out

def indexize(df):
    ilat = pd.factorize(df["lat"])[0]
    ilon = pd.factorize(df["lon"])[0]
    return ilat, ilon

def neighborhood_filter(df, flag_mask, min_neighbors):
    if min_neighbors <= 0:
        return flag_mask
    out = np.zeros(len(df), dtype=bool)
    ilat, ilon = indexize(df)
    df2 = df.loc[:, ["time"]].copy()
    df2["ilat"] = ilat
    df2["ilon"] = ilon
    df2["flag"] = flag_mask.astype(int)
    for _, chunk in df2.groupby("time", sort=False):
        if len(chunk)==0 or chunk["flag"].sum()==0:
            continue
        base = chunk[["ilat","ilon","flag"]].copy()
        present = dict.fromkeys(base.loc[base["flag"]==1, "ilat"].astype(str)+"_"+base.loc[base["flag"]==1,"ilon"].astype(str), 1)
        il = base["ilat"].to_numpy(); jl = base["ilon"].to_numpy(); n = len(base)
        neigh_ct = np.zeros(n, dtype=np.int16)
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di==0 and dj==0: continue
                k = (il+di).astype(int).astype(str)+"_"+(jl+dj).astype(int).astype(str)
                hit = pd.Series(k).map(present).fillna(0).to_numpy().astype(np.int16)
                neigh_ct += hit
        keep_mask = neigh_ct >= int(min_neighbors)
        out[chunk.index] = keep_mask & (chunk["flag"].to_numpy().astype(bool))
    return out

def hourly_throttle(df, score, base_mask, keep_quantile):
    if keep_quantile is None:
        return base_mask
    kept = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("time", sort=False).indices.items():
        cand = base_mask[idx]
        if not cand.any(): continue
        s = score[idx][cand]
        q = np.quantile(s, keep_quantile)
        subkeep = np.zeros_like(cand)
        subkeep[cand] = score[idx][cand] >= q
        kept[idx] = subkeep
    return kept

def main():
    ap = argparse.ArgumentParser(description="Grid alert sweep runner")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="pregen", choices=["storm","near_storm","pregen"])
    ap.add_argument("--leads", default="24")
    ap.add_argument("--thr-grid", default="0.04:0.10:0.002")
    ap.add_argument("--persist", default="2")
    ap.add_argument("--neighbors", default="3")
    ap.add_argument("--quantiles", default="0.90")
    ap.add_argument("--subsample-hours", type=float, default=1.0)
    ap.add_argument("--out", default="results/sweep_summary.csv")
    args = ap.parse_args()

    leads      = [int(x) for x in args.leads.split(",")]
    thr_grid   = parse_float_list(args.thr_grid)
    persists   = [int(x) for x in args.persist.split(",")]
    neighbors  = [int(x) for x in args.neighbors.split(",")]
    quantiles  = [float(x) for x in args.quantiles.split(",")]

    print("== Sweep runner ==")
    print("Labelled :", args.labelled)
    print("Model    :", args.model)
    print("Target   :", args.target)
    print("Leads    :", leads)
    print("Thresh   :", f"{thr_grid[:3]} … {thr_grid[-3:]} (n={len(thr_grid)})" if len(thr_grid)>6 else thr_grid)
    print("Persist  :", persists, "Neighbors:", neighbors, "Quantiles:", quantiles)
    print("Subsample hours:", args.subsample_hours)

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    model, scaler, use_cols = load_model(args.model)
    need = set(use_cols + ["time","lat","lon", args.target])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Labelled file missing required columns: {miss}")

    if args.subsample_hours < 1.0:
        hrs = df["time"].drop_duplicates().sort_values()
        keep_n = max(1, int(len(hrs) * args.subsample_hours))
        keep_hours = set(hrs.sample(keep_n, random_state=42).tolist())
        df = df[df["time"].isin(keep_hours)].copy()
        df = df.sort_values(["time","lat","lon"], kind="stable").reset_index(drop=True)
        print(f"Subsampled hours → rows: {len(df):,}")

    X = df[use_cols].to_numpy(float)
    if scaler is not None:
        X = scaler.transform(X)
    p = model.predict_proba(X)[:,1]

    rows = []
    for lead in leads:
        y = build_lead_labels(df, args.target, lead)
        base_auc   = roc_auc_score(y, p) if (y.sum() not in (0, len(y))) else np.nan
        base_ap    = average_precision_score(y, p) if not np.isnan(base_auc) else np.nan
        base_brier = brier_score_loss(y, p)

        for thr in thr_grid:
            base_mask = p >= thr
            for ph in persists:
                keep = apply_persistence_mask(df, base_mask, ph)
                for nb in neighbors:
                    keep2 = neighborhood_filter(df, keep, nb)
                    for q in quantiles:
                        keep3 = hourly_throttle(df, p, keep2, q)
                        if keep3.sum() == 0:
                            rows.append(dict(
                                lead=lead, thr=thr, persist=ph, neighbors=nb, quantile=q,
                                AUC=base_auc, PRAUC=base_ap, Brier=base_brier,
                                F1=np.nan, Precision=np.nan, Recall=np.nan,
                                Coverage=0.0, alerts=0, pos=int(y.sum()), rows=len(df)
                            ))
                            continue
                        cov = float(keep3.mean())
                        yy = y[keep3]; pp = p[keep3]
                        pr, rc, thv = precision_recall_curve(yy, pp)
                        f1 = (2*pr*rc/(pr+rc+1e-9))
                        best_f1 = float(np.nanmax(f1)) if len(f1) else np.nan
                        precision = float(yy.sum()/len(yy)) if len(yy) else 0.0
                        recall    = float(yy.sum()/y.sum()) if y.sum()>0 else np.nan
                        rows.append(dict(
                            lead=lead, thr=thr, persist=ph, neighbors=nb, quantile=q,
                            AUC=float(base_auc), PRAUC=float(base_ap), Brier=float(base_brier),
                            F1=best_f1, Precision=precision, Recall=recall,
                            Coverage=cov, alerts=int(keep3.sum()),
                            pos=int(y.sum()), rows=len(df)
                        ))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} rows: {len(rows)}")

if __name__ == "__main__":
    main()