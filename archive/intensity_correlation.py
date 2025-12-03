#!/usr/bin/env python3
# intensity_correlation.py (patched)

import argparse, os, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EARTH_KM = 6371.0
KM_PER_DEG = 111.32

def haversine_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance (km), safe across the dateline."""
    rlat1 = np.radians(lat1); rlat2 = np.radians(lat2)
    dlat = rlat2 - rlat1
    # normalize Δλ to (-180, 180]
    dlon = np.radians(((lon2 - lon1 + 540.0) % 360.0) - 180.0)
    a = np.sin(dlat/2.0)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def pick_prob_col(df):
    for c in ("prob", "risk", "p", "score"):
        if c in df.columns:
            return c
    raise ValueError("Alerts file must include one of: prob, risk, p, score")

def within_time_tol(base_by_hour, t0, tol_hours):
    """Return concatenated frames for all hours whose |Δt|≤tol_hours from t0."""
    hours = []
    # scan integer-hour neighbors that could be within tol
    span = int(math.ceil(tol_hours))
    for dt in range(-span, span + 1):
        cand = t0 + pd.Timedelta(hours=dt)
        g = base_by_hour.get(cand)
        if g is not None:
            hours.append(g)
    if not hours:
        return None
    cat = pd.concat(hours, ignore_index=True)
    # strict filter by absolute delta in hours
    ah = (cat["time_h"] - t0).dt.total_seconds().abs() / 3600.0
    return cat.loc[ah <= tol_hours].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Correlate GK probabilities with cyclone intensity.")
    ap.add_argument("--base-alerts", required=True)
    ap.add_argument("--lead-hours", type=int, required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-csv", required=True)

    # track mapping
    ap.add_argument("--trk-time-col", default="time")
    ap.add_argument("--trk-lat-col",  default="lat")
    ap.add_argument("--trk-lon-col",  default="lon")
    ap.add_argument("--trk-vmax-col", default="vmax")
    ap.add_argument("--trk-pmin-col", default="pmin")

    # matching
    ap.add_argument("--radius-deg", type=float, default=0.75)
    ap.add_argument("--time-tol-hours", type=float, default=1.0)
    ap.add_argument("--agg", choices=["max","mean","wmean"], default="max",
                    help="wmean: distance-weighted mean (1/(1+r_km))")
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--wrap-lon", action="store_true", help="Normalize longitudes to 0..360 to avoid seams")

    # extras
    ap.add_argument("--fit-powerlaw", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.base_alerts): raise FileNotFoundError(args.base_alerts)
    if not os.path.exists(args.tracks):      raise FileNotFoundError(args.tracks)

    # Load base alerts
    base = pd.read_csv(args.base_alerts, compression="infer")
    if "time" not in base.columns or "lat" not in base.columns or "lon" not in base.columns:
        raise ValueError("alerts must include columns: time, lat, lon, and a prob-like column")
    prob_col = pick_prob_col(base)
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce").dt.tz_localize(None)
    base = base.dropna(subset=["time","lat","lon", prob_col]).reset_index(drop=True)
    if args.wrap_lon:
        base["lon"] = np.mod(base["lon"].to_numpy(float), 360.0)
    base["time_h"] = base["time"].dt.floor("h")

    # Load tracks
    trk = pd.read_csv(args.tracks)
    need = [args.trk_time_col, args.trk_lat_col, args.trk_lon_col]
    for c in need:
        if c not in trk.columns:
            raise ValueError(f"tracks missing column: {c}")
    trk["_t_obs"] = pd.to_datetime(trk[args.trk_time_col], utc=True, errors="coerce").dt.tz_localize(None)
    trk = trk.dropna(subset=["_t_obs"]).reset_index(drop=True)
    if args.wrap_lon:
        trk[args.trk_lon_col] = np.mod(trk[args.trk_lon_col].to_numpy(float), 360.0)

    # Shift obs back by lead to match forecast timestamp; floor to hour for indexing
    trk["_t_match"] = trk["_t_obs"] - pd.to_timedelta(args.lead_hours, unit="h")
    trk["_t0"] = trk["_t_match"].dt.floor("h")

    vmax = trk[args.trk_vmax_col] if args.trk_vmax_col in trk.columns else pd.Series(np.nan, index=trk.index)
    pmin = trk[args.trk_pmin_col] if args.trk_pmin_col in trk.columns else pd.Series(np.nan, index=trk.index)

    base_by_hour = {t: g for t, g in base.groupby("time_h")}

    rows = []
    Rkm = args.radius_deg * KM_PER_DEG
    for i, r in trk.iterrows():
        t0 = r["_t0"]
        gcat = within_time_tol(base_by_hour, t0, args.time_tol_hours)
        if gcat is None or gcat.empty:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": pd.NaT,
                "lat": float(r[args.trk_lat_col]), "lon": float(r[args.trk_lon_col]),
                "vmax": float(vmax.iloc[i]) if vmax.notna().iloc[i] else np.nan,
                "pmin": float(pmin.iloc[i]) if pmin.notna().iloc[i] else np.nan,
                "pgeom": np.nan, "n_cells": 0, "dt_hours": np.nan, "r_km_med": np.nan
            })
            continue

        lat0 = float(r[args.trk_lat_col]); lon0 = float(r[args.trk_lon_col])

        # fast deg box, then exact haversine
        sel = (gcat["lat"].between(lat0 - args.radius_deg, lat0 + args.radius_deg) &
               gcat["lon"].between(lon0 - args.radius_deg, lon0 + args.radius_deg))
        gsel = gcat.loc[sel].copy()
        if gsel.empty:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": gcat["time_h"].iloc[0],
                "lat": lat0, "lon": lon0, "vmax": float(vmax.iloc[i]) if vmax.notna().iloc[i] else np.nan,
                "pmin": float(pmin.iloc[i]) if pmin.notna().iloc[i] else np.nan,
                "pgeom": np.nan, "n_cells": 0,
                "dt_hours": float((r["_t_match"] - gcat["time_h"].iloc[0]).total_seconds()/3600.0),
                "r_km_med": np.nan
            })
            continue

        gsel["dist_km"] = haversine_deg(lat0, lon0, gsel["lat"].to_numpy(), gsel["lon"].to_numpy())
        gsel = gsel.loc[gsel["dist_km"] <= Rkm]
        if args.min_prob > 0:
            gsel = gsel.loc[gsel[prob_col] >= args.min_prob]
        if gsel.empty:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": pd.NaT,
                "lat": lat0, "lon": lon0, "vmax": float(vmax.iloc[i]) if vmax.notna().iloc[i] else np.nan,
                "pmin": float(pmin.iloc[i]) if pmin.notna().iloc[i] else np.nan,
                "pgeom": np.nan, "n_cells": 0, "dt_hours": np.nan, "r_km_med": np.nan
            })
            continue

        # aggregators
        if args.agg == "max":
            pgeom = float(gsel[prob_col].max())
        elif args.agg == "mean":
            pgeom = float(gsel[prob_col].mean())
        else:  # wmean
            w = 1.0 / (1.0 + gsel["dist_km"].to_numpy())
            pgeom = float(np.average(gsel[prob_col].to_numpy(), weights=w))

        # match_time ~ majority hour among selected cells
        mt = gsel["time_h"].mode().iloc[0]
        dt_hours = float((r["_t_match"] - mt).total_seconds()/3600.0)

        rows.append({
            "obs_time": r["_t_obs"], "match_time": mt,
            "lat": lat0, "lon": lon0,
            "vmax": float(vmax.iloc[i]) if vmax.notna().iloc[i] else np.nan,
            "pmin": float(pmin.iloc[i]) if pmin.notna().iloc[i] else np.nan,
            "pgeom": pgeom, "n_cells": int(len(gsel)),
            "dt_hours": dt_hours, "r_km_med": float(gsel["dist_km"].median())
        })

    out = pd.DataFrame(rows).sort_values("obs_time", ignore_index=True)

    # correlations
    print("\n== Intensity correlation ==")
    valid = out.dropna(subset=["pgeom"]).copy()
    def safe_corr(x, y, name):
        x = np.asarray(x, float); y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            print(f"{name}: n<3")
            return
        try:
            rp = pearsonr(x[mask], y[mask])
            rs = spearmanr(x[mask], y[mask])
            print(f"{name}: Pearson r={rp.statistic:.3f} (p={rp.pvalue:.2e}) | Spearman ρ={rs.statistic:.3f} (p={rs.pvalue:.2e})")
        except Exception:
            print(f"{name}: failed")
    if "vmax" in valid.columns and valid["vmax"].notna().any():
        safe_corr(valid["pgeom"], valid["vmax"], "vmax vs pgeom")
    if "pmin" in valid.columns and valid["pmin"].notna().any():
        safe_corr(valid["pgeom"], -valid["pmin"], "-pmin vs pgeom")

    if args.fit_powerlaw and (valid["pgeom"] > 0).sum() >= 3 and valid["vmax"].notna().sum() >= 3:
        v = valid.loc[valid["pgeom"] > 0, ["pgeom","vmax"]].dropna()
        X = np.log(v["pgeom"].to_numpy()); Y = np.log(v["vmax"].to_numpy())
        b, a = np.polyfit(X, Y, 1)
        A = float(np.exp(a)); B = float(b)
        print(f"Power-law fit (vmax ≈ A * pgeom^B):  A={A:.3f},  B={B:.3f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keep = ["obs_time","match_time","lat","lon","vmax","pmin","pgeom","n_cells","dt_hours","r_km_med"]
    out[keep].to_csv(args.out_csv, index=False)
    print(f"Wrote per-point matches -> {args.out_csv} (n={len(out)})")

if __name__ == "__main__":
    main()