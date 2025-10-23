#!/usr/bin/env python3
"""
intensity_correlation.py
Match best-track cyclone intensity (vmax, pmin) to geometric-kernel probabilities.

Inputs
------
--base-alerts  : CSV(.gz) from apply_thresholds.py (must include: time, lat, lon, prob)
--lead-hours   : int, the forecast lead (hours) used to produce base-alerts
--tracks       : CSV with cyclone snapshots (time, lat, lon, and intensity columns)
--out-csv      : where to save per-point matches & stats

Track column mapping (override if your file differs):
  --trk-time-col  default: time       # timestamp of observation (UTC)
  --trk-lat-col   default: lat
  --trk-lon-col   default: lon
  --trk-vmax-col  default: vmax       # max sustained wind (e.g., kt or m/s)
  --trk-pmin-col  default: pmin       # min sea-level pressure (hPa)

Matching controls:
  --radius-deg      default: 0.75     # spatial match radius in degrees
  --time-tol-hours  default: 1        # allowed clock mismatch after shifting by lead
  --agg             default: max      # aggregator over matched grid: {max, mean}
  --min-prob        default: 0.0      # ignore probs below this when aggregating

Outputs
-------
- CSV with columns: obs_time, match_time, lat, lon, vmax, pmin, pgeom, n_cells, dt_hours, r_km (median)
- Prints Pearson/Spearman with vmax and -pmin
- (optional) --fit-powerlaw prints v ~ a*p^b on valid (p>0) rows
"""

import argparse, os, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EARTH_KM_PER_DEG = 111.32

def haversine_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance in km from deg arrays."""
    rlat1 = np.radians(lat1); rlat2 = np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    return 6371.0 * (2 * np.arcsin(np.sqrt(a)))

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
    ap.add_argument("--agg", choices=["max","mean"], default="max")
    ap.add_argument("--min-prob", type=float, default=0.0)

    # extras
    ap.add_argument("--fit-powerlaw", action="store_true", help="Fit vmax ~ a * pgeom^b on matched rows with p>0.")
    args = ap.parse_args()

    if not os.path.exists(args.base_alerts):
        raise FileNotFoundError(args.base_alerts)
    if not os.path.exists(args.tracks):
        raise FileNotFoundError(args.tracks)

    # Load base alerts (probabilities)
    base = pd.read_csv(args.base_alerts, compression="infer")
    req_cols = {"time","lat","lon","prob"}
    if not req_cols.issubset(base.columns):
        raise ValueError(f"{args.base_alerts} must include columns {req_cols}")
    base["time"] = pd.to_datetime(base["time"], utc=True, errors="coerce").dt.tz_localize(None)
    base = base.dropna(subset=["time","lat","lon","prob"]).reset_index(drop=True)
    base["lat"] = pd.to_numeric(base["lat"], errors="coerce")
    base["lon"] = pd.to_numeric(base["lon"], errors="coerce")
    base = base.dropna(subset=["lat","lon"]).reset_index(drop=True)
    base["time_h"] = base["time"].dt.floor("h")

    # Load tracks
    trk = pd.read_csv(args.tracks)
    for c in [args.trk_time_col, args.trk_lat_col, args.trk_lon_col]:
        if c not in trk.columns:
            raise ValueError(f"tracks missing column: {c}")
    trk["_t_obs"] = pd.to_datetime(trk[args.trk_time_col], utc=True, errors="coerce").dt.tz_localize(None)
    trk = trk.dropna(subset=["_t_obs"]).reset_index(drop=True)

    # shift observation time back by lead to match forecast stamp
    trk["_t_match"] = trk["_t_obs"] - pd.to_timedelta(args.lead_hours, unit="h")
    trk["_t0"] = trk["_t_match"].dt.floor("h")

    # optional intensity fields
    vmax = trk[args.trk_vmax_col] if args.trk_vmax_col in trk.columns else pd.Series([np.nan]*len(trk))
    pmin = trk[args.trk_pmin_col] if args.trk_pmin_col in trk.columns else pd.Series([np.nan]*len(trk))

    # Index base by hour for quick lookup
    groups = {t: g for t, g in base.groupby("time_h")}

    rows = []
    for i, r in trk.iterrows():
        t0 = r["_t0"]
        # time tolerance (± tol)
        candidates = []
        for dt in range(-int(math.floor(args.time_tol_hours)), int(math.floor(args.time_tol_hours))+1):
            candidates.append(t0 + pd.Timedelta(hours=dt))

        # concat matched hours
        frame = []
        for tt in candidates:
            g = groups.get(tt)
            if g is not None:
                frame.append(g)
        if frame:
            gcat = pd.concat(frame, ignore_index=True)
        else:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": pd.NaT,
                "lat": r[args.trk_lat_col], "lon": r[args.trk_lon_col],
                "vmax": vmax.iloc[i] if len(vmax)>i else np.nan,
                "pmin": pmin.iloc[i] if len(pmin)>i else np.nan,
                "pgeom": np.nan, "n_cells": 0, "dt_hours": np.nan, "r_km_med": np.nan
            })
            continue

        # spatial filter
        lat0 = float(r[args.trk_lat_col]); lon0 = float(r[args.trk_lon_col])
        # cheap prefilter in degrees
        sel = (gcat["lat"].between(lat0-args.radius_deg, lat0+args.radius_deg) &
               gcat["lon"].between(lon0-args.radius_deg, lon0+args.radius_deg))
        gsel = gcat.loc[sel].copy()
        if gsel.empty:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": gcat["time_h"].iat[0] if len(gcat)>0 else pd.NaT,
                "lat": lat0, "lon": lon0,
                "vmax": vmax.iloc[i] if len(vmax)>i else np.nan,
                "pmin": pmin.iloc[i] if len(pmin)>i else np.nan,
                "pgeom": np.nan, "n_cells": 0, "dt_hours": float((r["_t_match"] - (gcat["time_h"].iat[0] if len(gcat)>0 else r["_t_match"])).total_seconds()/3600) if len(gcat)>0 else np.nan,
                "r_km_med": np.nan
            })
            continue

        # precise distance
        gsel["dist_km"] = haversine_deg(lat0, lon0, gsel["lat"].to_numpy(), gsel["lon"].to_numpy())
        gsel = gsel.loc[gsel["dist_km"] <= args.radius_deg*EARTH_KM_PER_DEG]
        if args.min_prob > 0:
            gsel = gsel.loc[gsel["prob"] >= args.min_prob]

        if gsel.empty:
            rows.append({
                "obs_time": r["_t_obs"], "match_time": pd.NaT,
                "lat": lat0, "lon": lon0,
                "vmax": vmax.iloc[i] if len(vmax)>i else np.nan,
                "pmin": pmin.iloc[i] if len(pmin)>i else np.nan,
                "pgeom": np.nan, "n_cells": 0, "dt_hours": np.nan, "r_km_med": np.nan
            })
            continue

        # aggregate probability
        if args.agg == "max":
            pgeom = float(gsel["prob"].max())
        else:
            pgeom = float(gsel["prob"].mean())

        dt_hours = float((r["_t_match"] - gsel["time_h"].mode().iat[0]).total_seconds()/3600) if "time_h" in gsel else 0.0
        rows.append({
            "obs_time": r["_t_obs"], "match_time": gsel["time_h"].mode().iat[0] if "time_h" in gsel else pd.NaT,
            "lat": lat0, "lon": lon0,
            "vmax": vmax.iloc[i] if len(vmax)>i else np.nan,
            "pmin": pmin.iloc[i] if len(pmin)>i else np.nan,
            "pgeom": pgeom, "n_cells": int(len(gsel)),
            "dt_hours": dt_hours,
            "r_km_med": float(gsel["dist_km"].median())
        })

    out = pd.DataFrame(rows)
    out.sort_values("obs_time", inplace=True, ignore_index=True)

    # correlations
    valid = out.dropna(subset=["pgeom"])
    def _corr(x, y, name):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            print(f"[WARN] Not enough pairs for {name} (n={mask.sum()})")
            return np.nan, np.nan
        try:
            rp = pearsonr(x[mask], y[mask])
            rs = spearmanr(x[mask], y[mask])
            print(f"{name}: Pearson r={rp.statistic:.3f} (p={rp.pvalue:.2e}) | Spearman ρ={rs.statistic:.3f} (p={rs.pvalue:.2e})")
            return rp.statistic, rs.statistic
        except Exception:
            return np.nan, np.nan

    print("\n== Intensity correlation ==")
    if "vmax" in valid.columns and valid["vmax"].notna().any():
        _corr(valid["pgeom"].to_numpy(), valid["vmax"].to_numpy(), "vmax vs pgeom")
    if "pmin" in valid.columns and valid["pmin"].notna().any():
        _corr(valid["pgeom"].to_numpy(), (-valid["pmin"]).to_numpy(), "-pmin vs pgeom")

    if args.fit_powerlaw and valid["pgeom"].gt(0).sum() >= 3 and valid["vmax"].notna().sum() >= 3:
        v = valid.loc[valid["pgeom"] > 0, ["pgeom","vmax"]].dropna()
        X = np.log(v["pgeom"].to_numpy())
        Y = np.log(v["vmax"].to_numpy())
        b, a = np.polyfit(X, Y, 1)  # Y ≈ a + b X
        A = float(np.exp(a)); B = float(b)
        print(f"Power-law fit (vmax ≈ A * pgeom^B):  A={A:.3f},  B={B:.3f}")

    # write output
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    valid_cols = ["obs_time","match_time","lat","lon","vmax","pmin","pgeom","n_cells","dt_hours","r_km_med"]
    valid_cols = [c for c in valid_cols if c in out.columns]
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote per-point matches → {args.out_csv} (n={len(out)})")

if __name__ == "__main__":
    main()