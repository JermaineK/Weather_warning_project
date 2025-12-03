#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intensity_analysis.py — correlate geometric-kernel probability with storm intensity.

Adds:
- Optional per-storm summary (--by-storm)
- Optional --use-vmax-only
- Clearer logging of chosen columns and sample sizes
- Returns structured summary if imported
"""

import argparse, os, math
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# ---------------- utilities ----------------

def read_any(path, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        print(f"[read] reading Parquet: {path}")
        return pd.read_parquet(path, **kw)
    print(f"[read] reading CSV: {path}")
    return pd.read_csv(path, **kw)

def write_any(path, df: pd.DataFrame):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp)

def _to_utc_naive(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none": return x
    if mode == "0..360": return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180

def _parse_area(aoi: str | None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def _crop_df(df: pd.DataFrame, aoi, lat_col="lat", lon_col="lon"):
    if not aoi: return df
    latN, lonW, latS, lonE = aoi
    df = df[(df[lat_col] <= latN) & (df[lat_col] >= latS)]
    if lonW <= lonE:
        df = df[(df[lon_col] >= lonW) & (df[lon_col] <= lonE)]
    else:
        df = df[(df[lon_col] >= lonW) | (df[lon_col] <= lonE)]
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ---------------- loaders ----------------

def load_alerts(path, prob_col=None, time_offset_h=0.0,
                normalize_lon="none", area=None):
    df = read_any(path)
    if prob_col is None:
        for cand in ["prob", "p", "p_pregen", "probability"]:
            if cand in df.columns:
                prob_col = cand
                break
    if prob_col not in df.columns:
        raise ValueError(f"No prob-like column found in {path}")

    print(f"[alerts] using prob column '{prob_col}'")

    t = _to_utc_naive(df["time"])
    if time_offset_h:
        t = t + pd.to_timedelta(time_offset_h, unit="h")

    out = pd.DataFrame({
        "time": t,
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": _norm_lon(df["lon"], normalize_lon),
        "prob": pd.to_numeric(df[prob_col], errors="coerce")
    }).dropna().reset_index(drop=True)

    out = _crop_df(out, area)
    out["time_h"] = out["time"].dt.floor("h")
    out = (out.groupby(["time_h","lat","lon"], as_index=False)["prob"].max())
    return out

def load_tracks(path, time_offset_h=0.0, normalize_lon="none", area=None):
    tr = read_any(path)
    tcol = "obs_time" if "obs_time" in tr.columns else "time"
    for req in ["lat","lon","vmax"]:
        if req not in tr.columns:
            raise ValueError(f"{path}: missing column {req}")
    t = _to_utc_naive(tr[tcol])
    if time_offset_h: t = t + pd.to_timedelta(time_offset_h, unit="h")
    out = pd.DataFrame({
        "obs_time": t,
        "lat": pd.to_numeric(tr["lat"], errors="coerce"),
        "lon": _norm_lon(tr["lon"], normalize_lon),
        "vmax": pd.to_numeric(tr["vmax"], errors="coerce"),
        "pmin": pd.to_numeric(tr["pmin"], errors="coerce") if "pmin" in tr.columns else np.nan,
        "name": tr["name"] if "name" in tr.columns else None,
    }).dropna(subset=["obs_time","lat","lon"])
    return _crop_df(out, area)

# ---------------- core matching ----------------

def match_intensity(alerts_df, tracks_df, radius_deg=0.75, time_tol_h=1.0, agg="max"):
    out = []
    by_hour = {t: g for t,g in alerts_df.groupby("time_h")}
    for _, row in tracks_df.iterrows():
        t0, lat0, lon0 = row["obs_time"], row["lat"], row["lon"]
        hwin = range(-int(np.floor(time_tol_h)), int(np.ceil(time_tol_h))+1)
        candidates=[]
        for dh in hwin:
            tt=(t0+timedelta(hours=dh)).floor("h")
            g=by_hour.get(tt)
            if g is None: continue
            box=g[(g["lat"].between(lat0-radius_deg,lat0+radius_deg)) &
                  (g["lon"].between(lon0-radius_deg,lon0+radius_deg))]
            if box.empty: continue
            box["d_km"]=haversine_km(lat0,lon0,box["lat"],box["lon"])
            sel=box.loc[box["d_km"]<=radius_deg*111.32]
            if sel.empty: continue
            sel["time"]=tt; sel["dt_hours"]=(sel["time"]-t0)/np.timedelta64(1,"h")
            candidates.append(sel)
        if not candidates: continue
        cand=pd.concat(candidates,ignore_index=True)
        best = cand.loc[cand["prob"].idxmax()] if agg=="max" else cand.assign(prob=cand["prob"].mean()).iloc[0]
        out.append({
            "obs_time":t0,"match_time":best["time"],"lat":lat0,"lon":lon0,
            "vmax":row["vmax"],"pmin":row["pmin"],"name":row.get("name",""),
            "pgeom":float(best["prob"]),"dt_hours":float(best["dt_hours"]),
            "d_km_min":float(best["d_km"]),"n_cells":int(len(cand))
        })
    return pd.DataFrame(out)

def safe_corr(x,y,label,min_n=5):
    x,y=np.asarray(x),np.asarray(y)
    mask=np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    n=len(x)
    if n<min_n or np.allclose(x,x.mean()) or np.allclose(y,y.mean()):
        print(f"{label:<18s} n={n:<4d} insufficient")
        return None
    rs,ps=spearmanr(x,y)
    rk,pk=kendalltau(x,y)
    print(f"{label:<18s} Spearman={rs:6.3f}  (p={ps:6.2g}) | Kendall={rk:6.3f}  (p={pk:6.2g}) | n={n}")
    return dict(label=label,n=n,spearman=rs,p_spear=ps,kendall=rk,p_kend=pk)

# ---------------- main ----------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--alerts",required=True)
    ap.add_argument("--tracks",required=True)
    ap.add_argument("--radius-deg",type=float,default=0.75)
    ap.add_argument("--time-tol-hours",type=float,default=1.0)
    ap.add_argument("--prob-col",default=None)
    ap.add_argument("--agg",choices=["max","mean"],default="max")
    ap.add_argument("--out-csv",default="results/intensity/intensity_match.csv")
    ap.add_argument("--by-storm",default=None,help="optional CSV to save per-storm correlation summary")
    ap.add_argument("--use-vmax-only",action="store_true",help="skip pmin correlations")
    ap.add_argument("--normalize-lon",choices=["none","-180..180","0..360"],default="-180..180")
    ap.add_argument("--area",default=None)
    args=ap.parse_args()

    alerts=load_alerts(args.alerts,prob_col=args.prob_col,normalize_lon=args.normalize_lon,area=_parse_area(args.area))
    tracks=load_tracks(args.tracks,normalize_lon=args.normalize_lon,area=_parse_area(args.area))
    matches=match_intensity(alerts,tracks,args.radius_deg,args.time_tol_hours,args.agg)
    if matches.empty:
        print("[intensity] No matches found.")
        return

    matches=matches.replace([np.inf,-np.inf],np.nan).dropna(subset=["pgeom","vmax"])
    print(f"[intensity] matched={len(matches)}  median_d_km={matches['d_km_min'].median():.1f}")

    summary=[]
    summary.append(safe_corr(matches["pgeom"],matches["vmax"],"vmax ~ pgeom"))
    if not args.use_vmax_only and "pmin" in matches:
        summary.append(safe_corr(matches["pgeom"],-matches["pmin"],"-pmin ~ pgeom"))

    if args.by_storm and "name" in matches.columns and matches["name"].notna().any():
        rows=[]
        for nm,g in matches.groupby("name"):
            c=safe_corr(g["pgeom"],g["vmax"],f"{nm}")
            if c: c["storm"]=nm; rows.append(c)
        if rows:
            pd.DataFrame(rows).to_csv(args.by_storm,index=False)
            print(f"[intensity] per-storm summary -> {args.by_storm}")

    write_any(args.out_csv,matches)
    print(f"[intensity] wrote {args.out_csv}")

if __name__=="__main__":
    main()