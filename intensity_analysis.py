#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intensity_analysis.py
Correlate geometric kernel probability with storm intensity.

Inputs:
  --alerts   CSV/Parquet from apply_thresholds.py (must contain prob-like column, time, lat, lon)
  --tracks   CSV/Parquet with columns: obs_time (or time), lat, lon, vmax, pmin (plus optional name/id)

Outputs:
  - Spearman/Kendall rank correlations (if n>=5 and non-constant)
  - Optional power-law fit vmax ≈ a * pgeom^b (if n>=3 and pgeom>0)
  - Matched CSV with: obs_time, match_time, lat, lon, vmax, pmin, pgeom, dt_hours, d_km_min, n_cells
"""

import argparse, os, math
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# ---------------- I/O + common utils ----------------

def read_any(path, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path, **kw)
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
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def _parse_area(aoi: str | None):
    if not aoi: return None
    try:
        latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
        return latN, lonW, latS, lonE
    except Exception:
        raise ValueError("--area must be 'latN,lonW,latS,lonE' (e.g., -10,135,-30,155)")

def _crop_df(df: pd.DataFrame, aoi, lat_col="lat", lon_col="lon"):
    if not aoi:
        return df
    latN, lonW, latS, lonE = aoi
    df = df[(df[lat_col] <= latN) & (df[lat_col] >= latS)]
    if lonW <= lonE:
        df = df[(df[lon_col] >= lonW) & (df[lon_col] <= lonE)]
    else:
        # wrap across anti-meridian: (lon ≥ W) or (lon ≤ E)
        df = df[(df[lon_col] >= lonW) | (df[lon_col] <= lonE)]
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ---------------- loaders ----------------

def load_alerts(path, time_col="time", lat_col="lat", lon_col="lon",
                prob_col=None, time_offset_h=0.0,
                normalize_lon="none", area=None):
    df = read_any(path)
    # choose probability column
    if prob_col is None:
        for cand in ["prob", "p", "p_pregen", "probability"]:
            if cand in df.columns:
                prob_col = cand; break
    if prob_col is None or prob_col not in df.columns:
        raise ValueError(f"{path}: could not find a probability column (looked for 'prob', 'p', 'p_pregen', 'probability').")

    t = _to_utc_naive(df[time_col])
    if time_offset_h:
        t = t + pd.to_timedelta(time_offset_h, unit="h")

    out = pd.DataFrame({
        "time": t,
        "lat": _norm_lon(df[lat_col], "none" ),  # lat stays numeric
        "lon": _norm_lon(df[lon_col], normalize_lon),
        "prob": pd.to_numeric(df[prob_col], errors="coerce"),
    }).dropna(subset=["time","lat","lon","prob"]).reset_index(drop=True)

    # AOI (optional; off by default)
    out = _crop_df(out, area, "lat", "lon")

    out["time_h"] = out["time"].dt.floor("h")

    # Deduplicate defensively: max prob per (hour, lat, lon)
    out = (
        out.groupby(["time_h","lat","lon"], as_index=False, sort=False)["prob"]
           .max()
           .assign(time=lambda d: d["time_h"])  # representative time at hour
    )
    return out

def load_tracks(path, time_offset_h=0.0, normalize_lon="none", area=None):
    tr = read_any(path)
    time_col = "obs_time" if "obs_time" in tr.columns else ("time" if "time" in tr.columns else None)
    if time_col is None:
        raise ValueError(f"{path}: need an 'obs_time' (or 'time') column")
    for req in ["lat","lon","vmax","pmin"]:
        if req not in tr.columns:
            raise ValueError(f"{path}: missing column '{req}'")

    t = _to_utc_naive(tr[time_col])
    if time_offset_h:
        t = t + pd.to_timedelta(time_offset_h, unit="h")

    name = tr["name"] if "name" in tr.columns else ""
    out = pd.DataFrame({
        "obs_time": t,
        "lat": pd.to_numeric(tr["lat"], errors="coerce"),
        "lon": _norm_lon(tr["lon"], normalize_lon),
        "vmax": pd.to_numeric(tr["vmax"], errors="coerce"),
        "pmin": pd.to_numeric(tr["pmin"], errors="coerce"),
        "name": name,
    }).dropna(subset=["obs_time","lat","lon","vmax","pmin"]).reset_index(drop=True)

    # AOI (optional; off by default)
    out = _crop_df(out, area, "lat", "lon")
    return out

# ---------------- matching + stats ----------------

def match_intensity(alerts_df, tracks_df, radius_deg=0.75, time_tol_h=1.0, agg="max"):
    out = []
    r2 = radius_deg**2

    by_hour = {t: g for t, g in alerts_df.groupby("time_h", sort=False)}

    for _, row in tracks_df.iterrows():
        t0 = row["obs_time"]; lat0 = row["lat"]; lon0 = row["lon"]
        # hour grid around t0
        hwin = range(-int(np.floor(time_tol_h)), int(np.ceil(time_tol_h))+1)
        candidates = []
        for dh in hwin:
            tt = (t0 + timedelta(hours=dh)).floor("h")
            g = by_hour.get(tt)
            if g is None or g.empty:
                continue
            # fast bounding box then radius filter
            box = g.loc[(g["lat"] >= lat0 - radius_deg) & (g["lat"] <= lat0 + radius_deg) &
                        (g["lon"] >= lon0 - radius_deg) & (g["lon"] <= lon0 + radius_deg)]
            if box.empty:
                continue
            d2 = (box["lat"] - lat0)**2 + (box["lon"] - lon0)**2
            sel = box.loc[d2 <= r2].copy()
            if sel.empty:
                continue
            sel["time"] = tt
            sel["dt_hours"] = (sel["time"] - t0) / np.timedelta64(1, "h")
            sel["d_km"] = haversine_km(lat0, lon0, sel["lat"], sel["lon"])
            candidates.append(sel)

        if not candidates:
            continue
        cand = pd.concat(candidates, ignore_index=True)

        if agg == "mean":
            best_prob = cand["prob"].mean()
            rsel = cand.iloc[[cand["d_km"].idxmin()]].copy()
            rsel.loc[:, "prob"] = best_prob
            best = rsel.iloc[0]
        else:  # max
            best = cand.loc[cand["prob"].idxmax()]

        out.append({
            "obs_time": t0,
            "match_time": best["time"],
            "lat": lat0, "lon": lon0,
            "vmax": row["vmax"], "pmin": row["pmin"], "name": row.get("name",""),
            "pgeom": float(best["prob"]),
            "dt_hours": float(best["dt_hours"]),
            "d_km_min": float(best["d_km"]),
            "n_cells": int(len(cand))
        })
    return pd.DataFrame(out)

def safe_corr(x, y, label, min_n=5):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    const = (n > 0 and (np.allclose(x, x.mean()) or np.allclose(y, y.mean())))
    if n < min_n or const:
        print(f"{label:14s} n={n} → too small/constant for rank corr.")
        return None
    rs, ps = spearmanr(x, y)
    try:
        rk, pk = kendalltau(x, y)
    except Exception:
        rk, pk = (np.nan, np.nan)
    print(f"{label:14s} Spearman r={rs:.3f} (p={ps:.2g})   Kendall τ={rk:.3f} (p={pk:.2g})")
    return rs, ps, rk, pk

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True, help="apply_thresholds CSV/Parquet with 'prob'")
    ap.add_argument("--tracks", required=True, help="best-track CSV/Parquet with obs_time, lat, lon, vmax, pmin[,name]")
    ap.add_argument("--radius-deg", type=float, default=0.75)
    ap.add_argument("--time-tol-hours", type=float, default=1.0)
    ap.add_argument("--prob-col", default=None, help="probability column name if not 'prob'")
    ap.add_argument("--agg", choices=["max","mean"], default="max")
    ap.add_argument("--out-csv", default="results/intensity/intensity_match.csv")
    ap.add_argument("--summary-out", default=None, help="optional text summary path")
    ap.add_argument("--scatter-png", default=None, help="optional scatter plot path")
    ap.add_argument("--track-time-offset-hours", type=float, default=0.0,
                    help="shift all track times by this many hours (e.g., -10 for local→UTC)")
    ap.add_argument("--alert-time-offset-hours", type=float, default=0.0,
                    help="shift all alert times by this many hours")
    ap.add_argument("--min-samples", type=int, default=5, help="min matches required to report correlations")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180",
                    help="Normalize longitudes for BOTH datasets (default: -180..180).")
    ap.add_argument("--area", default=None,
                    help='Optional crop "latN,lonW,latS,lonE" after lon normalization (default: none/global).')
    ap.add_argument("--debug", action="store_true", help="print time windows and overlap diagnostics")
    args = ap.parse_args()

    alerts = load_alerts(args.alerts,
                         prob_col=args.prob_col,
                         time_offset_h=args.alert_time_offset_hours,
                         normalize_lon=args.normalize_lon,
                         area=_parse_area(args.area))
    tracks = load_tracks(args.tracks,
                         time_offset_h=args.track_time_offset_hours,
                         normalize_lon=args.normalize_lon,
                         area=_parse_area(args.area))

    if args.debug:
        def _rng(s):
            return (s.min(), s.max(), s.nunique())
        a_lo, a_hi, a_nh = _rng(alerts["time_h"])
        t_lo, t_hi, t_nh = _rng(tracks["obs_time"].dt.floor("h"))
        print(f"[debug] alerts:  {a_lo}  →  {a_hi}  ({a_nh} hours)")
        print(f"[debug] tracks:  {t_lo}  →  {t_hi}  ({t_nh} hours)")
        overlap = sorted(set(alerts["time_h"].unique()).intersection(set(tracks["obs_time"].dt.floor("h").unique())))
        print(f"[debug] hour-overlap: {len(overlap)} hours")
        if not overlap:
            print("[debug] No hour overlap — try time offsets and/or larger time tolerance")

    matches = match_intensity(alerts, tracks,
                              radius_deg=args.radius_deg,
                              time_tol_h=args.time_tol_hours,
                              agg=args.agg)

    if matches.empty:
        print("[intensity] No matches found. Use --debug to inspect time overlap; "
              "try larger --radius-deg (e.g., 1.0) and --time-tol-hours (e.g., 4), "
              "or apply a time offset with --track-time-offset-hours.")
        return

    matches = matches.replace([np.inf,-np.inf], np.nan).dropna(subset=["pgeom","vmax","pmin"])

    print(f"[intensity] matched rows: {len(matches)} "
          f"(median |dt_hours|={matches['dt_hours'].abs().median():.2f}, "
          f"median d_km={matches['d_km_min'].median():.1f})")

    # rank correlations
    if len(matches) >= args.min_samples:
        safe_corr(matches["pgeom"], matches["vmax"], "vmax ~ pgeom", min_n=args.min_samples)
        safe_corr(matches["pgeom"], -matches["pmin"], "-pmin ~ pgeom", min_n=args.min_samples)
    else:
        print(f"[intensity] Too few samples for correlations (n={len(matches)} < {args.min_samples}).")

    # power-law fit if enough positive pgeom
    eps = 1e-6
    m = (matches["pgeom"] > 0) & np.isfinite(matches["vmax"])
    if m.sum() >= 3:
        b, a_log = np.polyfit(np.log(matches.loc[m,"pgeom"]+eps),
                              np.log(matches.loc[m,"vmax"]), 1)
        a = float(np.exp(a_log))
        print(f"vmax ≈ {a:.2f} · pgeom^{b:.2f}   (n={m.sum()})")
    else:
        print("power-law fit: insufficient positive samples.")

    # optional scatter plot
    if args.scatter_png:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.scatter(matches["pgeom"], matches["vmax"], s=14, alpha=0.7)
            plt.xlabel("p_geom (prob)"); plt.ylabel("v_max")
            plt.title("Intensity vs geometric probability")
            plt.savefig(args.scatter_png, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"[intensity] wrote {args.scatter_png}")
        except Exception as e:
            print(f"[intensity] scatter plot skipped: {e}")

    write_any(args.out_csv, matches)
    print(f"[intensity] wrote {args.out_csv}")

if __name__ == "__main__":
    main()