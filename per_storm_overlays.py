#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_storm_overlays.py — per-storm overlays: IBTrACS tracks + seed starts.

- Robust column detection (case-insensitive; many synonyms).
- Optional explicit overrides for seed columns.
- Continuity merge for fragmented IBTrACS tracks (time+space).
"""

import argparse, os, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------- helpers ----------------

def to_time_utcnaive(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def norm_lon(x, mode):
    x = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def parse_area(s):
    if not s: return None
    latN, lonW, latS, lonE = [float(v.strip()) for v in s.split(",")]
    return latN, lonW, latS, lonE

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# canonicalize column names for fuzzy lookup
def canon(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

def pick_col_fuzzy(df, candidates):
    # candidates is a list of strings (canonical target names)
    lc_map = {canon(c): c for c in df.columns}
    for want in candidates:
        for k, orig in lc_map.items():
            if k == want:
                return orig
    # try “contains” match
    for want in candidates:
        for k, orig in lc_map.items():
            if want in k:
                return orig
    return None

# ---------------- loaders ----------------

# IBTrACS candidates (many variants)
IB_NAME = ["name","namewmo","namejtwc","stormname"]
IB_TIME = ["isotime","time","datetime","date_time","isotime"]
IB_LAT  = ["lat","latitude","usalat","wmolat"]
IB_LON  = ["lon","longitude","usalon","wmolon"]
IB_VMAX = ["wmowind","usawind","vmax","maxwind","wind"]

def load_ibtracs(path, normalize_lon_mode, start=None, end=None, time_offset_h=0.0, area=None):
    df = pd.read_csv(path, low_memory=False)
    cols = [canon(c) for c in df.columns]

    name_col = pick_col_fuzzy(df, IB_NAME); 
    t_col    = pick_col_fuzzy(df, IB_TIME)
    lat_col  = pick_col_fuzzy(df, IB_LAT)
    lon_col  = pick_col_fuzzy(df, IB_LON)
    vmax_col = pick_col_fuzzy(df, IB_VMAX)

    if not (name_col and t_col and lat_col and lon_col):
        raise ValueError(f"{path}: missing one of NAME/time/lat/lon columns.")

    t = to_time_utcnaive(df[t_col])
    if time_offset_h:
        t = t + pd.to_timedelta(time_offset_h, unit="h")

    out = pd.DataFrame({
        "name": df[name_col].astype(str).str.strip().str.upper(),
        "time": t,
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": norm_lon(df[lon_col], normalize_lon_mode),
        "vmax": pd.to_numeric(df[vmax_col], errors="coerce") if vmax_col else np.nan,
    }).dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    if start: out = out.loc[out["time"] >= pd.to_datetime(start)].reset_index(drop=True)
    if end:   out = out.loc[out["time"] <= pd.to_datetime(end)].reset_index(drop=True)

    if area:
        latN, lonW, latS, lonE = area
        out = out.loc[(out["lat"] <= latN) & (out["lat"] >= latS) &
                      (out["lon"] >= lonW) & (out["lon"] <= lonE)].reset_index(drop=True)

    return out

# seed candidates (very permissive)
SEED_LAT_CANDS  = ["seedlat","startlat","lat","y","seed_lat","start_lat"]
SEED_LON_CANDS  = ["seedlon","startlon","lon","x","seed_lon","start_lon"]
SEED_TIME_CANDS = ["seedstart","seedtime","starttime","time","start","start_time","seed_start","seed_start_time"]
SEED_NAME_CANDS = ["name","stormname","seedname"]

def load_seeds(path, normalize_lon_mode, override_lat=None, override_lon=None, override_time=None):
    m = pd.read_csv(path, low_memory=False)

    lat_col  = override_lat  or pick_col_fuzzy(m, [canon(c) for c in SEED_LAT_CANDS])
    lon_col  = override_lon  or pick_col_fuzzy(m, [canon(c) for c in SEED_LON_CANDS])
    time_col = override_time or pick_col_fuzzy(m, [canon(c) for c in SEED_TIME_CANDS])
    name_col = pick_col_fuzzy(m, [canon(c) for c in SEED_NAME_CANDS])

    if not (lat_col and lon_col and time_col):
        raise ValueError(
            f"{path}: couldn’t find seed lat/lon/time columns.\n"
            f"  Detected columns: {list(m.columns)[:10]}...\n"
            f"  Try --seed-lat-col / --seed-lon-col / --seed-time-col."
        )

    print(f"[seeds] using columns lat={lat_col} lon={lon_col} time={time_col}" +
          (f" name={name_col}" if name_col else ""))

    seeds = pd.DataFrame({
        "seed_lat": pd.to_numeric(m[lat_col], errors="coerce"),
        "seed_lon": norm_lon(m[lon_col], normalize_lon_mode),
        "seed_time": to_time_utcnaive(m[time_col]),
        "name": (m[name_col].astype(str).str.strip().str.upper() if name_col else None)
    }).dropna(subset=["seed_lat","seed_lon","seed_time"]).reset_index(drop=True)

    return seeds

# ---------------- continuity merge ----------------

def merge_track_fragments(df_storm, gap_hours=48, max_km=500):
    if df_storm.empty:
        return df_storm
    srt = df_storm.sort_values("time").reset_index(drop=True)
    gaps = srt["time"].diff().dt.total_seconds().div(3600).fillna(0)
    frag_id = (gaps > gap_hours).cumsum()
    frags = [g.copy().reset_index(drop=True) for _, g in srt.groupby(frag_id, sort=True)]
    if len(frags) <= 1:
        return srt
    merged = frags[0]
    for nxt in frags[1:]:
        lat1, lon1 = merged.iloc[-1][["lat","lon"]]
        lat2, lon2 = nxt.iloc[0][["lat","lon"]]
        d = haversine_km(lat1, lon1, lat2, lon2)
        if d <= max_km:
            merged = pd.concat([merged, nxt], ignore_index=True)
        else:
            nanrow = pd.DataFrame({"time":[np.nan], "lat":[np.nan], "lon":[np.nan], "vmax":[np.nan], "name":[merged.iloc[0]['name']]})
            merged = pd.concat([merged, nanrow, nxt], ignore_index=True)
    return merged

def merge_all_tracks(tracks, gap_hours=48, max_km=500, min_points=3):
    out = []
    for nm, g in tracks.groupby("name", sort=False):
        mg = merge_track_fragments(g, gap_hours=gap_hours, max_km=max_km)
        if mg["time"].notna().sum() >= min_points:
            out.append(mg)
    if not out:
        return tracks.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)

# ---------------- plotting ----------------

def plot_one(storm_name, track_df, seeds_df, out_png, dpi=160, title=None):
    lat_all = pd.concat([track_df["lat"].dropna(), seeds_df["seed_lat"].dropna()], ignore_index=True)
    lon_all = pd.concat([track_df["lon"].dropna(), seeds_df["seed_lon"].dropna()], ignore_index=True)
    if len(lat_all)==0 or len(lon_all)==0:
        return False
    lat_pad = max(1.0, 0.1*(lat_all.max()-lat_all.min()+1e-6))
    lon_pad = max(1.0, 0.1*(lon_all.max()-lon_all.min()+1e-6))
    extent = [lon_all.min()-lon_pad, lon_all.max()+lon_pad,
              lat_all.min()-lat_pad, lat_all.max()+lat_pad]

    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#e5e5e5", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle='--')

    td = track_df.sort_values("time")
    if "vmax" in td.columns and td["vmax"].notna().any():
        vmax = td["vmax"].to_numpy(); lat = td["lat"].to_numpy(); lon = td["lon"].to_numpy()
        bins = [-np.inf, 34, 50, 64, 83, 96, 113, np.inf]
        colors = ["#4c78a8","#72b7b2","#54a24b","#f58518","#e45756","#723c7b","#000000"]
        for i in range(len(vmax)-1):
            if any(np.isnan([lat[i],lon[i],lat[i+1],lon[i+1]])): continue
            b = np.digitize(vmax[i], bins)-1
            ax.plot([lon[i],lon[i+1]],[lat[i],lat[i+1]],
                    color=colors[max(0,min(b,len(colors)-1))],
                    linewidth=2.2, transform=ccrs.PlateCarree(), zorder=2)
    else:
        ax.plot(td["lon"], td["lat"], color="#1f77b4", linewidth=2.2,
                transform=ccrs.PlateCarree(), zorder=2)

    td_valid = td.dropna(subset=["time","lat","lon"])
    if not td_valid.empty:
        ax.scatter([td_valid.iloc[0]["lon"]],[td_valid.iloc[0]["lat"]],
                   s=70, marker="^", color="green", edgecolor="k", zorder=3,
                   transform=ccrs.PlateCarree(), label="Storm start")
        ax.scatter([td_valid.iloc[-1]["lon"]],[td_valid.iloc[-1]["lat"]],
                   s=70, marker="v", color="red", edgecolor="k", zorder=3,
                   transform=ccrs.PlateCarree(), label="Storm end")

    if not seeds_df.empty:
        ax.scatter(seeds_df["seed_lon"], seeds_df["seed_lat"], s=24,
                   color="#f5a623", edgecolor="k", linewidths=0.3, alpha=0.85,
                   zorder=3, transform=ccrs.PlateCarree(), label="Seed starts")

    ax.set_title(title or f"Seed–Track Overlay — {storm_name}", fontsize=16)
    ax.legend(loc="lower left")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Per-storm overlays: IBTrACS + seeds.")
    ap.add_argument("--matches", help="seed_track_matches.csv or seed_starts_points.csv")
    ap.add_argument("--ibtracs", required=True, help="IBTrACS CSV (ibtracs.ALL.list.v04r01.csv)")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--area", default=None, help='"latN,lonW,latS,lonE"')
    ap.add_argument("--time-offset-hours", type=float, default=0.0)
    ap.add_argument("--out-dir", default="results/per_storm"); ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--merge-gap-hours", type=float, default=48.0)
    ap.add_argument("--merge-max-km", type=float, default=500.0)
    ap.add_argument("--min-points", type=int, default=3)
    # NEW: explicit seed column overrides
    ap.add_argument("--seed-lat-col", default=None)
    ap.add_argument("--seed-lon-col", default=None)
    ap.add_argument("--seed-time-col", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    area = parse_area(args.area) if args.area else None

    tracks = load_ibtracs(args.ibtracs, args.normalize_lon, args.start, args.end,
                          args.time_offset_hours, area)
    if tracks.empty:
        raise SystemExit("No IBTrACS rows after filters.")

    if args.matches:
        seeds = load_seeds(args.matches, args.normalize_lon,
                           override_lat=args.seed_lat_col,
                           override_lon=args.seed_lon_col,
                           override_time=args.seed_time_col)
    else:
        seeds = pd.DataFrame(columns=["seed_lat","seed_lon","seed_time","name"])

    tracks_m = merge_all_tracks(tracks,
                                gap_hours=float(args.merge_gap_hours),
                                max_km=float(args.merge_max_km),
                                min_points=int(args.min_points))

    storms = sorted(tracks_m["name"].dropna().unique().tolist())
    wrote = 0
    for nm in storms:
        tdf = tracks_m.loc[tracks_m["name"] == nm].copy()
        # if seeds have names, filter; else plot all seeds in view
        if "name" in seeds.columns and seeds["name"].notna().any():
            sdf = seeds.loc[seeds["name"] == nm].copy()
        else:
            sdf = seeds.copy()
        out_png = out_dir / f"{nm.replace('/','-').replace(' ','_')}.png"
        if plot_one(nm, tdf, sdf, str(out_png), dpi=args.dpi):
            wrote += 1

    print(f"[per-storm] Wrote {wrote} figure(s) to {out_dir}")

if __name__ == "__main__":
    main()