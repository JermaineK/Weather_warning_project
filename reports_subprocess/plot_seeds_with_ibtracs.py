#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_seeds_with_ibtracs.py
Overlay IBTrACS storm tracks over seed points (or matched patches).

Inputs (any of these):
  --seeds <parquet/csv>            # e.g., data/seed_cells_H72.parquet
  --matches <csv>                  # optional: seed_starts_vs_tracks output (to plot only starts)

Storms:
  --ibtracs <csv>                  # ibtracs.ALL.list.v04r01.csv

Filtering / options:
  --start YYYY-MM-DD
  --end   YYYY-MM-DD
  --area "latN,lonW,latS,lonE"     # after lon normalization (supports anti-meridian)
  --normalize-lon {none,-180..180,0..360}
  --time-offset-hours <float>      # apply to IBTrACS (e.g. -10 if file is local time)
  --out-png <path>                 # default: results/maps/seeds_ibtracs.png
  --storms-out <path>              # optional CSV of storms included
  --title <str>
  --seed-alpha <0..1>              # default 0.8
  --seed-size <float>              # default 16
  --dpi <int>                      # default 180

Notes:
- If both --seeds and --matches are given, seeds are drawn from --matches (columns seed_lat/seed_lon if present else lat/lon).
- If cartopy is missing, falls back to plain axes (no coastlines).
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- I/O helpers ----------

def read_any(p, **kw):
    p = str(p)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith((".parquet",".pq",".pqt")):
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, low_memory=False, **kw)

def to_utc_naive(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def norm_lon(x, mode):
    x = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def parse_area(aoi):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(u.strip()) for u in aoi.split(",")]
    return latN, lonW, latS, lonE

def crop_aoi(df, aoi, lat_col="lat", lon_col="lon"):
    if not aoi:
        return df
    latN, lonW, latS, lonE = aoi
    df = df[(df[lat_col] <= latN) & (df[lat_col] >= latS)]
    if lonW <= lonE:
        return df[(df[lon_col] >= lonW) & (df[lon_col] <= lonE)]
    # anti-meridian wrap: (lon ≥ W) or (lon ≤ E)
    return df[(df[lon_col] >= lonW) | (df[lon_col] <= lonE)]

# ---------- plotting backend detection ----------

def _have_cartopy():
    try:
        import cartopy.crs as ccrs  # noqa
        import cartopy.feature as cfeature  # noqa
        return True
    except Exception:
        return False

# categorical color by vmax (kt)
def vmax_color(v):
    # Beaufort-ish / Saffir-Simpson-ish thresholds (kt)
    if not np.isfinite(v):
        return "#888888"
    if v < 34:   return "#9ecae1"  # TD
    if v < 50:   return "#3182bd"  # TS
    if v < 64:   return "#31a354"  # STS
    if v < 83:   return "#fd8d3c"  # Cat1-2
    if v < 96:   return "#e6550d"  # Cat3
    if v < 113:  return "#d62728"  # Cat4
    return "#8c2d04"               # Cat5+

def pick_ci(df, names):
    """Case-insensitive single-column pick; returns actual column name or None."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def build_time_from_parts(df, parts):
    try:
        y = df[parts[0]].astype(int)
        m = df[parts[1]].astype(int)
        d = df[parts[2]].astype(int)
        h = df[parts[3]].astype(int)
        return pd.to_datetime(
            pd.DataFrame({"Y": y, "M": m, "D": d, "h": h}).astype(str).agg("-".join, axis=1) + ":00",
            utc=True, errors="coerce"
        ).dt.tz_localize(None)
    except Exception:
        return pd.NaT

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Overlay IBTrACS storm tracks over seeds.")
    ap.add_argument("--seeds", default=None, help="Seeds file (parquet/csv)")
    ap.add_argument("--matches", default=None, help="Optional matched_patches.csv to plot seed starts")
    ap.add_argument("--ibtracs", required=True, help="ibtracs.ALL.list.v04r01.csv")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE (after lon normalization; anti-meridian ok)")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--time-offset-hours", type=float, default=0.0, help="shift IBTrACS times")
    ap.add_argument("--out-png", default="results/maps/seeds_ibtracs.png")
    ap.add_argument("--storms-out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--seed-alpha", type=float, default=0.8)
    ap.add_argument("--seed-size", type=float, default=16.0)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)

    # ---- load seeds (from matches preferred)
    seeds = None
    seed_label = "Seeds"
    if args.matches:
        m = read_any(args.matches)
        latc = pick_ci(m, ["seed_lat","lat"])
        lonc = pick_ci(m, ["seed_lon","lon"])
        tc   = pick_ci(m, ["seed_time","time","time_h"])
        if not (latc and lonc):
            raise ValueError(f"{args.matches}: need columns seed_lat/seed_lon or lat/lon")
        seeds = pd.DataFrame({
            "time": to_utc_naive(m[tc]) if tc else pd.NaT,
            "lat": pd.to_numeric(m[latc], errors="coerce"),
            "lon": pd.to_numeric(m[lonc], errors="coerce"),
        }).dropna(subset=["lat","lon"]).reset_index(drop=True)
        seed_label = "Seed starts"
    elif args.seeds:
        s = read_any(args.seeds)
        tcol = pick_ci(s, ["time","time_h"])
        seeds = pd.DataFrame({
            "time": to_utc_naive(s[tcol]) if tcol else pd.NaT,
            "lat": pd.to_numeric(s[pick_ci(s, ["lat"])], errors="coerce"),
            "lon": pd.to_numeric(s[pick_ci(s, ["lon"])], errors="coerce"),
        }).dropna(subset=["lat","lon"]).reset_index(drop=True)

    # normalize lon (seeds)
    if seeds is not None and len(seeds):
        seeds["lon"] = norm_lon(seeds["lon"], args.normalize_lon)

    # ---- IBTrACS
    ib = read_any(args.ibtracs)

    # Time column detection (ISO -> parts -> generic)
    tcol = pick_ci(ib, ["iso_time","time","datetime","date_time","obs_time"])
    if tcol is None:
        # try common parts
        built = None
        for parts in (["season","month","day","hour"], ["year","month","day","hour"],
                      ["Year","Month","Day","Hour"]):
            if all(p in ib.columns for p in parts):
                built = build_time_from_parts(ib, parts)
                break
        if built is None:
            # as last resort, try first column parsing
            built = to_utc_naive(ib.iloc[:,0])
        ib["_t_"] = built
    else:
        ib["_t_"] = to_utc_naive(ib[tcol])

    if args.time_offset_hours:
        ib["_t_"] = ib["_t_"] + pd.to_timedelta(args.time_offset_hours, unit="h")

    # Core columns
    latc = pick_ci(ib, ["latitude","lat","LAT"])
    lonc = pick_ci(ib, ["longitude","lon","LON"])
    vmaxc= pick_ci(ib, ["wmo_wind","usa_wind","wind_wmo","wind","vmax","USA_WIND"])
    namec= pick_ci(ib, ["name","NAME","storm_name","stormname"])
    idc  = pick_ci(ib, ["sid","usa_atcf_id","identifier","serial_num","storm_id","ID","num","NUMBER"])

    ib["lat"] = pd.to_numeric(ib[latc], errors="coerce")
    ib["lon"] = norm_lon(pd.to_numeric(ib[lonc], errors="coerce"), args.normalize_lon)
    ib["vmax"] = pd.to_numeric(ib[vmaxc], errors="coerce") if vmaxc else np.nan
    if idc is not None:
        ib["_id_"] = ib[idc].astype(str)
    else:
        # fall back to name+time chunk id
        nm = ib[namec].astype(str) if namec else "storm"
        ib["_id_"] = nm

    ib = ib.dropna(subset=["lat","lon","_t_"]).reset_index(drop=True)

    # ---- filters
    tmin = None
    tmax = None
    if seeds is not None and "time" in seeds and seeds["time"].notna().any():
        tmin = seeds["time"].min()
        tmax = seeds["time"].max()
    if args.start:
        tmin = to_utc_naive(pd.Series([args.start])).iloc[0]
    if args.end:
        tmax = to_utc_naive(pd.Series([args.end])).iloc[0]
    if tmin is not None:
        ib = ib.loc[ib["_t_"] >= tmin]
    if tmax is not None:
        ib = ib.loc[ib["_t_"] <= tmax]

    aoi = parse_area(args.area)
    if aoi:
        ib = crop_aoi(ib, aoi, "lat", "lon")
        if seeds is not None and len(seeds):
            seeds = crop_aoi(seeds, aoi, "lat", "lon")

    storms = sorted(ib["_id_"].dropna().unique().tolist())
    if args.storms_out:
        Path(args.storms_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"storm_id": storms}).to_csv(args.storms_out, index=False)

    # ---- plot
    have_ct = _have_cartopy()
    if have_ct:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#dddddd", edgecolor="none", zorder=0)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
        # extent
        if aoi:
            ax.set_extent([aoi[1], aoi[3], aoi[2], aoi[0]], crs=proj)
        else:
            lonmin = np.nanmin([seeds["lon"].min() if seeds is not None and len(seeds) else np.nan, ib["lon"].min()])
            lonmax = np.nanmax([seeds["lon"].max() if seeds is not None and len(seeds) else np.nan, ib["lon"].max()])
            latmin = np.nanmin([seeds["lat"].min() if seeds is not None and len(seeds) else np.nan, ib["lat"].min()])
            latmax = np.nanmax([seeds["lat"].max() if seeds is not None and len(seeds) else np.nan, ib["lat"].max()])
            padx = max(1.0, (lonmax - lonmin) * 0.05 if np.isfinite(lonmax - lonmin) else 5.0)
            pady = max(1.0, (latmax - latmin) * 0.05 if np.isfinite(latmax - latmin) else 5.0)
            ax.set_extent([lonmin-padx, lonmax+padx, latmin-pady, latmax+pady], crs=proj)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, ls=":", alpha=0.4)

    # plot storm tracks
    for stid, g in ib.groupby("_id_", sort=False):
        g = g.sort_values("_t_")
        xs = g["lon"].to_numpy(); ys = g["lat"].to_numpy(); vs = g["vmax"].to_numpy()
        for i in range(max(0, len(g)-1)):
            c = vmax_color((vs[i]+vs[i+1])/2 if i+1 < len(vs) else vs[i])
            if have_ct:
                ax.plot([xs[i], xs[i+1]],[ys[i], ys[i+1]],
                        transform=ccrs.PlateCarree(), color=c, lw=2, alpha=0.9, zorder=2)
            else:
                ax.plot([xs[i], xs[i+1]],[ys[i], ys[i+1]], color=c, lw=2, alpha=0.9, zorder=2)
        sz = np.clip((g["vmax"].fillna(20)/10.0)**2, 6, 80)
        if have_ct:
            ax.scatter(xs, ys, s=sz, c=[vmax_color(v) for v in vs],
                       transform=ccrs.PlateCarree(), edgecolor="k", linewidths=0.2, alpha=0.9, zorder=3)
        else:
            ax.scatter(xs, ys, s=sz, c=[vmax_color(v) for v in vs],
                       edgecolor="k", linewidths=0.2, alpha=0.9, zorder=3)

    # seeds layer
    if seeds is not None and len(seeds):
        if have_ct:
            ax.scatter(seeds["lon"], seeds["lat"], s=args.seed_size,
                       transform=ccrs.PlateCarree(),
                       color="#1f77b4", alpha=args.seed_alpha, label=seed_label, zorder=4)
        else:
            ax.scatter(seeds["lon"], seeds["lat"], s=args.seed_size,
                       color="#1f77b4", alpha=args.seed_alpha, label=seed_label, zorder=4)

    # legend for vmax colors (proxy; discrete)
    import matplotlib.lines as mlines
    legend_elems = [
        mlines.Line2D([],[], color=vmax_color(20), lw=3, label="<34 kt"),
        mlines.Line2D([],[], color=vmax_color(40), lw=3, label="34–49 kt"),
        mlines.Line2D([],[], color=vmax_color(55), lw=3, label="50–63 kt"),
        mlines.Line2D([],[], color=vmax_color(70), lw=3, label="64–82 kt"),
        mlines.Line2D([],[], color=vmax_color(90), lw=3, label="83–95 kt"),
        mlines.Line2D([],[], color=vmax_color(105),lw=3, label="96–112 kt"),
        mlines.Line2D([],[], color=vmax_color(120),lw=3, label="≥113 kt"),
    ]
    if seeds is not None and len(seeds):
        legend_elems.insert(0, mlines.Line2D([],[], marker='o', lw=0, color="#1f77b4",
                          label=seed_label, markersize=6))

    ttl = args.title or "Seed–Track Matches with IBTrACS"
    if have_ct:
        ax.set_title(ttl)
        ax.legend(handles=legend_elems, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    else:
        plt.title(ttl)
        plt.legend(handles=legend_elems, loc="lower left", fontsize=8, ncol=2, framealpha=0.9)

    fig.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    storms_drawn = len(storms)
    print(f"[map] wrote {args.out_png} | storms drawn: {storms_drawn} | "
          f"seeds: {0 if seeds is None else len(seeds)}")

if __name__ == "__main__":
    main()