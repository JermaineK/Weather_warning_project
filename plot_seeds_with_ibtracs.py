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
  --area "latN,lonW,latS,lonE"     # after lon normalization
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

import argparse, os, math
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

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Overlay IBTrACS storm tracks over seeds.")
    ap.add_argument("--seeds", default=None, help="Seeds file (parquet/csv)")
    ap.add_argument("--matches", default=None, help="Optional matched_patches.csv to plot seed starts")
    ap.add_argument("--ibtracs", required=True, help="ibtracs.ALL.list.v04r01.csv")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE")
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
        # try seed_lat/lon from matches; fallback to lat/lon
        latc = "seed_lat" if "seed_lat" in m.columns else ("lat" if "lat" in m.columns else None)
        lonc = "seed_lon" if "seed_lon" in m.columns else ("lon" if "lon" in m.columns else None)
        tc   = "seed_time" if "seed_time" in m.columns else ("time" if "time" in m.columns else None)
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
        # heuristic time column
        tcol = "time" if "time" in s.columns else ("time_h" if "time_h" in s.columns else None)
        if tcol is None:  # accept no time
            t = pd.NaT
        else:
            t = to_utc_naive(s[tcol])
        seeds = pd.DataFrame({
            "time": t,
            "lat": pd.to_numeric(s["lat"], errors="coerce"),
            "lon": pd.to_numeric(s["lon"], errors="coerce"),
        }).dropna(subset=["lat","lon"]).reset_index(drop=True)
    else:
        print("[warn] no seeds/matches given; plotting tracks only")

    # normalize lon (seeds)
    if seeds is not None:
        seeds["lon"] = norm_lon(seeds["lon"], args.normalize_lon)

    # time span from seeds to filter storms reasonably
    tmin = seeds["time"].min() if (seeds is not None and "time" in seeds) else None
    tmax = seeds["time"].max() if (seeds is not None and "time" in seeds) else None
    if args.start:
        tmin = to_utc_naive(pd.Series([args.start])).iloc[0]
    if args.end:
        tmax = to_utc_naive(pd.Series([args.end])).iloc[0]

    # ---- load ibtracs
    ib = read_any(args.ibtracs)
    # pick time column (varies by file); try ISO time or combine year/mon/day/hr
    tcol = None
    for c in ["iso_time","time","datetime","date_time","ObsTime","obs_time"]:
        if c in ib.columns:
            tcol = c; break
    if tcol is None and {"season","number","month","day","hour"}.issubset(ib.columns):
        # not common for v04r01 list; fallback handled below
        pass

    if tcol is None:
        # Build from year/month/day/hour columns if available
        poss = [("season","number","month","day","hour"), ("Year","Month","Day","Hour")]
        built = None
        for tpl in poss:
            if set(tpl).issubset(ib.columns):
                if len(tpl)==5:
                    y = ib[tpl[0]]; m=ib[tpl[2]]; d=ib[tpl[3]]; h=ib[tpl[4]]
                else:
                    y=ib["Year"]; m=ib["Month"]; d=ib["Day"]; h=ib["Hour"]
                try:
                    built = pd.to_datetime(
                        pd.DataFrame({"Y":y,"M":m,"D":d,"h":h}).astype(int).astype(str).agg("-".join, axis=1)
                        + ":00", utc=True, errors="coerce"
                    ).dt.tz_localize(None)
                except Exception:
                    built = None
        if built is None:
            # last resort: try to_datetime on a likely column
            guess = ib.iloc[:,0]
            built = to_utc_naive(guess)
        ib["_t_"] = built
    else:
        ib["_t_"] = to_utc_naive(ib[tcol])

    if args.time_offset_hours:
        ib["_t_"] = ib["_t_"] + pd.to_timedelta(args.time_offset_hours, unit="h")

    # lat/lon/vmax columns (IBTrACS common names)
    latc = "latitude"  if "latitude"  in ib.columns else ("lat" if "lat" in ib.columns else "LAT")
    lonc = "longitude" if "longitude" in ib.columns else ("lon" if "lon" in ib.columns else "LON")
    vmaxc= "wind_wmo"  if "wind_wmo"  in ib.columns else ("USA_WIND" if "USA_WIND" in ib.columns else "vmax")

    ib["lat"] = pd.to_numeric(ib[latc], errors="coerce")
    ib["lon"] = norm_lon(pd.to_numeric(ib[lonc], errors="coerce"), args.normalize_lon)
    ib["vmax"] = pd.to_numeric(ib[vmaxc], errors="coerce")

    # storm id (varies); try SID first else join of name+season+num
    sid = None
    for c in ["sid","SID","usa_atcf_id","identifier","serial_num","storm_id","ID","num","NUMBER"]:
        if c in ib.columns:
            sid = c; break
    if sid is None:
        namec = "name" if "name" in ib.columns else ("NAME" if "NAME" in ib.columns else None)
        seas  = "season" if "season" in ib.columns else ("SEASON" if "SEASON" in ib.columns else None)
        num   = "number" if "number" in ib.columns else ("NUMBER" if "NUMBER" in ib.columns else None)
        if namec and seas and num:
            ib["_id_"] = ib[namec].astype(str)+"_"+ib[seas].astype(str)+"_"+ib[num].astype(str)
        else:
            ib["_id_"] = "storm"
        sid = "_id_"

    # time / AOI filter
    ib = ib.dropna(subset=["lat","lon","_t_"]).reset_index(drop=True)
    if tmin is not None:
        ib = ib.loc[ib["_t_"] >= tmin]
    if tmax is not None:
        ib = ib.loc[ib["_t_"] <= tmax]
    aoi = parse_area(args.area)
    if aoi:
        latN, lonW, latS, lonE = aoi
        ib = ib.loc[(ib["lat"] <= latN) & (ib["lat"] >= latS) &
                    (ib["lon"] >= lonW) & (ib["lon"] <= lonE)]

    # storms included
    storms = sorted(ib[sid].dropna().unique().tolist())
    if args.storms_out:
        Path(args.storms_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ "storm_id": storms }).to_csv(args.storms_out, index=False)

    # ---- plot
    have_ct = _have_cartopy()
    if have_ct:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection=proj)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.6)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#dddddd", edgecolor="none", zorder=0)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")
        # extent from seeds/AOI or storms
        if aoi:
            ax.set_extent([aoi[1], aoi[3], aoi[2], aoi[0]], crs=proj)
        else:
            lonmin = np.nanmin([seeds["lon"].min() if seeds is not None else np.nan, ib["lon"].min()])
            lonmax = np.nanmax([seeds["lon"].max() if seeds is not None else np.nan, ib["lon"].max()])
            latmin = np.nanmin([seeds["lat"].min() if seeds is not None else np.nan, ib["lat"].min()])
            latmax = np.nanmax([seeds["lat"].max() if seeds is not None else np.nan, ib["lat"].max()])
            padx = max(1.0, (lonmax - lonmin) * 0.05)
            pady = max(1.0, (latmax - latmin) * 0.05)
            ax.set_extent([lonmin-padx, lonmax+padx, latmin-pady, latmax+pady], crs=proj)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, ls=":", alpha=0.4)

    # plot storm tracks
    for stid, g in ib.groupby(sid, sort=False):
        g = g.sort_values("_t_")
        # line by segments colored by vmax
        xs = g["lon"].to_numpy(); ys = g["lat"].to_numpy(); vs = g["vmax"].to_numpy()
        for i in range(len(g)-1):
            c = vmax_color( (vs[i]+vs[i+1])/2 if i+1 < len(vs) else vs[i] )
            if have_ct:
                ax.plot([xs[i], xs[i+1]],[ys[i], ys[i+1]],
                        transform=ccrs.PlateCarree(), color=c, lw=2, alpha=0.9, zorder=2)
            else:
                ax.plot([xs[i], xs[i+1]],[ys[i], ys[i+1]], color=c, lw=2, alpha=0.9, zorder=2)
        # positions as small markers sized by wind
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
    print(f"[map] wrote {args.out_png} | storms drawn: {len(storms)}")

if __name__ == "__main__":
    main()