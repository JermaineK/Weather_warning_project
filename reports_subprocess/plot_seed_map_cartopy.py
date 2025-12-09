#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _norm_lon(x, mode):
    x = pd.to_numeric(x, errors="coerce")
    if mode == "0..360":
        return (x % 360 + 360) % 360
    if mode == "-180..180":
        return ((x + 180) % 360) - 180
    return x

def _parse_area(aoi):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(t) for t in aoi.split(",")]
    return latN, lonW, latS, lonE

def _load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception as e:
        raise SystemExit(
            "Cartopy is required for seed maps. Try:\n"
            "  pip install cartopy shapely pyproj\n\n"
            f"Import error: {e}"
        )

def main():
    ap = argparse.ArgumentParser(description="Cartopy seed map renderer")
    ap.add_argument("--seeds", required=True, help="CSV/Parquet with at least: lat, lon[, value][, time]")
    ap.add_argument("--out-png", required=True, help="Output PNG path (or prefix if --per-hour).")
    ap.add_argument("--value-col", default="prob_max", help="Optional numeric column to color/size by")
    ap.add_argument("--time-col", default=None, help="Optional time column; if provided with --per-hour, make one map/hour")
    ap.add_argument("--per-hour", action="store_true", help="Produce one PNG per hour if time is available")
    ap.add_argument("--flag-col", default=None, help="Optional flag column; if provided, filter to rows == 1")
    ap.add_argument("--min-prob", type=float, default=0.5, help="Minimum value/prob to plot (filters points)")
    ap.add_argument("--top-quantile", type=float, default=None, help="Keep only rows with value_col above this quantile (0-1)")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help='Optional AOI "latN,lonW,latS,lonE"')
    ap.add_argument("--tracks", default=None, help="Optional tracks file to time/space-filter seeds (CSV/Parquet).")
    ap.add_argument("--storm-radius-deg", type=float, default=5.0, help="Lat/lon padding around track bbox for filtering.")
    ap.add_argument("--storm-window-before-h", type=float, default=240.0, help="Hours before track times to include seeds.")
    ap.add_argument("--storm-window-after-h", type=float, default=72.0, help="Hours after track times to include seeds.")
    ap.add_argument("--title", default=None, help="Figure title")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    # Load data
    p = str(args.seeds).lower()
    if p.endswith((".parquet",".parq",".pq")):
        df = pd.read_parquet(args.seeds)
    else:
        df = pd.read_csv(args.seeds)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("Seeds file must have 'lat' and 'lon' columns.")

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _norm_lon(df["lon"], args.normalize_lon)
    if args.flag_col and args.flag_col in df.columns:
        df = df.loc[pd.to_numeric(df[args.flag_col], errors="coerce").fillna(0) > 0]
    if args.min_prob is not None and args.value_col and args.value_col in df.columns:
        df = df.loc[pd.to_numeric(df[args.value_col], errors="coerce") >= float(args.min_prob)]
    if args.top_quantile is not None and args.value_col and args.value_col in df.columns:
        vals = pd.to_numeric(df[args.value_col], errors="coerce")
        cutoff = vals.quantile(float(args.top_quantile))
        df = df.loc[vals >= cutoff]
    # Optional filter: restrict to storms window/bbox
    if args.tracks:
        try:
            tp = str(args.tracks).lower()
            if tp.endswith((".parquet",".parq",".pq")):
                tr = pd.read_parquet(args.tracks)
            else:
                tr = pd.read_csv(args.tracks)
            if {"lat","lon","time"}.issubset(tr.columns):
                tr = tr.copy()
                tr["lat"] = pd.to_numeric(tr["lat"], errors="coerce")
                tr["lon"] = _norm_lon(tr["lon"], args.normalize_lon)
                ttime = pd.to_datetime(tr["time"], utc=True, errors="coerce").dt.tz_convert(None)
                tr = tr.assign(time=ttime).dropna(subset=["lat","lon","time"])
                if args.area:
                    latN, lonW, latS, lonE = _parse_area(args.area)
                    tr = tr.loc[(tr["lat"] <= latN) & (tr["lat"] >= latS) &
                                (tr["lon"] >= lonW) & (tr["lon"] <= lonE)]
                if not tr.empty:
                    tmin = tr["time"].min() - pd.Timedelta(hours=args.storm_window_before_h)
                    tmax = tr["time"].max() + pd.Timedelta(hours=args.storm_window_after_h)
                    lat_min = tr["lat"].min() - args.storm_radius_deg
                    lat_max = tr["lat"].max() + args.storm_radius_deg
                    lon_min = tr["lon"].min() - args.storm_radius_deg
                    lon_max = tr["lon"].max() + args.storm_radius_deg
                    if args.time_col and args.time_col in df.columns:
                        tcol = args.time_col
                    else:
                        # auto-detect time column
                        tcol = "time" if "time" in df.columns else None
                    if tcol:
                        tseeds = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert(None)
                        df = df.loc[
                            tseeds.between(tmin, tmax)
                            & df["lat"].between(lat_min, lat_max)
                            & df["lon"].between(lon_min, lon_max)
                        ]
        except Exception as e:
            print(f"[map] warning: failed to apply track filter: {e}")
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    if args.area:
        latN, lonW, latS, lonE = _parse_area(args.area)
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)

    # Value column (optional)
    val = None
    if args.value_col and args.value_col in df.columns:
        val = pd.to_numeric(df[args.value_col], errors="coerce")
    else:
        if args.value_col:
            print(f"[map] value_col '{args.value_col}' not found; ignoring.", flush=True)

    # Time column (optional)
    tcol = None
    if args.time_col and args.time_col in df.columns:
        tcol = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
        df["_time_h"] = tcol
    elif args.per_hour:
        print("[map] --per-hour given but time column missing; producing single map.", flush=True)

    # Cartopy import
    ccrs, cfeature = _load_cartopy()
    import matplotlib.pyplot as plt

    def _render(ddf, out_path, title):
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=proj)

        # Coastlines / land / borders
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f3f3f3")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, alpha=0.6)

        # Extent (AOI or data-driven)
        if args.area:
            ax.set_extent([lonW, lonE, latS, latN], crs=proj)
        else:
            pad = 5
            lat_min = float(ddf["lat"].min()) - pad
            lat_max = float(ddf["lat"].max()) + pad
            lon_min = float(ddf["lon"].min()) - pad
            lon_max = float(ddf["lon"].max()) + pad
            if np.isfinite([lat_min, lat_max, lon_min, lon_max]).all():
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

        # Scatter points
        if val is not None:
            vv = pd.to_numeric(ddf[args.value_col], errors="coerce")
            s = 6 + 24 * (vv - vv.min()) / (vv.max() - vv.min() + 1e-12)
            sc = ax.scatter(ddf["lon"], ddf["lat"], c=vv, s=s, cmap="viridis", alpha=0.85, transform=proj)
            cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
            cb.set_label(args.value_col)
        else:
            ax.scatter(ddf["lon"], ddf["lat"], s=18, alpha=0.85, transform=proj)

        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
        if title:
            ax.set_title(title, fontsize=12)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[map] wrote {out_path}")

    # Single map or per-hour
    if args.per_hour and "_time_h" in df.columns:
        for th, grp in df.groupby("_time_h", sort=True):
            suffix = f"_{th:%Y%m%d_%H%M}"
            base, ext = (Path(args.out_png).with_suffix("").as_posix(), Path(args.out_png).suffix or ".png")
            out = f"{base}{suffix}{ext}"
            ttl = args.title or "Seeds"
            _render(grp, out, f"{ttl} — {th:%Y-%m-%d %H:00}")
    else:
        _render(df, args.out_png, args.title or "Seeds")

if __name__ == "__main__":
    main()
