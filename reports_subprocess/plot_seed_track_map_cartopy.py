#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_seed_track_map_cartopy.py
Visualize seed vs track matches on a basemap with coastlines or simple tiles.

New features:
- Detects flexible column names (lat_cen/lon_cen, lat/lon, seed_lat/seed_lon).
- Optional AOI or auto extent.
- Optional color overlay by probability or any numeric column.
- Optional background tiles (e.g. StamenTerrain) for context.
- Optional direction arrows from bearing columns or seed drift.
"""

import argparse
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.img_tiles import StamenTerrain
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _thin_points(df: pd.DataFrame, time_col: Optional[str], max_per_hour: int, max_total: int) -> pd.DataFrame:
    # Agent: thin dense seed layers to avoid blob-like maps.
    if df.empty:
        return df
    out = df
    if max_per_hour and max_per_hour > 0 and time_col and time_col in out.columns:
        tvals = pd.to_datetime(out[time_col], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        if tvals.notna().any():
            keep_idx = []
            for _, sub in out.groupby(tvals, sort=False):
                if len(sub) > max_per_hour:
                    sub = sub.sample(int(max_per_hour), random_state=42)
                keep_idx.extend(sub.index.tolist())
            out = out.loc[keep_idx]
    if max_total and max_total > 0 and len(out) > max_total:
        out = out.sample(int(max_total), random_state=42)
    return out.reset_index(drop=True)

def _time_color_vals(df: pd.DataFrame, time_col: Optional[str]) -> tuple[Optional[np.ndarray], Optional[str]]:
    if time_col is None or time_col not in df.columns:
        return None, None
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert(None)
    if t.notna().sum() == 0:
        return None, None
    t0 = t.min()
    hours = (t - t0).dt.total_seconds() / 3600.0
    label = f"hours since {t0.strftime('%Y-%m-%d %H:%M')} UTC"
    return hours.to_numpy(), label


def _safe_storm_id(value: object) -> str:
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return safe or "storm"


def _pick_direction_col(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override and override in df.columns:
        return override
    for cand in ("track_bearing_deg", "obj_motion_bearing_deg", "obj_flow_bearing_deg", "bearing_deg"):
        if cand in df.columns:
            return cand
    return None


def _direction_subset(df: pd.DataFrame, max_arrows: int, seed: int) -> pd.DataFrame:
    if max_arrows and max_arrows > 0 and len(df) > max_arrows:
        return df.sample(n=int(max_arrows), random_state=seed)
    return df


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    if dlat == 0 and dlon == 0:
        return float("nan")
    return (math.degrees(math.atan2(dlon, dlat)) + 360.0) % 360.0


def main():
    ap = argparse.ArgumentParser(description="Plot seed-track matches on a map.")
    ap.add_argument("--matches", required=True, help="CSV/Parquet with seed-track matches.")
    ap.add_argument("--out", default="results/maps/seed_track_map.png", help="Output image path.")
    ap.add_argument("--lat-range", nargs=2, type=float, default=None, help="Optional map latitude range.")
    ap.add_argument("--lon-range", nargs=2, type=float, default=None, help="Optional map longitude range.")
    ap.add_argument("--overlay-prob", default=None, help="Optional numeric column for coloring seeds (e.g., prob_max).")
    ap.add_argument("--min-prob", type=float, default=None, help="Optional minimum overlay-prob to keep.")
    ap.add_argument("--top-quantile", type=float, default=None, help="Optional quantile filter on overlay-prob.")
    ap.add_argument("--per-storm", action="store_true", help="If storm id is present, emit one map per storm.")
    ap.add_argument("--storm-id-col", default=None, help="Storm id column name in matches (e.g., storm_id or name).")
    ap.add_argument("--use-tiles", action="store_true", help="Add background tiles (requires internet).")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-points-per-hour", type=int, default=0, help="Cap points per hour (0 disables).")
    ap.add_argument("--max-points-total", type=int, default=0, help="Cap total points after sampling (0 disables).")
    ap.add_argument("--color-by-time", action="store_true", help="Color seeds by time (hours since first seed).")
    ap.add_argument("--time-col", default=None, help="Optional time column override for coloring.")
    ap.add_argument("--time-cmap", default="viridis", help="Colormap for time coloring.")
    ap.add_argument("--direction-col", default=None, help="Optional bearing column (deg) for direction arrows.")
    ap.add_argument("--direction-scale", type=float, default=0.6, help="Arrow length in degrees.")
    ap.add_argument("--direction-color", default="tab:green", help="Color for direction arrows.")
    ap.add_argument("--max-direction-arrows", type=int, default=0, help="Cap direction arrows (0 disables).")
    ap.add_argument("--direction-seed", type=int, default=42, help="Random seed for arrow sampling.")
    ap.add_argument("--per-hour", action="store_true", help="Emit one map per hour when time is available.")
    ap.add_argument("--hour-step", type=int, default=1, help="Step between hours (e.g., 2 = every 2nd hour).")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 disables).")
    args = ap.parse_args()

    p = Path(args.matches)
    df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)

    # Pick columns flexibly
    lat_c = pick_col(df, ["lat_cen", "seed_lat", "lat"])
    lon_c = pick_col(df, ["lon_cen", "seed_lon", "lon"])
    lat_t = pick_col(df, ["storm_lat", "track_lat", "tc_lat"])
    lon_t = pick_col(df, ["storm_lon", "track_lon", "tc_lon"])

    if not lat_c or not lon_c:
        raise ValueError("Could not find seed latitude/longitude columns.")

    for c in [lat_c, lon_c, lat_t, lon_t]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional prob-based filters
    if args.overlay_prob and args.overlay_prob in df.columns:
        vals = pd.to_numeric(df[args.overlay_prob], errors="coerce")
        if args.min_prob is not None:
            df = df.loc[vals >= float(args.min_prob)]
        if args.top_quantile is not None:
            df = df.sample(frac=1.0, random_state=42)
            vals = pd.to_numeric(df[args.overlay_prob], errors="coerce")
            if "time" in df.columns:
                t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
                keep_idx = []
                for _, sub in df.groupby(t, sort=False):
                    svals = pd.to_numeric(sub[args.overlay_prob], errors="coerce")
                    if svals.notna().any():
                        cutoff = svals.quantile(float(args.top_quantile))
                        keep_idx.extend(sub.index[svals >= cutoff].tolist())
                df = df.loc[keep_idx]
            else:
                cutoff = vals.quantile(float(args.top_quantile))
                df = df.loc[vals >= cutoff]

    df = df.dropna(subset=[lat_c, lon_c]).reset_index(drop=True)
    time_col = args.time_col
    if time_col is None:
        for cand in ("time", "time_h", "seed_time"):
            if cand in df.columns:
                time_col = cand
                break
    if len(df):
        before = len(df)
        df = _thin_points(df, time_col, args.max_points_per_hour, args.max_points_total)
        if len(df) != before:
            print(f"[map] thinned points: {before:,} -> {len(df):,}")
    if len(df) == 0:
        print("[seed-track-map] no valid coordinate rows; skipping plot.")
        return

    dir_col = _pick_direction_col(df, args.direction_col)
    if dir_col is None:
        sid_col = args.storm_id_col
        if sid_col is None:
            for cand in ("storm_id", "name", "sid"):
                if cand in df.columns:
                    sid_col = cand
                    break
        if sid_col and time_col and sid_col in df.columns and time_col in df.columns:
            dir_vals = pd.Series(index=df.index, dtype="float64")
            for _, grp in df.groupby(sid_col, sort=False):
                grp = grp.sort_values(time_col)
                idxs = grp.index.to_list()
                coords = grp[[lat_c, lon_c]].to_numpy()
                for i, idx in enumerate(idxs):
                    if i + 1 < len(coords):
                        lat2, lon2 = coords[i + 1]
                    elif i > 0:
                        lat2, lon2 = coords[i - 1]
                    else:
                        dir_vals.loc[idx] = np.nan
                        continue
                    lat1, lon1 = coords[i]
                    dir_vals.loc[idx] = _bearing_deg(float(lat1), float(lon1), float(lat2), float(lon2))
            df["_dir_bearing"] = dir_vals
            dir_col = "_dir_bearing"
            print("[map] direction arrows from seed drift")
    if dir_col:
        df[dir_col] = pd.to_numeric(df[dir_col], errors="coerce")
        print(f"[map] direction arrows from {dir_col}")

    # Extent
    if args.lon_range and args.lat_range:
        lon_min, lon_max = args.lon_range
        lat_min, lat_max = args.lat_range
    else:
        lon_min, lon_max = float(df[lon_c].min()) - 1, float(df[lon_c].max()) + 1
        lat_min, lat_max = float(df[lat_c].min()) - 1, float(df[lat_c].max()) + 1

    # --- Plot ---
    use_cartopy = CARTOPY_AVAILABLE
    if not use_cartopy:
        print("[warn] Cartopy not available - falling back to plain scatter.")

    def render(ddf: pd.DataFrame, out_path: str, title: str):
        if not use_cartopy:
            plt.figure(figsize=(8,6))
            tvals, tlabel = _time_color_vals(ddf, time_col) if args.color_by_time else (None, None)
            if tvals is not None:
                sc = plt.scatter(ddf[lon_c], ddf[lat_c], s=12, c=tvals, cmap=args.time_cmap, alpha=0.6, label="Seeds")
                cb = plt.colorbar(sc, orientation="vertical", shrink=0.7)
                cb.set_label(tlabel or "hours since first seed")
            elif args.overlay_prob and args.overlay_prob in ddf.columns:
                vals = pd.to_numeric(ddf[args.overlay_prob], errors="coerce")
                sc = plt.scatter(ddf[lon_c], ddf[lat_c], s=12, c=vals, cmap="viridis", alpha=0.6, label="Seeds")
                cb = plt.colorbar(sc, orientation="vertical", shrink=0.7)
                cb.set_label(args.overlay_prob)
            else:
                plt.scatter(ddf[lon_c], ddf[lat_c], s=12, c="tab:blue", alpha=0.6, label="Seeds")
            if dir_col and dir_col in ddf.columns:
                dir_vals = pd.to_numeric(ddf[dir_col], errors="coerce")
                sub = ddf.loc[np.isfinite(dir_vals)]
                sub = _direction_subset(sub, args.max_direction_arrows, args.direction_seed)
                for _, r in sub.iterrows():
                    bearing = math.radians(float(r[dir_col]))
                    dx = args.direction_scale * math.sin(bearing)
                    dy = args.direction_scale * math.cos(bearing)
                    plt.arrow(
                        r[lon_c],
                        r[lat_c],
                        dx,
                        dy,
                        width=0.03,
                        head_width=0.12,
                        head_length=0.12,
                        color=args.direction_color,
                        alpha=0.6,
                        length_includes_head=True,
                    )
            if lat_t and lon_t:
                plt.scatter(ddf[lon_t], ddf[lat_t], s=20, c="tab:red", marker="x", label="Tracks")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(title)
            plt.legend()
            plt.xlim(lon_min, lon_max)
            plt.ylim(lat_min, lat_max)
            plt.tight_layout()
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=args.dpi)
            plt.close()
            print(f"[map] saved (simple) -> {out_path}")
            return

        fig = plt.figure(figsize=(9,7))
        if args.use_tiles:
            tiler = StamenTerrain()
            ax = plt.axes(projection=tiler.crs)
            ax.add_image(tiler, 6)
        else:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.6)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
            ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.6)

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        tvals, tlabel = _time_color_vals(ddf, time_col) if args.color_by_time else (None, None)
        if tvals is not None:
            sc = ax.scatter(ddf[lon_c], ddf[lat_c], s=18, c=tvals, cmap=args.time_cmap,
                            transform=ccrs.PlateCarree(), label="Seeds", alpha=0.8)
            cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
            cb.set_label(tlabel or "hours since first seed")
        elif args.overlay_prob and args.overlay_prob in ddf.columns:
            vals = pd.to_numeric(ddf[args.overlay_prob], errors="coerce")
            sc = ax.scatter(ddf[lon_c], ddf[lat_c], s=18, c=vals, cmap="viridis",
                            transform=ccrs.PlateCarree(), label="Seeds", alpha=0.8)
            cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
            cb.set_label(args.overlay_prob)
        else:
            ax.scatter(ddf[lon_c], ddf[lat_c], s=18, color="tab:blue", transform=ccrs.PlateCarree(),
                       alpha=0.7, label="Seeds")
        if dir_col and dir_col in ddf.columns:
            dir_vals = pd.to_numeric(ddf[dir_col], errors="coerce")
            sub = ddf.loc[np.isfinite(dir_vals)]
            sub = _direction_subset(sub, args.max_direction_arrows, args.direction_seed)
            for _, r in sub.iterrows():
                bearing = math.radians(float(r[dir_col]))
                dx = args.direction_scale * math.sin(bearing)
                dy = args.direction_scale * math.cos(bearing)
                ax.arrow(
                    r[lon_c],
                    r[lat_c],
                    dx,
                    dy,
                    width=0.03,
                    head_width=0.12,
                    head_length=0.12,
                    color=args.direction_color,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                    length_includes_head=True,
                )

        if lat_t and lon_t:
            ax.scatter(ddf[lon_t], ddf[lat_t], s=25, color="tab:red", marker="x",
                       transform=ccrs.PlateCarree(), label="Track")
            for _, r in ddf.iterrows():
                ax.plot([r[lon_c], r[lon_t]], [r[lat_c], r[lat_t]],
                        color="gray", lw=0.5, alpha=0.5, transform=ccrs.PlateCarree())

        ax.legend(loc="lower left", frameon=False)
        ax.set_title(title)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"[map] saved -> {out_path}")

    # per-hour maps (optionally per-storm)
    if args.per_hour:
        if time_col is None or time_col not in df.columns:
            print("[seed-track-map] --per-hour requested but no time column found; falling back to single map.")
        else:
            tvals = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
            if tvals.notna().sum() == 0:
                print("[seed-track-map] --per-hour requested but time values are invalid; falling back to single map.")
            else:
                df = df.assign(_time_h=tvals)
                hours = sorted(df["_time_h"].dropna().unique().tolist())
                step = max(1, int(args.hour_step))
                if step > 1:
                    hours = hours[::step]
                if args.max_frames and args.max_frames > 0:
                    hours = hours[: int(args.max_frames)]

                out_base = Path(args.out).with_suffix("")
                ext = Path(args.out).suffix or ".png"
                if args.per_storm:
                    sid_col = args.storm_id_col
                    if sid_col is None:
                        for cand in ("storm_id", "name", "sid"):
                            if cand in df.columns:
                                sid_col = cand
                                break
                    if sid_col and sid_col in df.columns:
                        for sid, grp in df.groupby(sid_col):
                            sid_safe = _safe_storm_id(sid)
                            for h in hours:
                                sub = grp.loc[grp["_time_h"] == h]
                                if sub.empty:
                                    continue
                                stamp = pd.Timestamp(h).strftime("%Y%m%d%H")
                                render(
                                    sub,
                                    f"{out_base}_storm_{sid_safe}_{stamp}{ext}",
                                    f"Seed-Track Matches - {sid} {stamp}",
                                )
                        return
                for h in hours:
                    sub = df.loc[df["_time_h"] == h]
                    if sub.empty:
                        continue
                    stamp = pd.Timestamp(h).strftime("%Y%m%d%H")
                    render(sub, f"{out_base}_{stamp}{ext}", f"Seed-Track Matches {stamp}")
                return

    # per-storm if requested
    if args.per_storm:
        sid_col = args.storm_id_col
        if sid_col is None:
            for cand in ("storm_id", "name", "sid"):
                if cand in df.columns:
                    sid_col = cand
                    break
        if sid_col and sid_col in df.columns:
            out_base = Path(args.out).with_suffix("")
            ext = Path(args.out).suffix or ".png"
            for sid, grp in df.groupby(sid_col):
                sid_safe = _safe_storm_id(sid)
                render(grp, f"{out_base}_storm_{sid_safe}{ext}", f"Seed-Track Matches - {sid}")
            return

    # single map
    render(df, args.out, "Seed–Track Matches")


if __name__ == "__main__":
    main()
