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

def _stable_sample(df: pd.DataFrame, n: int, cols: list[str]) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        return df.head(n)
    key = df[use_cols].copy()
    for c in use_cols:
        if np.issubdtype(key[c].dtype, np.datetime64):
            key[c] = key[c].view("int64")
    hashes = pd.util.hash_pandas_object(key, index=False)
    return df.loc[hashes.sort_values().head(n).index]

def _sample_rows(df: pd.DataFrame, n: int, cols: list[str], mode: str, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    if mode == "stable":
        return _stable_sample(df, n, cols)
    return df.sample(int(n), random_state=seed)

def _thin_points(
    df: pd.DataFrame,
    time_col: Optional[str],
    max_per_hour: int,
    max_total: int,
    mode: str,
    seed: int,
    key_cols: list[str],
) -> pd.DataFrame:
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
                    sub = _sample_rows(sub, max_per_hour, key_cols, mode, seed)
                keep_idx.extend(sub.index.tolist())
            out = out.loc[keep_idx]
    if max_total and max_total > 0 and len(out) > max_total:
        out = _sample_rows(out, max_total, key_cols, mode, seed)
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
    for cand in (
        "track_bearing_deg",
        "obj_motion_bearing_deg",
        "obj_flow_bearing_deg",
        "obj_axis_bearing_deg",
        "bearing_deg",
    ):
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

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq", ".pqt"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


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
    ap.add_argument("--sample-mode", choices=["random", "stable"], default="stable", help="Sampling mode for thinning.")
    ap.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling.")
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
    ap.add_argument("--trail-hours", type=int, default=0, help="Include prior hours in per-hour frames.")
    ap.add_argument("--background", default=None, help="Optional background CSV/Parquet (objects/alerts).")
    ap.add_argument("--background-lat-col", default=None, help="Background latitude column override.")
    ap.add_argument("--background-lon-col", default=None, help="Background longitude column override.")
    ap.add_argument("--background-time-col", default=None, help="Background time column override.")
    ap.add_argument("--background-value-col", default=None, help="Background numeric column for filtering.")
    ap.add_argument("--background-min-value", type=float, default=None, help="Min background value to keep.")
    ap.add_argument("--background-top-quantile", type=float, default=None, help="Per-hour background quantile filter.")
    ap.add_argument("--background-max-points-per-hour", type=int, default=0, help="Cap background points per hour.")
    ap.add_argument("--background-max-points-total", type=int, default=0, help="Cap total background points.")
    ap.add_argument("--background-color", default="#9aa0a6", help="Background point color.")
    ap.add_argument("--background-alpha", type=float, default=0.25, help="Background point alpha.")
    ap.add_argument("--background-size", type=float, default=10.0, help="Background point size.")
    ap.add_argument("--background-color-by-time", action="store_true", help="Color background points by time.")
    ap.add_argument("--background-time-cmap", default="viridis", help="Colormap for background time coloring.")
    args = ap.parse_args()

    p = Path(args.matches)
    df = pd.read_parquet(p) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(p)

    # Pick columns flexibly
    lat_c = pick_col(df, ["lat_cen", "seed_lat", "obj_centroid_lat", "obj_core_lat", "lat"])
    lon_c = pick_col(df, ["lon_cen", "seed_lon", "obj_centroid_lon", "obj_core_lon", "lon"])
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
        for cand in ("time", "time_h", "seed_time", "object_time"):
            if cand in df.columns:
                time_col = cand
                break
    key_cols = [c for c in [lat_c, lon_c, time_col, "patch_id", "object_id"] if c]
    if len(df):
        before = len(df)
        df = _thin_points(
            df,
            time_col,
            args.max_points_per_hour,
            args.max_points_total,
            args.sample_mode,
            args.sample_seed,
            key_cols,
        )
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

    bg_df = None
    bg_lat = None
    bg_lon = None
    bg_time_col = None
    if args.background:
        bg_path = Path(args.background)
        if not bg_path.exists():
            print(f"[map] background file not found: {bg_path}")
        else:
            bg_df = _read_any(bg_path)
            if not bg_df.empty:
                bg_lat = args.background_lat_col or pick_col(
                    bg_df,
                    ["obj_centroid_lat", "lat_cen", "seed_lat", "lat"],
                )
                bg_lon = args.background_lon_col or pick_col(
                    bg_df,
                    ["obj_centroid_lon", "lon_cen", "seed_lon", "lon"],
                )
                bg_time_col = args.background_time_col or pick_col(
                    bg_df,
                    ["time", "time_h", "object_time", "seed_time"],
                )
                if not bg_lat or not bg_lon:
                    print("[map] background missing lat/lon columns; skipping background overlay.")
                    bg_df = None
                else:
                    bg_df[bg_lat] = pd.to_numeric(bg_df[bg_lat], errors="coerce")
                    bg_df[bg_lon] = pd.to_numeric(bg_df[bg_lon], errors="coerce")
                    bg_df = bg_df.dropna(subset=[bg_lat, bg_lon]).reset_index(drop=True)
            if bg_df is not None and not bg_df.empty:
                if bg_time_col and bg_time_col in bg_df.columns:
                    tvals = pd.to_datetime(bg_df[bg_time_col], utc=True, errors="coerce").dt.tz_convert(None)
                    bg_df = bg_df.assign(_bg_time_h=tvals.dt.floor("h"))
                if args.background_value_col and args.background_value_col in bg_df.columns:
                    vals = pd.to_numeric(bg_df[args.background_value_col], errors="coerce")
                    if args.background_min_value is not None:
                        bg_df = bg_df.loc[vals >= float(args.background_min_value)]
                    if args.background_top_quantile is not None:
                        if "_bg_time_h" in bg_df.columns:
                            keep_idx = []
                            for _, sub in bg_df.groupby("_bg_time_h", sort=False):
                                svals = pd.to_numeric(sub[args.background_value_col], errors="coerce")
                                if svals.notna().any():
                                    cutoff = svals.quantile(float(args.background_top_quantile))
                                    keep_idx.extend(sub.index[svals >= cutoff].tolist())
                            bg_df = bg_df.loc[keep_idx]
                        else:
                            cutoff = vals.quantile(float(args.background_top_quantile))
                            bg_df = bg_df.loc[vals >= cutoff]
                bg_key_cols = [c for c in [bg_lat, bg_lon, bg_time_col, "object_id"] if c]
                bg_df = _thin_points(
                    bg_df,
                    bg_time_col if bg_time_col in (bg_df.columns if bg_df is not None else []) else None,
                    args.background_max_points_per_hour,
                    args.background_max_points_total,
                    args.sample_mode,
                    args.sample_seed,
                    bg_key_cols,
                )
                if bg_df is not None and bg_df.empty:
                    bg_df = None

    # Extent
    if args.lon_range and args.lat_range:
        lon_min, lon_max = args.lon_range
        lat_min, lat_max = args.lat_range
    else:
        lon_min, lon_max = float(df[lon_c].min()) - 1, float(df[lon_c].max()) + 1
        lat_min, lat_max = float(df[lat_c].min()) - 1, float(df[lat_c].max()) + 1
        if bg_df is not None and bg_lat and bg_lon and not bg_df.empty:
            lon_min = min(lon_min, float(bg_df[bg_lon].min()) - 1)
            lon_max = max(lon_max, float(bg_df[bg_lon].max()) + 1)
            lat_min = min(lat_min, float(bg_df[bg_lat].min()) - 1)
            lat_max = max(lat_max, float(bg_df[bg_lat].max()) + 1)

    # --- Plot ---
    use_cartopy = CARTOPY_AVAILABLE
    if not use_cartopy:
        print("[warn] Cartopy not available - falling back to plain scatter.")

    def render(ddf: pd.DataFrame, out_path: str, title: str, bg: Optional[pd.DataFrame] = None):
        if not use_cartopy:
            plt.figure(figsize=(8,6))
            if bg is not None and bg_lat and bg_lon and not bg.empty:
                if args.background_color_by_time and bg_time_col and bg_time_col in bg.columns:
                    bt, _ = _time_color_vals(bg, bg_time_col)
                    if bt is not None:
                        plt.scatter(
                            bg[bg_lon],
                            bg[bg_lat],
                            s=args.background_size,
                            c=bt,
                            cmap=args.background_time_cmap,
                            alpha=args.background_alpha,
                            label=None,
                        )
                    else:
                        plt.scatter(
                            bg[bg_lon],
                            bg[bg_lat],
                            s=args.background_size,
                            color=args.background_color,
                            alpha=args.background_alpha,
                            label=None,
                        )
                else:
                    plt.scatter(
                        bg[bg_lon],
                        bg[bg_lat],
                        s=args.background_size,
                        color=args.background_color,
                        alpha=args.background_alpha,
                        label=None,
                    )
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

        if bg is not None and bg_lat and bg_lon and not bg.empty:
            if args.background_color_by_time and bg_time_col and bg_time_col in bg.columns:
                bt, _ = _time_color_vals(bg, bg_time_col)
                if bt is not None:
                    ax.scatter(
                        bg[bg_lon],
                        bg[bg_lat],
                        s=args.background_size,
                        c=bt,
                        cmap=args.background_time_cmap,
                        transform=ccrs.PlateCarree(),
                        alpha=args.background_alpha,
                        label=None,
                    )
                else:
                    ax.scatter(
                        bg[bg_lon],
                        bg[bg_lat],
                        s=args.background_size,
                        color=args.background_color,
                        transform=ccrs.PlateCarree(),
                        alpha=args.background_alpha,
                        label=None,
                    )
            else:
                ax.scatter(
                    bg[bg_lon],
                    bg[bg_lat],
                    s=args.background_size,
                    color=args.background_color,
                    transform=ccrs.PlateCarree(),
                    alpha=args.background_alpha,
                    label=None,
                )

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

    def _subset_for_hour(
        ddf: Optional[pd.DataFrame],
        hour: pd.Timestamp,
        time_h_col: str,
    ) -> Optional[pd.DataFrame]:
        if ddf is None or ddf.empty or time_h_col not in ddf.columns:
            return ddf
        if args.trail_hours and args.trail_hours > 0:
            start = hour - pd.Timedelta(hours=int(args.trail_hours))
            return ddf.loc[(ddf[time_h_col] >= start) & (ddf[time_h_col] <= hour)]
        return ddf.loc[ddf[time_h_col] == hour]

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
                                sub = _subset_for_hour(grp, pd.Timestamp(h), "_time_h")
                                if sub.empty:
                                    continue
                                bg_sub = _subset_for_hour(bg_df, pd.Timestamp(h), "_bg_time_h") if bg_df is not None else None
                                stamp = pd.Timestamp(h).strftime("%Y%m%d%H")
                                render(
                                    sub,
                                    f"{out_base}_storm_{sid_safe}_{stamp}{ext}",
                                    f"Seed-Track Matches - {sid} {stamp}",
                                    bg_sub,
                                )
                        return
                    print("[seed-track-map] --per-storm requested but no storm id column found.")
                for h in hours:
                    sub = _subset_for_hour(df, pd.Timestamp(h), "_time_h")
                    if sub.empty:
                        continue
                    bg_sub = _subset_for_hour(bg_df, pd.Timestamp(h), "_bg_time_h") if bg_df is not None else None
                    stamp = pd.Timestamp(h).strftime("%Y%m%d%H")
                    render(sub, f"{out_base}_{stamp}{ext}", f"Seed-Track Matches {stamp}", bg_sub)
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
                render(grp, f"{out_base}_storm_{sid_safe}{ext}", f"Seed-Track Matches - {sid}", bg_df)
            return

    # single map
    render(df, args.out, "Seed-Track Matches", bg_df)


if __name__ == "__main__":
    main()
