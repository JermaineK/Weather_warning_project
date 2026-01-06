#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

def _norm_lon(x, mode):
    x = pd.to_numeric(x, errors="coerce")
    if mode == "0..360":
        return (x % 360 + 360) % 360
    if mode == "-180..180":
        return ((x + 180) % 360) - 180
    return x

def _safe_storm_id(value: object) -> str:
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return safe or "storm"

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

def _sample_rows(
    df: pd.DataFrame,
    n: int,
    cols: list[str],
    mode: str,
    seed: int,
) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    if mode == "stable":
        return _stable_sample(df, n, cols)
    return df.sample(int(n), random_state=seed)

def _time_color_vals(df: pd.DataFrame, time_col: str) -> tuple[np.ndarray | None, str | None]:
    if time_col not in df.columns:
        return None, None
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert(None)
    if t.notna().sum() == 0:
        return None, None
    t0 = t.min()
    hours = (t - t0).dt.total_seconds() / 3600.0
    label = f"hours since {t0.strftime('%Y-%m-%d %H:%M')} UTC"
    return hours.to_numpy(), label

def _apply_sampling(
    df: pd.DataFrame,
    time_h_col: str | None,
    max_per_hour: int,
    max_total: int,
    sample_mode: str,
    sample_seed: int,
    key_cols: list[str],
) -> pd.DataFrame:
    # Agent: stable thinning to avoid GIF jitter while keeping per-hour balance.
    out = df
    if max_per_hour and max_per_hour > 0 and time_h_col and time_h_col in out.columns:
        keep_idx = []
        for _, sub in out.groupby(time_h_col, sort=False):
            if len(sub) > max_per_hour:
                sub = _sample_rows(sub, max_per_hour, key_cols, sample_mode, sample_seed)
            keep_idx.extend(sub.index.tolist())
        out = out.loc[keep_idx]
    if max_total and max_total > 0 and len(out) > max_total:
        out = _sample_rows(out, max_total, key_cols, sample_mode, sample_seed)
    return out.reset_index(drop=True)

def _parse_area(aoi):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(t.strip()) for t in aoi.split(",")]
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

def _load_tracks(path, normalize_lon_mode, area):
    tp = str(path).lower()
    if tp.endswith((".parquet", ".parq", ".pq")):
        tr = pd.read_parquet(path)
    else:
        tr = pd.read_csv(path)
    if not {"lat", "lon", "time"}.issubset(tr.columns):
        return None
    tr = tr.copy()
    tr["lat"] = pd.to_numeric(tr["lat"], errors="coerce")
    tr["lon"] = _norm_lon(tr["lon"], normalize_lon_mode)
    ttime = pd.to_datetime(tr["time"], utc=True, errors="coerce").dt.tz_convert(None)
    tr = tr.assign(time=ttime).dropna(subset=["lat", "lon", "time"])
    if area:
        latN, lonW, latS, lonE = _parse_area(area)
        tr = tr.loc[
            (tr["lat"] <= latN) & (tr["lat"] >= latS) & (tr["lon"] >= lonW) & (tr["lon"] <= lonE)
        ]
    return tr if not tr.empty else None


def main():
    ap = argparse.ArgumentParser(description="Cartopy seed map renderer")
    ap.add_argument("--seeds", required=True, help="CSV/Parquet with at least: lat, lon[, value][, time]")
    ap.add_argument("--out-png", required=True, help="Output PNG path (or prefix if --per-hour).")
    ap.add_argument("--value-col", default="prob_max", help="Optional numeric column to color/size by")
    ap.add_argument("--time-col", default=None, help="Optional time column; if provided with --per-hour, make one map/hour")
    ap.add_argument("--per-hour", action="store_true", help="Produce one PNG per hour if time is available")
    ap.add_argument("--hour-step", type=int, default=1, help="Step between hours (e.g., 2 = every 2nd hour).")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit per-hour frames (0 disables).")
    ap.add_argument("--flag-col", default=None, help="Optional flag column; if provided, filter to rows == 1")
    ap.add_argument("--min-prob", type=float, default=0.5, help="Minimum value/prob to plot (filters points)")
    ap.add_argument("--top-quantile", type=float, default=None, help="Keep only rows with value_col above this quantile (0-1)")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help='Optional AOI "latN,lonW,latS,lonE"')
    ap.add_argument("--tracks", default=None, help="Optional tracks file to time/space-filter seeds (CSV/Parquet).")
    ap.add_argument("--storm-radius-deg", type=float, default=5.0, help="Lat/lon padding around track bbox for filtering.")
    ap.add_argument("--storm-window-before-h", type=float, default=240.0, help="Hours before track times to include seeds.")
    ap.add_argument("--storm-window-after-h", type=float, default=72.0, help="Hours after track times to include seeds.")
    ap.add_argument("--per-storm", action="store_true", help="If tracks provided, emit one map per storm id/name.")
    ap.add_argument("--storm-id-col", default=None, help="ID column in tracks (e.g., storm_id or name).")
    ap.add_argument("--title", default=None, help="Figure title")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--max-points-per-hour", type=int, default=2000, help="Cap points per hour for scatter maps.")
    ap.add_argument("--max-points-total", type=int, default=20000, help="Cap total points after sampling (0 disables).")
    ap.add_argument("--sample-mode", choices=["random", "stable"], default="stable", help="Sampling mode for thinning.")
    ap.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling.")
    ap.add_argument("--color-by-time", action="store_true", help="Color seeds by time (hours since first seed).")
    ap.add_argument("--time-cmap", default="viridis", help="Colormap for time coloring.")
    ap.add_argument("--trail-hours", type=int, default=0, help="Include prior hours in per-hour frames.")
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
        df = df.sample(frac=1.0, random_state=42)
        vals = pd.to_numeric(df[args.value_col], errors="coerce")
        if "time" in df.columns:
            t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
            keep_idx = []
            for _, sub in df.groupby(t, sort=False):
                svals = pd.to_numeric(sub[args.value_col], errors="coerce")
                if svals.notna().any():
                    cutoff = svals.quantile(float(args.top_quantile))
                    keep_idx.extend(sub.index[svals >= cutoff].tolist())
            df = df.loc[keep_idx]
        else:
            cutoff = vals.quantile(float(args.top_quantile))
            df = df.loc[vals >= cutoff]
    # Optional filter: restrict to storms window/bbox
    tracks_df = None
    if args.tracks:
        try:
            tracks_df = _load_tracks(args.tracks, args.normalize_lon, args.area)
        except Exception as e:
            print(f"[map] warning: failed to load tracks {args.tracks}: {e}")
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

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
    tcol_name = None
    if args.time_col and args.time_col in df.columns:
        tcol_name = args.time_col
    elif "time" in df.columns:
        tcol_name = "time"
    elif "time_h" in df.columns:
        tcol_name = "time_h"
    if tcol_name:
        tcol = pd.to_datetime(df[tcol_name], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
        df["_time_h"] = tcol
    elif args.per_hour:
        print("[map] --per-hour given but time column missing; producing single map.", flush=True)

    key_cols = [c for c in ["patch_id", "lat", "lon", tcol_name] if c]
    if not (args.per_storm and tracks_df is not None):
        df = _apply_sampling(
            df,
            "_time_h" if "_time_h" in df.columns else None,
            args.max_points_per_hour,
            args.max_points_total,
            args.sample_mode,
            args.sample_seed,
            key_cols,
        )

    # Cartopy import
    ccrs, cfeature = _load_cartopy()
    import matplotlib.pyplot as plt

    def _render(ddf, out_path, title, tracks=None):
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
        tvals, tlabel = _time_color_vals(ddf, tcol_name) if args.color_by_time and tcol_name else (None, None)
        if tvals is not None:
            sc = ax.scatter(
                ddf["lon"],
                ddf["lat"],
                c=tvals,
                s=14,
                cmap=args.time_cmap,
                alpha=0.85,
                transform=proj,
            )
            cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
            cb.set_label(tlabel or "hours since first seed")
        elif val is not None:
            vv = pd.to_numeric(ddf[args.value_col], errors="coerce")
            s = 6 + 24 * (vv - vv.min()) / (vv.max() - vv.min() + 1e-12)
            sc = ax.scatter(ddf["lon"], ddf["lat"], c=vv, s=s, cmap="viridis", alpha=0.85, transform=proj)
            cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
            cb.set_label(args.value_col)
        else:
            ax.scatter(ddf["lon"], ddf["lat"], s=18, alpha=0.85, transform=proj)

        # Optional track overlay
        if tracks is not None and not tracks.empty:
            ax.plot(tracks["lon"], tracks["lat"], color="tab:red", lw=1.2, alpha=0.8, transform=proj, label="Track")
            ax.scatter(
                tracks["lon"].iloc[:1],
                tracks["lat"].iloc[:1],
                color="tab:red",
                s=30,
                marker="x",
                transform=proj,
                label="Track start",
            )
            ax.scatter(
                tracks["lon"].iloc[-1:],
                tracks["lat"].iloc[-1:],
                color="tab:red",
                s=24,
                marker="o",
                facecolors="none",
                transform=proj,
                label="Track end",
            )
            ax.legend(loc="lower left", frameon=False)

        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
        if title:
            ax.set_title(title, fontsize=12)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[map] wrote {out_path}")

    # Utility to filter seeds to track window/bbox
    def filter_to_tracks(seeds_df: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
        tcol = tcol_name if tcol_name and tcol_name in seeds_df.columns else ("time" if "time" in seeds_df.columns else None)
        if tcol is None:
            return seeds_df
        tseeds = pd.to_datetime(seeds_df[tcol], utc=True, errors="coerce").dt.tz_convert(None)
        lat_min = tr["lat"].min() - args.storm_radius_deg
        lat_max = tr["lat"].max() + args.storm_radius_deg
        lon_min = tr["lon"].min() - args.storm_radius_deg
        lon_max = tr["lon"].max() + args.storm_radius_deg
        tmin = tr["time"].min() - pd.Timedelta(hours=args.storm_window_before_h)
        tmax = tr["time"].max() + pd.Timedelta(hours=args.storm_window_after_h)
        return seeds_df.loc[
            tseeds.between(tmin, tmax)
            & seeds_df["lat"].between(lat_min, lat_max)
            & seeds_df["lon"].between(lon_min, lon_max)
        ]

    def _subset_for_hour(ddf: pd.DataFrame, hour: pd.Timestamp) -> pd.DataFrame:
        if args.trail_hours and args.trail_hours > 0:
            start = hour - pd.Timedelta(hours=int(args.trail_hours))
            return ddf.loc[(ddf["_time_h"] >= start) & (ddf["_time_h"] <= hour)]
        return ddf.loc[ddf["_time_h"] == hour]

    # Single map, per-hour, or per-storm
    if args.per_storm and tracks_df is not None:
        id_col = args.storm_id_col
        if id_col is None:
            for cand in ("storm_id", "name", "sid"):
                if cand in tracks_df.columns:
                    id_col = cand
                    break
        if id_col is None or id_col not in tracks_df.columns:
            print("[map] per-storm requested but no storm id column found; falling back to single map.")
        else:
            for sid, tr_grp in tracks_df.groupby(id_col):
                seeds_sub = filter_to_tracks(df, tr_grp)
                if seeds_sub.empty:
                    continue
                seeds_sub = _apply_sampling(
                    seeds_sub,
                    "_time_h" if "_time_h" in seeds_sub.columns else None,
                    args.max_points_per_hour,
                    args.max_points_total,
                    args.sample_mode,
                    args.sample_seed,
                    key_cols,
                )
                ttl = f"{args.title or 'Seeds'} - storm {sid}"
                base = Path(args.out_png).with_suffix("")
                ext = Path(args.out_png).suffix or ".png"
                sid_safe = _safe_storm_id(sid)
                if args.per_hour and "_time_h" in seeds_sub.columns:
                    hours = sorted(seeds_sub["_time_h"].dropna().unique().tolist())
                    step = max(1, int(args.hour_step))
                    if step > 1:
                        hours = hours[::step]
                    if args.max_frames and args.max_frames > 0:
                        hours = hours[: int(args.max_frames)]
                    for th in hours:
                        grp = _subset_for_hour(seeds_sub, pd.Timestamp(th))
                        if grp.empty:
                            continue
                        suffix = f"_{pd.Timestamp(th):%Y%m%d_%H%M}"
                        out = f"{base}_storm_{sid_safe}{suffix}{ext}"
                        _render(grp, out, f"{ttl} - {pd.Timestamp(th):%Y-%m-%d %H:00}", tr_grp)
                else:
                    out = f"{base}_storm_{sid_safe}{ext}"
                    _render(seeds_sub, out, ttl, tr_grp)
            return

    if args.per_hour and "_time_h" in df.columns:
        hours = sorted(df["_time_h"].dropna().unique().tolist())
        step = max(1, int(args.hour_step))
        if step > 1:
            hours = hours[::step]
        if args.max_frames and args.max_frames > 0:
            hours = hours[: int(args.max_frames)]
        base, ext = (Path(args.out_png).with_suffix("").as_posix(), Path(args.out_png).suffix or ".png")
        for th in hours:
            grp = _subset_for_hour(df, pd.Timestamp(th))
            if grp.empty:
                continue
            suffix = f"_{pd.Timestamp(th):%Y%m%d_%H%M}"
            out = f"{base}{suffix}{ext}"
            ttl = args.title or "Seeds"
            _render(grp, out, f"{ttl} - {pd.Timestamp(th):%Y-%m-%d %H:00}")
    else:
        _render(df, args.out_png, args.title or "Seeds", tracks_df)

if __name__ == "__main__":
    main()
