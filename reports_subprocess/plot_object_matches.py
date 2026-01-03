#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_object_matches.py

Agent: render per-storm object-based maps with motion overlays.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq", ".pqt"))


def _read_any(path: str | Path) -> pd.DataFrame:
    if _is_parquet(path):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180


def _load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def _pick_track_cols(df: pd.DataFrame, overrides: Dict[str, Optional[str]]) -> Dict[str, str]:
    def _first(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    time_col = overrides.get("time") or _first(["time", "obs_time", "datetime", "valid_time"])
    lat_col = overrides.get("lat") or _first(["lat", "latitude"])
    lon_col = overrides.get("lon") or _first(["lon", "longitude"])
    id_col = overrides.get("id") or _first(["storm_id", "sid", "name"])
    if not time_col or not lat_col or not lon_col:
        raise SystemExit("Tracks file missing required time/lat/lon columns.")
    return {"time": time_col, "lat": lat_col, "lon": lon_col, "id": id_col or "storm_id"}


def _plot_one(
    ax,
    tracks: pd.DataFrame,
    all_objs: pd.DataFrame,
    cand_objs: pd.DataFrame,
    match_objs: pd.DataFrame,
    title: str,
    show_arrows: bool,
    show_flow_arrows: bool,
    arrow_scale: float,
    label_arrows: bool,
):
    if tracks is not None and not tracks.empty:
        t_sorted = tracks.sort_values("time")
        ax.plot(t_sorted["lon"], t_sorted["lat"], color="#1f77b4", linewidth=2.0, alpha=0.9, label="Track")

    if all_objs is not None and not all_objs.empty:
        ax.scatter(all_objs["obj_centroid_lon"], all_objs["obj_centroid_lat"],
                   s=10, c="#9aa0a6", alpha=0.35, label="All objects")

    if cand_objs is not None and not cand_objs.empty:
        ax.scatter(cand_objs["obj_centroid_lon"], cand_objs["obj_centroid_lat"],
                   s=18, c="#f58518", alpha=0.8, label="Candidates")

    if match_objs is not None and not match_objs.empty:
        ax.scatter(match_objs["obj_centroid_lon"], match_objs["obj_centroid_lat"],
                   s=28, c="#d62728", alpha=0.9, label="Matched objects")
        if show_arrows:
            for _, r in match_objs.iterrows():
                if not np.isfinite(r.get("obj_motion_bearing_deg", np.nan)):
                    continue
                bearing = math.radians(float(r["obj_motion_bearing_deg"]))
                dx = arrow_scale * math.sin(bearing)
                dy = arrow_scale * math.cos(bearing)
                ax.arrow(
                    r["obj_centroid_lon"],
                    r["obj_centroid_lat"],
                    dx,
                    dy,
                    width=0.03,
                    head_width=0.12,
                    head_length=0.12,
                    color="#2ca02c",
                    alpha=0.7,
                    length_includes_head=True,
                )
                if label_arrows:
                    spd = r.get("obj_motion_speed_kmh", np.nan)
                    lbl = f"{spd:.0f} km/h {float(r['obj_motion_bearing_deg']):.0f}°" if np.isfinite(spd) else f"{float(r['obj_motion_bearing_deg']):.0f}°"
                    ax.text(
                        r["obj_centroid_lon"] + 0.05,
                        r["obj_centroid_lat"] + 0.05,
                        lbl,
                        fontsize=7,
                        color="#2ca02c",
                    )
                if show_flow_arrows and np.isfinite(r.get("obj_flow_bearing_deg", np.nan)):
                    fb = math.radians(float(r["obj_flow_bearing_deg"]))
                    fdx = arrow_scale * math.sin(fb)
                    fdy = arrow_scale * math.cos(fb)
                    ax.arrow(
                        r["obj_centroid_lon"],
                        r["obj_centroid_lat"],
                        fdx,
                        fdy,
                        width=0.02,
                        head_width=0.09,
                        head_length=0.09,
                        color="#1f77b4",
                        alpha=0.6,
                        length_includes_head=True,
                    )
                    if label_arrows:
                        fspd = r.get("obj_flow_speed_kmh", np.nan)
                        flbl = f"{fspd:.0f} km/h {float(r['obj_flow_bearing_deg']):.0f}°" if np.isfinite(fspd) else f"{float(r['obj_flow_bearing_deg']):.0f}°"
                        ax.text(
                            r["obj_centroid_lon"] + 0.05,
                            r["obj_centroid_lat"] - 0.08,
                            flbl,
                            fontsize=7,
                            color="#1f77b4",
                        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-storm object-based maps with optional hourly slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--matches", required=True, help="storm_object_matches.parquet")
    ap.add_argument("--objects", default=None, help="Optional objects_by_hour table for background points.")
    ap.add_argument("--tracks", required=True, help="IBTrACS subset (CSV/Parquet).")
    ap.add_argument("--out-dir", default="results/reports/object_maps")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180")
    ap.add_argument("--hours-before", type=float, default=72.0)
    ap.add_argument("--hours-after", type=float, default=24.0)
    ap.add_argument("--objects-score-col", default=None, help="Score column for objects (optional).")
    ap.add_argument("--objects-min-score", type=float, default=None, help="Minimum object score to plot.")
    ap.add_argument("--objects-top-quantile", type=float, default=None, help="Per-hour score quantile to plot.")
    ap.add_argument("--objects-pad-deg", type=float, default=5.0, help="Padding around track bbox for objects.")
    ap.add_argument("--per-hour", action="store_true", help="Emit one map per hour in the window.")
    ap.add_argument("--hour-step", type=int, default=1, help="Step between hours (e.g., 2 = every 2nd hour).")
    ap.add_argument("--show-arrows", action="store_true", help="Overlay motion direction arrows.")
    ap.add_argument("--show-flow-arrows", action="store_true", help="Overlay flow-direction arrows.")
    ap.add_argument("--label-arrows", action="store_true", help="Label arrows with speed/bearing.")
    ap.add_argument("--arrow-scale", type=float, default=0.6, help="Arrow length in degrees.")
    ap.add_argument("--match-top-n", type=int, default=1, help="Highlight top-N matches per hour.")
    ap.add_argument("--track-time-col", default=None)
    ap.add_argument("--track-lat-col", default=None)
    ap.add_argument("--track-lon-col", default=None)
    ap.add_argument("--track-id-col", default=None)
    ap.add_argument("--chunk-rows", dest="chunk_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", dest="chunk_rows", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", dest="parquet_rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    args = ap.parse_args()

    matches = _read_any(args.matches)
    if matches.empty:
        raise SystemExit("[maps] matches file is empty.")
    if "storm_id" not in matches.columns:
        raise SystemExit("[maps] matches file missing storm_id.")
    matches["track_time"] = pd.to_datetime(matches["track_time"], utc=True, errors="coerce").dt.tz_localize(None)
    matches["object_time"] = pd.to_datetime(matches["object_time"], utc=True, errors="coerce").dt.tz_localize(None)
    matches["obj_centroid_lat"] = pd.to_numeric(matches["obj_centroid_lat"], errors="coerce")
    matches["obj_centroid_lon"] = _norm_lon(matches["obj_centroid_lon"], args.normalize_lon)

    tracks = _read_any(args.tracks)
    colmap = _pick_track_cols(
        tracks,
        {"time": args.track_time_col, "lat": args.track_lat_col, "lon": args.track_lon_col, "id": args.track_id_col},
    )
    tracks = tracks.rename(columns={colmap["time"]: "time", colmap["lat"]: "lat", colmap["lon"]: "lon"})
    if colmap["id"] in tracks.columns:
        tracks = tracks.rename(columns={colmap["id"]: "storm_id"})
    else:
        tracks["storm_id"] = "storm"
    tracks["time"] = pd.to_datetime(tracks["time"], utc=True, errors="coerce").dt.tz_localize(None)
    tracks["lat"] = pd.to_numeric(tracks["lat"], errors="coerce")
    tracks["lon"] = _norm_lon(tracks["lon"], args.normalize_lon)
    tracks = tracks.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    objects = None
    if args.objects:
        objects = _read_any(args.objects)
        if not objects.empty:
            objects["time"] = pd.to_datetime(objects["time"], utc=True, errors="coerce").dt.tz_localize(None)
            objects["obj_centroid_lat"] = pd.to_numeric(objects["obj_centroid_lat"], errors="coerce")
            objects["obj_centroid_lon"] = _norm_lon(objects["obj_centroid_lon"], args.normalize_lon)
            objects = objects.dropna(subset=["time", "obj_centroid_lat", "obj_centroid_lon"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ccrs, cfeature = _load_cartopy()
    storms = sorted(matches["storm_id"].dropna().unique().tolist())
    for sid in storms:
        m = matches.loc[matches["storm_id"] == sid].copy()
        t = tracks.loc[tracks["storm_id"] == sid].copy()
        if t.empty or m.empty:
            continue
        genesis = t["time"].min()
        t0 = genesis - pd.Timedelta(hours=float(args.hours_before))
        t1 = genesis + pd.Timedelta(hours=float(args.hours_after))
        m = m.loc[(m["object_time"] >= t0) & (m["object_time"] <= t1)]
        t = t.loc[(t["time"] >= t0) & (t["time"] <= t1)]
        if m.empty or t.empty:
            continue

        objs = None
        if objects is not None and not objects.empty:
            objs = objects.loc[(objects["time"] >= t0) & (objects["time"] <= t1)].copy()
            lat_pad = float(args.objects_pad_deg)
            lon_pad = float(args.objects_pad_deg)
            lat_min = t["lat"].min() - lat_pad
            lat_max = t["lat"].max() + lat_pad
            lon_min = t["lon"].min() - lon_pad
            lon_max = t["lon"].max() + lon_pad
            objs = objs.loc[
                (objs["obj_centroid_lat"] >= lat_min)
                & (objs["obj_centroid_lat"] <= lat_max)
                & (objs["obj_centroid_lon"] >= lon_min)
                & (objs["obj_centroid_lon"] <= lon_max)
            ].reset_index(drop=True)
            score_col = args.objects_score_col
            if score_col not in (objs.columns if objs is not None else []):
                for cand in ["obj_score_topk_mean", "obj_score_max", "obj_score_mean"]:
                    if cand in objs.columns:
                        score_col = cand
                        break
            if score_col and score_col in objs.columns:
                vals = pd.to_numeric(objs[score_col], errors="coerce")
                if args.objects_min_score is not None:
                    objs = objs.loc[vals >= float(args.objects_min_score)].reset_index(drop=True)
                if args.objects_top_quantile is not None:
                    q = float(args.objects_top_quantile)
                    objs["__score__"] = vals
                    kept = []
                    for _, sub in objs.groupby(objs["time"].dt.floor("h"), sort=False):
                        svals = pd.to_numeric(sub["__score__"], errors="coerce")
                        if svals.notna().any():
                            thr = float(svals.quantile(q))
                            kept.append(sub.loc[svals >= thr])
                    objs = pd.concat(kept, ignore_index=True) if kept else objs.head(0)
                    objs = objs.drop(columns=["__score__"], errors="ignore")

        hours = sorted(m["object_time"].dt.floor("h").unique().tolist())
        if not args.per_hour:
            hours = [None]
        else:
            step = max(1, int(args.hour_step))
            if step > 1:
                hours = hours[::step]

        for h in hours:
            if h is None:
                subset = m
                title = f"Objects vs Track ({sid})"
                tag = "all"
                objs_h = objs
            else:
                subset = m.loc[m["object_time"].dt.floor("h") == h]
                title = f"Objects vs Track ({sid}) {pd.Timestamp(h).strftime('%Y-%m-%d %H:%M')}"
                tag = pd.Timestamp(h).strftime("%Y%m%d%H")
                objs_h = objs.loc[objs["time"].dt.floor("h") == h] if objs is not None else None

            if subset.empty and (objs_h is None or objs_h.empty):
                continue

            cand = subset.copy()
            if "match_rank" in cand.columns:
                match_mask = pd.to_numeric(cand["match_rank"], errors="coerce") <= int(args.match_top_n)
            elif "match_is_primary" in cand.columns:
                match_mask = pd.to_numeric(cand["match_is_primary"], errors="coerce").fillna(0).astype(int) > 0
            else:
                match_mask = np.ones(len(cand), dtype=bool)
            matched = cand.loc[match_mask].copy()

            if ccrs is None:
                import matplotlib.pyplot as plt  # type: ignore

                fig, ax = plt.subplots(figsize=(8, 6))
                _plot_one(
                    ax,
                    t,
                    objs_h,
                    cand,
                    matched,
                    title,
                    args.show_arrows,
                    args.show_flow_arrows,
                    args.arrow_scale,
                    args.label_arrows or args.show_arrows or args.show_flow_arrows,
                )
                fig.tight_layout()
                out_png = out_dir / f"{sid}_{tag}.png"
                fig.savefig(out_png, dpi=160)
                plt.close(fig)
                continue

            import matplotlib.pyplot as plt  # type: ignore

            fig = plt.figure(figsize=(8, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#e5e5e5", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
            _plot_one(
                ax,
                t,
                objs_h,
                cand,
                matched,
                title,
                args.show_arrows,
                args.show_flow_arrows,
                args.arrow_scale,
                args.label_arrows or args.show_arrows or args.show_flow_arrows,
            )
            fig.tight_layout()
            out_png = out_dir / f"{sid}_{tag}.png"
            fig.savefig(out_png, dpi=160)
            plt.close(fig)

    print(f"[maps] wrote per-storm maps to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
