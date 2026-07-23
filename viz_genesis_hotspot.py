#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_genesis_hotspot.py — animate a genesis-precursor "hotspot" signal over a
lat/lon grid and overlay a real storm track, so you can see WHERE the hotspot
forms and HOW LONG before the storm.

The grid file is read with predicate pushdown on `time` so only the storm's
window is loaded (not the whole multi-GB file). Output is a set of PNG frames
plus an assembled GIF.

USAGE
    python viz_genesis_hotspot.py \\
        --grid  data/grid_labelled_FMA_gka_realthermo_sph_ms_id_state_slim.parquet \\
        --tracks data/tracks/tracks_subset.parquet \\
        --storm-id 2025052S14148 \\
        --signal G_persist_24h \\
        --start 2025-02-19 --end 2025-03-01 --step-h 3 \\
        --out-dir figures/alfred_genesis_hotspot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter


# ---------------- helpers ----------------

def read_track(tracks_path: str, storm_id: str) -> pd.DataFrame:
    tr = pd.read_parquet(tracks_path)
    tr = tr[tr["storm_id"].astype(str) == str(storm_id)].copy()
    if tr.empty:
        raise SystemExit(f"[viz] storm_id {storm_id} not found in {tracks_path}")
    tr["time"] = pd.to_datetime(tr["time"])
    tr = tr.sort_values("time").reset_index(drop=True)
    for c in ("lat", "lon", "vmax"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    return tr


def genesis_time(tr: pd.DataFrame, thresh_kt: float = 34.0) -> pd.Timestamp:
    """First time sustained wind reaches TC intensity (default 34 kt)."""
    g = tr.dropna(subset=["vmax"])
    g = g[g["vmax"] >= thresh_kt]
    return g["time"].iloc[0] if len(g) else tr["time"].iloc[0]


def interp_track_to(tr: pd.DataFrame, when: pd.Timestamp):
    """Linear-interpolate storm lat/lon/vmax to an arbitrary time (or None if
    outside the track's own time span)."""
    if when < tr["time"].iloc[0] or when > tr["time"].iloc[-1]:
        return None
    ts = tr["time"].astype("int64").to_numpy()
    w = np.int64(when.value)
    lat = float(np.interp(w, ts, tr["lat"].to_numpy()))
    lon = float(np.interp(w, ts, tr["lon"].to_numpy()))
    vmax = float(np.interp(w, ts, tr["vmax"].ffill().bfill().to_numpy()))
    return lat, lon, vmax


def load_window(grid_path: str, signal: str, start, end, bbox):
    latS, latN, lonW, lonE = bbox
    cols = ["time", "lat", "lon", signal]
    df = pd.read_parquet(
        grid_path, columns=cols,
        filters=[("time", ">=", pd.Timestamp(start)), ("time", "<", pd.Timestamp(end))],
    )
    df["time"] = pd.to_datetime(df["time"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df[signal] = pd.to_numeric(df[signal], errors="coerce")
    df = df[(df["lat"].between(latS, latN)) & (df["lon"].between(lonW, lonE))]
    return df.dropna(subset=["lat", "lon"])


def _smooth_nan(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a grid that may contain NaNs (NaN-aware normalisation)."""
    if sigma <= 0:
        return grid
    mask = np.isfinite(grid).astype(float)
    filled = np.where(np.isfinite(grid), grid, 0.0)
    num = gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask, sigma=sigma, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)
    out[mask == 0] = np.nan
    return out


def pivot_frame(fr: pd.DataFrame, signal: str, lats: np.ndarray, lons: np.ndarray):
    """Build a (nlat, nlon) grid of the signal for one timestamp."""
    grid = np.full((len(lats), len(lons)), np.nan)
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    li = fr["lat"].round(2).map(lat_idx)
    lj = fr["lon"].round(2).map(lon_idx)
    ok = li.notna() & lj.notna()
    grid[li[ok].astype(int).to_numpy(), lj[ok].astype(int).to_numpy()] = \
        fr[signal][ok].to_numpy()
    return grid


# ---------------- main render ----------------

def render(args):
    tr = read_track(args.tracks, args.storm_id)
    name = str(tr["name"].iloc[0]) if "name" in tr.columns else args.storm_id
    gen_t = genesis_time(tr, args.genesis_thresh)
    print(f"[viz] {name} ({args.storm_id})  genesis(>= {args.genesis_thresh:.0f}kt) = {gen_t}")

    # bbox: pad around the track unless user forced one
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
    else:
        pad = args.pad_deg
        bbox = (tr["lat"].min() - pad, tr["lat"].max() + pad,
                tr["lon"].min() - pad, tr["lon"].max() + pad)
    latS, latN, lonW, lonE = bbox
    print(f"[viz] bbox lat[{latS:.1f},{latN:.1f}] lon[{lonW:.1f},{lonE:.1f}]")

    df = load_window(args.grid, args.signal, args.start, args.end, bbox)
    if df.empty:
        raise SystemExit("[viz] no grid rows in window/bbox.")
    if args.negate:
        df[args.signal] = -df[args.signal]
    print(f"[viz] loaded {len(df):,} grid rows for signal '{args.signal}'"
          f"{' (negated)' if args.negate else ''}")

    label = ("cyclonic " if args.negate else "") + args.signal

    lats = np.array(sorted(df["lat"].round(2).unique()))
    lons = np.array(sorted(df["lon"].round(2).unique()))

    # robust colour scale from the whole window (2–98 pct)
    vlo, vhi = np.nanpercentile(df[args.signal], [2, 98])
    if not np.isfinite(vlo): vlo = 0.0
    if not np.isfinite(vhi) or vhi <= vlo: vhi = vlo + 1e-6

    # frame times
    frames = pd.date_range(args.start, args.end, freq=f"{args.step_h}h", inclusive="left")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    groups = {t: g for t, g in df.groupby("time")}

    for k, ft in enumerate(frames):
        fr = groups.get(ft)
        if fr is None:
            # nearest available timestamp within step
            near = df["time"].iloc[(df["time"] - ft).abs().argsort()[:1]]
            if len(near):
                fr = groups.get(near.iloc[0])
        if fr is None or fr.empty:
            continue
        grid = pivot_frame(fr, args.signal, lats, lons)
        grid = _smooth_nan(grid, args.smooth_sigma)

        fig, ax = plt.subplots(figsize=(8, 6.2), dpi=110)
        im = ax.imshow(grid, origin="lower",
                       extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                       aspect="auto", cmap="inferno", vmin=vlo, vmax=vhi)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label)

        # storm track: past (solid) + current position (star)
        past = tr[tr["time"] <= ft]
        if len(past) >= 2:
            ax.plot(past["lon"], past["lat"], "-", color="#39d0ff", lw=1.6, alpha=0.9)
        pos = interp_track_to(tr, ft)
        if pos is not None:
            plat, plon, pv = pos
            ax.scatter([plon], [plat], s=60 + 3.0 * max(pv, 0), marker="*",
                       color="#39d0ff", edgecolor="white", linewidth=0.8, zorder=5)
            ax.text(plon + 0.4, plat + 0.4, f"{name}\n{pv:.0f} kt",
                    color="white", fontsize=8, va="bottom")
        # genesis location marker
        gpos = interp_track_to(tr, gen_t)
        if gpos is not None:
            ax.scatter([gpos[1]], [gpos[0]], s=90, marker="o",
                       facecolor="none", edgecolor="#7CFC00", linewidth=1.6, zorder=4)

        dt_h = (ft - gen_t) / pd.Timedelta(hours=1)
        lead = f"T{dt_h:+.0f} h to genesis" if dt_h < 0 else f"T{dt_h:+.0f} h (post-genesis)"
        ax.set_title(f"{name} — {ft:%Y-%m-%d %H:%M} UTC   |   {lead}", fontsize=11)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_xlim(lons.min(), lons.max()); ax.set_ylim(lats.min(), lats.max())

        fp = out_dir / f"frame_{k:03d}.png"
        fig.tight_layout(); fig.savefig(fp); plt.close(fig)
        frame_paths.append(fp)

    if not frame_paths:
        raise SystemExit("[viz] no frames rendered.")

    gif_path = out_dir / f"{name.lower()}_{args.signal}.gif"
    with imageio.get_writer(gif_path, mode="I", duration=1.0 / args.fps, loop=0) as w:
        for fp in frame_paths:
            w.append_data(imageio.imread(fp))
    print(f"[viz] wrote {len(frame_paths)} frames + {gif_path}")
    return gif_path


def parse_args():
    ap = argparse.ArgumentParser(description="Animate a genesis-hotspot signal with a storm track overlay.")
    ap.add_argument("--grid", required=True, help="Grid parquet with time/lat/lon/<signal>.")
    ap.add_argument("--tracks", required=True, help="Tracks parquet (storm_id/time/lat/lon/vmax/name).")
    ap.add_argument("--storm-id", required=True)
    ap.add_argument("--signal", default="G_persist_24h")
    ap.add_argument("--negate", action="store_true",
                    help="Plot -signal (e.g. cyclonic vorticity = -zeta in the Southern Hemisphere).")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="Gaussian spatial smoothing in grid cells (NaN-aware). 0 = off.")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-h", type=int, default=3)
    ap.add_argument("--bbox", default=None, help="latS,latN,lonW,lonE (default: pad around track).")
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--genesis-thresh", type=float, default=34.0)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--out-dir", default="figures/genesis_hotspot")
    return ap.parse_args()


def main():
    render(parse_args())


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
