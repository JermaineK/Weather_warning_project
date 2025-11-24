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
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def main():
    ap = argparse.ArgumentParser(description="Plot seed–track matches on a map.")
    ap.add_argument("--matches", required=True, help="CSV/Parquet with seed–track matches.")
    ap.add_argument("--out", default="results/maps/seed_track_map.png", help="Output image path.")
    ap.add_argument("--lat-range", nargs=2, type=float, default=None, help="Optional map latitude range.")
    ap.add_argument("--lon-range", nargs=2, type=float, default=None, help="Optional map longitude range.")
    ap.add_argument("--overlay-prob", default=None, help="Optional numeric column for coloring seeds (e.g., prob).")
    ap.add_argument("--use-tiles", action="store_true", help="Add background tiles (requires internet).")
    ap.add_argument("--dpi", type=int, default=200)
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

    df = df.dropna(subset=[lat_c, lon_c]).reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No valid coordinate rows to plot.")

    # Extent
    if args.lon_range and args.lat_range:
        lon_min, lon_max = args.lon_range
        lat_min, lat_max = args.lat_range
    else:
        lon_min, lon_max = float(df[lon_c].min()) - 1, float(df[lon_c].max()) + 1
        lat_min, lat_max = float(df[lat_c].min()) - 1, float(df[lat_c].max()) + 1

    # --- Plot ---
    if not CARTOPY_AVAILABLE:
        print("[warn] Cartopy not available — falling back to plain scatter.")
        plt.figure(figsize=(8,6))
        plt.scatter(df[lon_c], df[lat_c], s=12, c="tab:blue", alpha=0.6, label="Seeds")
        if lat_t and lon_t:
            plt.scatter(df[lon_t], df[lat_t], s=20, c="tab:red", marker="x", label="Tracks")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Seed–Track Matches (no map projection)")
        plt.legend()
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        plt.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=args.dpi)
        plt.close()
        print(f"[map] saved (simple) → {args.out}")
        return

    # cartopy path
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

    # --- Seed plotting ---
    if args.overlay_prob and args.overlay_prob in df.columns:
        vals = pd.to_numeric(df[args.overlay_prob], errors="coerce")
        sc = ax.scatter(df[lon_c], df[lat_c], s=15, c=vals, cmap="viridis",
                        transform=ccrs.PlateCarree(), label="Seeds", alpha=0.8)
        cb = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7)
        cb.set_label(args.overlay_prob)
    else:
        ax.scatter(df[lon_c], df[lat_c], s=15, color="tab:blue", transform=ccrs.PlateCarree(),
                   alpha=0.7, label="Seeds")

    # --- Track overlay + connectors ---
    if lat_t and lon_t:
        ax.scatter(df[lon_t], df[lat_t], s=25, color="tab:red", marker="x",
                   transform=ccrs.PlateCarree(), label="Track")
        # connecting lines
        for _, r in df.iterrows():
            ax.plot([r[lon_c], r[lon_t]], [r[lat_c], r[lat_t]],
                    color="gray", lw=0.5, alpha=0.5, transform=ccrs.PlateCarree())

    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Seed–Track Matches")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"[map] saved → {args.out}")


if __name__ == "__main__":
    main()