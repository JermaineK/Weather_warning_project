#!/usr/bin/env python3
"""
plot_seed_track_map_cartopy.py
Plot seed vs track matches on a basemap with coastlines.
Requires: cartopy, matplotlib, pandas
"""

import argparse, pandas as pd, matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, help="CSV with seed–track matches")
    ap.add_argument("--out", default="results/maps/seed_track_map_basemap.png", help="Output image path")
    ap.add_argument("--lon-range", nargs=2, type=float, default=[130, 160])
    ap.add_argument("--lat-range", nargs=2, type=float, default=[-25, -5])
    args = ap.parse_args()

    df = pd.read_csv(args.matches)
    for c in ["lat_cen","lon_cen","storm_lat","storm_lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat_cen","lon_cen"]).reset_index(drop=True)

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([args.lon_range[0], args.lon_range[1], args.lat_range[0], args.lat_range[1]])

    # add geography
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # plot seeds and track points
    ax.scatter(df["lon_cen"], df["lat_cen"], s=15, color="tab:blue", transform=ccrs.PlateCarree(), label="Seed")

    if {"storm_lat","storm_lon"}.issubset(df.columns):
        ax.scatter(df["storm_lon"], df["storm_lat"], s=25, color="tab:red", marker="x",
                   transform=ccrs.PlateCarree(), label="Track")
        for _,r in df.iterrows():
            ax.plot([r["lon_cen"], r["storm_lon"]],
                    [r["lat_cen"], r["storm_lat"]],
                    color="gray", lw=0.5, alpha=0.6, transform=ccrs.PlateCarree())

    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Seed–Track Matches with Coastlines")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[map] saved → {args.out}")

if __name__ == "__main__":
    main()