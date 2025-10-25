#!/usr/bin/env python3
"""
plot_seed_track_matches.py
Visualize seed–track correspondences:
  - map of seed vs. matched track points
  - lead-time histograms
Usage:
  python plot_seed_track_matches.py --matches data/seeds_analysis_wide/seed_track_matches.csv
"""

import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.lines as mlines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True, help="CSV with matched seeds and tracks")
    ap.add_argument("--outdir", default="results/maps", help="Directory for plots")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.matches)
    if df.empty:
        print("No matches found in file.")
        return

    # Normalize lat/lon numeric and filter extremes
    for c in ["lat_cen","lon_cen","storm_lat","storm_lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat_cen","lon_cen"]).reset_index(drop=True)

    # ---- Map of matches ----
    plt.figure(figsize=(7.2,5.8))
    plt.title("Seed–Track Matches")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.grid(True, ls=":", alpha=0.5)

    # plot connecting lines
    if {"storm_lat","storm_lon"}.issubset(df.columns):
        for _,r in df.iterrows():
            plt.plot([r["lon_cen"], r["storm_lon"]],
                     [r["lat_cen"], r["storm_lat"]],
                     color="gray", lw=0.5, alpha=0.6)

    # scatter seeds and tracks
    plt.scatter(df["lon_cen"], df["lat_cen"], s=15, c="tab:blue", label="Seed")
    if {"storm_lat","storm_lon"}.issubset(df.columns):
        plt.scatter(df["storm_lon"], df["storm_lat"], s=30, c="tab:red", marker="x", label="Track")

    plt.legend(frameon=False)
    plt.tight_layout()
    map_path = outdir / "seed_track_map.png"
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"[map] saved → {map_path}")

    # ---- Lead-time histograms ----
    leads = {}
    for col,label in [("lead_to_25h","≥25 kt"),("lead_to_34h","≥34 kt"),("lead_to_64h","≥64 kt")]:
        if col in df.columns:
            leads[label] = pd.to_numeric(df[col], errors="coerce").dropna()

    if leads:
        plt.figure(figsize=(7,4.5))
        for i,(lbl,v) in enumerate(leads.items()):
            plt.hist(v, bins=np.arange(-12,60,3), alpha=0.6, label=f"{lbl} (n={len(v)})")
        plt.axvline(0, color="k", lw=0.8)
        plt.xlabel("Lead time from seed start (hours)")
        plt.ylabel("Count")
        plt.title("Lead-time distribution by intensity threshold")
        plt.legend()
        plt.tight_layout()
        hist_path = outdir / "leadtime_histograms.png"
        plt.savefig(hist_path, dpi=200)
        plt.close()
        print(f"[hist] saved → {hist_path}")

if __name__ == "__main__":
    main()