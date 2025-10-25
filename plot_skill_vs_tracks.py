#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_skill_vs_tracks.py
Overlay per-hour skill (coverage/F1) with storm-track presence.

- Reads CSV or Parquet (auto by extension).
- Accepts explicit --metrics-file OR auto-detects inside --dir.
- Detects hour column robustly (or use --hour-col).
- Normalizes common column name variants (coverage/F1/precision/recall).
- Optionally crops tracks to AOI and normalizes longitudes.
- Saves aligned table if requested and writes an overlay plot.

Examples (PowerShell):
  python .\plot_skill_vs_tracks.py `
    --lead-hours 24 `
    --metrics-file ".\results\per_hour\per_hour_skill_lead24.csv" `
    --tracks ".\data\besttrack_intensity.csv" `
    --out-dir ".\results\per_hour\plots" `
    --smooth-k 5 `
    --save-aligned-parquet ".\results\per_hour\aligned_lead24.parquet"
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------- tiny IO helpers -----------------------

def read_any(path: str, **kw) -> pd.DataFrame:
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, **kw)

def write_any(df: pd.DataFrame, path: str):
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in [".parquet", ".pq"]:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)

def to_utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def norm_lon_ser(x: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180


# ----------------------- autodetect helpers -----------------------

CAND_HOUR_NAMES = [
    "issue_hour", "issue_time", "issue", "time", "t", "hour",
    "t_issue", "t_issue_hour", "timestamp", "dt"
]

def autodetect_metrics(dir_path: str, lead: int) -> str | None:
    d = Path(dir_path)
    pats = [
        f"*metrics*lead{lead}*.parquet",
        f"*per_hour*lead{lead}*.parquet",
        f"*metrics*lead{lead}*.csv",
        f"*per_hour*lead{lead}*.csv",
    ]
    for pat in pats:
        hits = sorted(d.glob(pat))
        if hits:
            return str(hits[0])
    return None

def detect_hour_col(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit and explicit in df.columns:
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for nm in CAND_HOUR_NAMES:
        if nm in lower:
            return lower[nm]
    # fallback: sniff any column that parses to datetimes for most rows
    best = None
    best_frac = 0.0
    for c in df.columns:
        t = pd.to_datetime(df[c], utc=True, errors="coerce")
        frac = t.notna().mean()
        if frac >= 0.6 and t.nunique(dropna=True) >= 6:
            if frac > best_frac:
                best, best_frac = c, frac
    if best:
        return best
    raise ValueError("Could not detect an hour/time column. Use --hour-col to specify it explicitly.")

def pick_column(df: pd.DataFrame, prefer: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for nm in prefer:
        if nm in lower:
            return lower[nm]
    return None


# ----------------------- loaders -----------------------

def load_metrics(metrics_file: str | None,
                 coverage_file: str | None,
                 lead_hours: int,
                 auto_dir: str | None,
                 hour_col_override: str | None) -> pd.DataFrame:
    """Return dataframe indexed by 'issue_hour' with columns: coverage, f1 (optional)."""
    path = metrics_file or autodetect_metrics(auto_dir, lead_hours)
    if not path:
        if auto_dir:
            raise FileNotFoundError(f"No metrics file found in {auto_dir} for lead {lead_hours}h.")
        raise FileNotFoundError("Provide --metrics-file or --dir to find one.")

    m = read_any(path)
    hcol = detect_hour_col(m, hour_col_override)

    # coverage / f1 / precision / recall
    cov_col = pick_column(m, ["coverage", "cov", "cover"])
    f1_col  = pick_column(m, ["f1", "f_1", "fscore"])
    prec_col= pick_column(m, ["precision", "prec", "p"])
    rec_col = pick_column(m, ["recall", "rec", "r"])

    print(f"[metrics] using hour='{hcol}'  "
          f"coverage='{cov_col or '-'}'  f1='{f1_col or '-'}'  "
          f"precision='{prec_col or '-'}' recall='{rec_col or '-'}'  from {path}")

    out = pd.DataFrame({
        "issue_hour": to_utc_naive(m[hcol]).dt.floor("h"),
    })
    if cov_col: out["coverage"] = pd.to_numeric(m[cov_col], errors="coerce")
    if f1_col:  out["f1"]       = pd.to_numeric(m[f1_col],  errors="coerce")
    if prec_col:out["precision"]= pd.to_numeric(m[prec_col],errors="coerce")
    if rec_col: out["recall"]   = pd.to_numeric(m[rec_col], errors="coerce")

    # merge separate coverage if supplied
    if coverage_file:
        c = read_any(coverage_file)
        chcol = detect_hour_col(c, None)
        ccov = pick_column(c, ["coverage", "cov", "cover"])
        if ccov is None:
            raise ValueError(f"{coverage_file}: cannot find coverage column.")
        cover = pd.DataFrame({
            "issue_hour": to_utc_naive(c[chcol]).dt.floor("h"),
            "coverage": pd.to_numeric(c[ccov], errors="coerce")
        }).dropna(subset=["issue_hour"])
        out = out.merge(cover, on="issue_hour", how="left", suffixes=("", "_from_cover"))
        if "coverage_from_cover" in out and "coverage" in out:
            out["coverage"] = out["coverage_from_cover"].fillna(out["coverage"])
            out.drop(columns=["coverage_from_cover"], inplace=True)

    out = out.dropna(subset=["issue_hour"]).sort_values("issue_hour").reset_index(drop=True)
    return out

def load_tracks(path: str,
                normalize_lon: str = "none",
                area: str | None = None) -> pd.DataFrame:
    tr = pd.read_csv(path)
    tcol = "obs_time" if "obs_time" in tr.columns else ("time" if "time" in tr.columns else None)
    if tcol is None:
        raise ValueError(f"{path}: need obs_time/time column.")
    for req in ["lat","lon"]:
        if req not in tr.columns:
            raise ValueError(f"{path}: missing column '{req}'")
    out = pd.DataFrame({
        "obs_time": to_utc_naive(tr[tcol]),
        "lat": pd.to_numeric(tr["lat"], errors="coerce"),
        "lon": norm_lon_ser(tr["lon"], normalize_lon)
    }).dropna().reset_index(drop=True)

    if area:
        latN, lonW, latS, lonE = [float(x.strip()) for x in area.split(",")]
        out = out.loc[
            (out["lat"] <= latN) & (out["lat"] >= latS) &
            (out["lon"] >= lonW) & (out["lon"] <= lonE)
        ].reset_index(drop=True)

    out["hour"] = out["obs_time"].dt.floor("h")
    print(f"[tracks] rows={len(out)}  hours={out['hour'].nunique()}")
    return out


# ----------------------- align + plot -----------------------

def align_skill_tracks(skill: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    hours = skill["issue_hour"].drop_duplicates().sort_values()
    pres = tracks.groupby("hour").size().reindex(hours, fill_value=0).rename("track_count")
    out = skill.set_index("issue_hour").join(pres, how="left")
    out["track_count"] = out["track_count"].fillna(0).astype(int)
    out["track_presence"] = (out["track_count"] > 0).astype(int)
    out = out.reset_index().rename(columns={"index":"issue_hour"})
    return out

def smooth_series(y: pd.Series, k: int) -> pd.Series:
    if k is None or k <= 1:
        return y
    return y.rolling(window=int(k), min_periods=1, center=True).mean()

def make_overlay_plot(df: pd.DataFrame, lead: int, out_png: str):
    t = df["issue_hour"]
    cov = df["coverage"] if "coverage" in df else None
    f1  = df["f1"] if "f1" in df else None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(f"Skill vs Tracks (lead {lead}h)")
    ax1.set_xlabel("Issue hour")
    ax1.set_ylabel("Coverage")

    if cov is not None:
        ax1.plot(t, cov, lw=1.2, label="Coverage", alpha=0.9)

    ax2 = None
    if f1 is not None and f1.notna().any():
        ax2 = ax1.twinx()
        ax2.set_ylabel("F1")
        ax2.plot(t, f1, lw=1.0, ls="--", label="F1", alpha=0.9)

    # Shade track presence
    if "track_presence" in df:
        for hh, pres in zip(t, df["track_presence"].to_numpy()):
            if pres:
                ax1.axvspan(hh, hh + pd.Timedelta(hours=1), color="k", alpha=0.06)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay per-hour skill with storm track presence.")
    ap.add_argument("--lead-hours", type=int, required=True)

    ap.add_argument("--metrics-file", default=None, help="CSV/Parquet with per-hour skill.")
    ap.add_argument("--coverage-file", default=None, help="Optional coverage file if separate.")
    ap.add_argument("--dir", default=None, help="Auto-detect metrics inside this folder.")
    ap.add_argument("--hour-col", default=None, help="Explicit hour/time column in metrics file.")

    ap.add_argument("--tracks", required=True, help="CSV from prepare_besttrack_intensity.py")
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help='Optional AOI "latN,lonW,latS,lonE"')

    ap.add_argument("--out-dir", default="results/per_hour/plots")
    ap.add_argument("--smooth-k", type=int, default=0, help="Rolling window (hours) for smoothing coverage/F1")
    ap.add_argument("--save-aligned-parquet", default=None)
    ap.add_argument("--save-aligned-csv", default=None)
    args = ap.parse_args()

    metrics = load_metrics(args.metrics_file, args.coverage_file,
                           args.lead_hours, args.dir, args.hour_col)

    for c in ["coverage", "f1"]:
        if c in metrics.columns:
            metrics[c] = smooth_series(metrics[c], args.smooth_k)

    tracks = load_tracks(args.tracks, normalize_lon=args.normalize_lon, area=args.area)
    aligned = align_skill_tracks(metrics, tracks)

    if args.save_aligned_parquet:
        write_any(aligned, args.save_aligned_parquet)
    if args.save_aligned_csv:
        write_any(aligned, args.save_aligned_csv)

    out_png = os.path.join(args.out_dir, f"skill_vs_tracks_lead{args.lead_hours}.png")
    make_overlay_plot(aligned, args.lead_hours, out_png)
    print(f"[plot] wrote {out_png}")
    if args.save_aligned_parquet:
        print(f"[plot] aligned → {args.save_aligned_parquet}")
    if args.save_aligned_csv:
        print(f"[plot] aligned → {args.save_aligned_csv}")

if __name__ == "__main__":
    main()