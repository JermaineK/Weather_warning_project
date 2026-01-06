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
"""

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from utils import join_audit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------- IO helpers -----------------------

def read_any(path: str, **kw) -> pd.DataFrame:
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(p, **kw)
    return pd.read_csv(p, **kw)

def write_any(df: pd.DataFrame, path: str):
    p = str(path)
    ext = Path(p).suffix.lower()
    comp = "gzip" if ext.endswith(".gz") else None
    if ext in [".parquet", ".pq"]:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False, compression=comp)

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
    "issue_hour","issue_time","issue","time","t","hour",
    "t_issue","t_issue_hour","timestamp","dt"
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
    # fallback: sniff datetime-like
    best, best_frac = None, 0.0
    for c in df.columns:
        t = pd.to_datetime(df[c], utc=True, errors="coerce")
        frac = t.notna().mean()
        if frac >= 0.6 and t.nunique(dropna=True) >= 6 and frac > best_frac:
            best, best_frac = c, frac
    if best:
        return best
    raise ValueError("Could not detect an hour/time column; use --hour-col.")

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
    """Return dataframe indexed by 'issue_hour' with coverage/f1/precision/recall."""
    path = metrics_file or (autodetect_metrics(auto_dir, lead_hours) if auto_dir else None)
    if not path:
        raise FileNotFoundError("Provide --metrics-file or --dir for auto-detect.")

    m = read_any(path)
    hcol = detect_hour_col(m, hour_col_override)

    cov_col = pick_column(m, ["coverage","cov","cover"])
    f1_col  = pick_column(m, ["f1","f_1","fscore"])
    prec_col= pick_column(m, ["precision","prec","p"])
    rec_col = pick_column(m, ["recall","rec","r"])

    print(f"[metrics] using hour='{hcol}' cov='{cov_col or '-'}' f1='{f1_col or '-'}' "
          f"prec='{prec_col or '-'}' rec='{rec_col or '-'}'  ({Path(path).name})")

    out = pd.DataFrame({"issue_hour": to_utc_naive(m[hcol]).dt.floor("h")})
    for name, col in [("coverage", cov_col), ("f1", f1_col),
                      ("precision", prec_col), ("recall", rec_col)]:
        if col:
            out[name] = pd.to_numeric(m[col], errors="coerce")

    # optional coverage supplement
    if coverage_file:
        c = read_any(coverage_file)
        chcol = detect_hour_col(c, None)
        ccov = pick_column(c, ["coverage","cov","cover"])
        if not ccov:
            raise ValueError(f"{coverage_file}: no coverage column.")
        cover = pd.DataFrame({
            "issue_hour": to_utc_naive(c[chcol]).dt.floor("h"),
            "coverage": pd.to_numeric(c[ccov], errors="coerce")
        }).dropna(subset=["issue_hour"])
        left_df = out
        out = out.merge(cover, on="issue_hour", how="left", suffixes=("", "_cfile"))
        left_dupe = int(left_df.duplicated(subset=["issue_hour"]).sum())
        right_dupe = int(cover.duplicated(subset=["issue_hour"]).sum())
        unmatched = join_audit.estimate_unmatched_keys(left_df, cover, ["issue_hour"])
        entry = join_audit.build_entry(
            step="eval.skill-vs-tracks.coverage-merge",
            keys=["issue_hour"],
            join_type="left",
            left_rows=len(left_df),
            right_rows=len(cover),
            out_rows=len(out),
            left_dupe_keys=left_dupe,
            right_dupe_keys=right_dupe,
            left_key_count=unmatched.get("left_key_count"),
            right_key_count=unmatched.get("right_key_count"),
            left_unmatched_keys=unmatched.get("left_unmatched_keys"),
            right_unmatched_keys=unmatched.get("right_unmatched_keys"),
            unmatched_sampled=unmatched.get("unmatched_sampled"),
            extra={"coverage_file": str(coverage_file)},
        )
        join_audit.append_entry(join_audit.default_path(), entry)
        if "coverage_cfile" in out:
            out["coverage"] = out["coverage_cfile"].fillna(out.get("coverage"))
            out.drop(columns=["coverage_cfile"], inplace=True)

    out = out.dropna(subset=["issue_hour"]).sort_values("issue_hour").reset_index(drop=True)
    return out


def load_tracks(path: str,
                normalize_lon: str = "none",
                area: str | None = None) -> pd.DataFrame:
    tr = pd.read_csv(path)
    tcol = "obs_time" if "obs_time" in tr.columns else pick_column(tr, ["time","datetime","date"])
    if not tcol:
        raise ValueError(f"{path}: need obs_time/time column.")
    for req in ["lat","lon"]:
        if req not in tr.columns:
            raise ValueError(f"{path}: missing '{req}' column.")
    out = pd.DataFrame({
        "obs_time": to_utc_naive(tr[tcol]),
        "lat": pd.to_numeric(tr["lat"], errors="coerce"),
        "lon": norm_lon_ser(tr["lon"], normalize_lon)
    }).dropna(subset=["obs_time","lat","lon"]).reset_index(drop=True)

    if area:
        latN, lonW, latS, lonE = [float(x.strip()) for x in area.split(",")]
        out = out[(out["lat"] <= latN) & (out["lat"] >= latS)]
        if lonW <= lonE:
            out = out[(out["lon"] >= lonW) & (out["lon"] <= lonE)]
        else:
            out = out[(out["lon"] >= lonW) | (out["lon"] <= lonE)]

    out["hour"] = out["obs_time"].dt.floor("h")
    print(f"[tracks] rows={len(out):,} hours={out['hour'].nunique()}")
    return out


# ----------------------- align + plot -----------------------

def align_skill_tracks(skill: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    hours = skill["issue_hour"].drop_duplicates().sort_values()
    pres = tracks.groupby("hour").size().reindex(hours, fill_value=0).rename("track_count")
    out = skill.set_index("issue_hour").join(pres, how="left")
    out["track_count"] = out["track_count"].fillna(0).astype(int)
    out["track_presence"] = (out["track_count"] > 0).astype(int)
    return out.reset_index()

def smooth_series(y: pd.Series, k: int) -> pd.Series:
    if not k or k <= 1:
        return y
    return y.rolling(window=int(k), min_periods=1, center=True).mean()

def make_overlay_plot(df: pd.DataFrame, lead: int, out_png: str):
    t = df["issue_hour"]
    cov = df.get("coverage")
    f1  = df.get("f1")

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.set_title(f"Skill vs Tracks (lead {lead}h)")
    ax1.set_xlabel("Issue hour (UTC)")
    ax1.set_ylabel("Coverage", color="tab:blue")

    if cov is not None:
        ax1.plot(t, cov, lw=1.2, label="Coverage", color="tab:blue")

    ax2 = None
    if f1 is not None and f1.notna().any():
        ax2 = ax1.twinx()
        ax2.set_ylabel("F1", color="tab:orange")
        ax2.plot(t, f1, lw=1.0, ls="--", color="tab:orange", label="F1")

    # Shade hours with tracks
    if "track_presence" in df.columns:
        for hh, pres in zip(t, df["track_presence"].to_numpy()):
            if pres:
                ax1.axvspan(hh, hh + pd.Timedelta(hours=1), color="k", alpha=0.05)

    fig.autofmt_xdate()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Overlay per-hour skill with storm track presence.")
    ap.add_argument("--lead-hours", type=int, required=True)
    ap.add_argument("--metrics-file", default=None)
    ap.add_argument("--coverage-file", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--hour-col", default=None)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help='AOI "latN,lonW,latS,lonE"')
    ap.add_argument("--out-dir", default="results/per_hour/plots")
    ap.add_argument("--smooth-k", type=int, default=0)
    ap.add_argument("--save-aligned-parquet", default=None)
    ap.add_argument("--save-aligned-csv", default=None)
    args = ap.parse_args()

    metrics = load_metrics(args.metrics_file, args.coverage_file,
                           args.lead_hours, args.dir, args.hour_col)
    for c in ["coverage","f1"]:
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
    print(f"[aligned] hours={aligned['issue_hour'].nunique()} "
          f"track_hours={aligned['track_presence'].sum()} "
          f"avg_coverage={aligned.get('coverage',pd.Series()).mean():.3f}")

if __name__ == "__main__":
    main()
