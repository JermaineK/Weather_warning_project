#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_lead_skill.py — how many hours before a storm does a precursor signal
actually have skill?

For each storm in a tracks file, this crops two row-aligned-by-key grids over
the storm's window/box (predicate pushdown on `time`, so only a slice is read):

  * lead grid  (grid_slim_start): time, lat, lon, t_to_storm_min_h, near_storm
  * signal grid (state_slim):     time, lat, lon, <signal cols>

merges them on (time, lat, lon), and computes the AUC of each signal for
separating "a storm will reach this cell in L hours" (positives, binned by
t_to_storm_min_h) from far-from-storm background (negatives). Pooling across
storms, it plots AUC vs lead time and writes a CSV.

USAGE
    python eval_lead_skill.py \\
        --lead-grid   data/grid_slim_start.parquet \\
        --signal-grid data/grid_labelled_FMA_gka_realthermo_sph_ms_id_state_slim.parquet \\
        --tracks      data/tracks/tracks_subset.parquet \\
        --signals     zeta:neg,G_persist_24h,SFI \\
        --out-dir     figures/lead_skill
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


LEAD_EDGES = [0, 6, 12, 18, 24, 36, 48, 72, 96, 120]   # hours


def auc_roc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score(pos) > score(neg)) via the Mann-Whitney U statistic."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(len(cnt))
    np.add.at(tie_mean, inv, ranks)
    tie_mean /= cnt
    ranks = tie_mean[inv]
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * n0))


def parse_signals(spec: str):
    """'zeta:neg,G_persist_24h' -> [('zeta', True), ('G_persist_24h', False)]."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, flag = tok.split(":", 1)
            out.append((name.strip(), flag.strip().lower() in ("neg", "negate", "-")))
        else:
            out.append((tok, False))
    return out


def crop(path, cols, t0, t1, bbox):
    latS, latN, lonW, lonE = bbox
    df = pd.read_parquet(path, columns=cols,
                         filters=[("time", ">=", pd.Timestamp(t0)),
                                  ("time", "<", pd.Timestamp(t1))])
    df["time"] = pd.to_datetime(df["time"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").round(2)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce").round(2)
    df = df[(df["lat"].between(latS, latN)) & (df["lon"].between(lonW, lonE))]
    return df


def process_storm(tr_row_id, tr, args, signals, acc):
    g = tr[tr["storm_id"].astype(str) == str(tr_row_id)].copy()
    g["time"] = pd.to_datetime(g["time"])
    g = g.sort_values("time")
    name = str(g["name"].iloc[0]) if "name" in g.columns else str(tr_row_id)
    t0 = g["time"].min() - pd.Timedelta(hours=args.pre_h)
    t1 = g["time"].max() + pd.Timedelta(hours=6)
    pad = args.pad_deg
    bbox = (g["lat"].min() - pad, g["lat"].max() + pad,
            g["lon"].min() - pad, g["lon"].max() + pad)

    sig_cols = ["time", "lat", "lon"] + [s for s, _ in signals]
    lead = crop(args.lead_grid, ["time", "lat", "lon", "t_to_storm_min_h", "near_storm"],
                t0, t1, bbox)
    sig = crop(args.signal_grid, sig_cols, t0, t1, bbox)
    if lead.empty or sig.empty:
        print(f"[lead-skill] {name}: empty crop; skipping", file=sys.stderr)
        return
    m = lead.merge(sig, on=["time", "lat", "lon"], how="inner")
    if m.empty:
        print(f"[lead-skill] {name}: empty merge; skipping", file=sys.stderr)
        return

    is_pos = m["near_storm"].to_numpy() == 1
    lead_h = pd.to_numeric(m["t_to_storm_min_h"], errors="coerce").to_numpy()
    neg_mask = (m["near_storm"].to_numpy() == 0)

    # subsample negatives to bound memory (background is large & shared per bin)
    neg_idx = np.where(neg_mask)[0]
    if len(neg_idx) > args.max_neg:
        rng = np.random.default_rng(int(tr_row_id[-4:]) if str(tr_row_id)[-4:].isdigit() else 0)
        neg_idx = rng.choice(neg_idx, args.max_neg, replace=False)

    n_pos_total = 0
    for name_s, negate in signals:
        s = pd.to_numeric(m[name_s], errors="coerce").to_numpy(float)
        if negate:
            s = -s
        neg_scores = s[neg_idx]
        for lo, hi in zip(LEAD_EDGES[:-1], LEAD_EDGES[1:]):
            binmask = is_pos & (lead_h >= lo) & (lead_h < hi)
            pos_scores = s[binmask]
            key = (name_s, lo, hi)
            acc.setdefault(key, {"pos": [], "neg": []})
            acc[key]["pos"].append(pos_scores)
            acc[key]["neg"].append(neg_scores)
            if name_s == signals[0][0]:
                n_pos_total += int(binmask.sum())
    print(f"[lead-skill] {name}: merged={len(m):,}  pos(near_storm)={int(is_pos.sum()):,}  "
          f"neg_sampled={len(neg_idx):,}")


def main():
    args = parse_args()
    signals = parse_signals(args.signals)
    tr = pd.read_parquet(args.tracks)
    storm_ids = list(pd.unique(tr["storm_id"].astype(str)))
    if args.storm_id:
        storm_ids = [args.storm_id]
    print(f"[lead-skill] storms: {len(storm_ids)}  signals: {signals}")

    acc = {}
    for sid in storm_ids:
        process_storm(sid, tr, args, signals, acc)

    # compute AUC per (signal, lead bin)
    rows = []
    for (name_s, lo, hi), d in acc.items():
        pos = np.concatenate(d["pos"]) if d["pos"] else np.array([])
        neg = np.concatenate(d["neg"]) if d["neg"] else np.array([])
        rows.append({"signal": name_s, "lead_lo_h": lo, "lead_hi_h": hi,
                     "lead_mid_h": (lo + hi) / 2.0,
                     "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                     "auc": auc_roc(pos, neg)})
    res = pd.DataFrame(rows).sort_values(["signal", "lead_mid_h"]).reset_index(drop=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lead_skill.csv"
    res.to_csv(csv_path, index=False)
    print(f"\n[lead-skill] wrote {csv_path}")
    print(res.to_string(index=False))

    # plot AUC vs lead
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=120)
    for name_s, negate in signals:
        sub = res[res["signal"] == name_s]
        lbl = ("cyclonic " if negate else "") + name_s
        ax.plot(sub["lead_mid_h"], sub["auc"], "-o", label=lbl)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="no skill (0.5)")
    ax.invert_xaxis()  # lead decreases toward the storm on the right
    ax.set_xlabel("lead time before storm reaches cell (hours)")
    ax.set_ylabel("AUC (signal vs background)")
    ax.set_title("Precursor skill vs lead time (pooled over storms)")
    ax.grid(alpha=0.3); ax.legend()
    png_path = out_dir / "lead_skill_auc.png"
    fig.tight_layout(); fig.savefig(png_path); plt.close(fig)
    print(f"[lead-skill] wrote {png_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="AUC of precursor signals vs hours-before-storm.")
    ap.add_argument("--lead-grid", required=True, help="grid with time/lat/lon/t_to_storm_min_h/near_storm")
    ap.add_argument("--signal-grid", required=True, help="grid with time/lat/lon/<signal columns>")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--storm-id", default=None, help="limit to one storm_id (default: all in tracks)")
    ap.add_argument("--signals", default="zeta:neg,G_persist_24h,SFI",
                    help="comma list; append ':neg' to plot -signal (e.g. cyclonic vorticity)")
    ap.add_argument("--pre-h", type=float, default=120.0, help="hours before first track point to include")
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-neg", type=int, default=60000, help="max background negatives sampled per storm")
    ap.add_argument("--out-dir", default="figures/lead_skill")
    return ap.parse_args()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
