#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_early_warning_loso.py — hardened early-warning trigger.

Removes the in-sample optimism of eval_early_warning.py:
  * LEAVE-ONE-STORM-OUT: for each held-out storm, the curated model AND the
    standardisation are fit on the OTHER storms only, then used to score the
    held-out storm's cells (instantaneous s and causal trailing A24/A48).
    Every scored cell is thus predicted by a model that never saw its storm.
  * STORM-LEVEL BLOCK BOOTSTRAP: because the long-lead positive bins are small
    and cells within a storm are correlated, 95% CIs are built by resampling
    whole storms (not cells) with replacement.

Reports, per warning lead L and feature in {instantaneous, trailing-24h,
trailing-48h}, the held-out AUC with a storm-block-bootstrap 95% CI, so the
"accumulation wins at 24-48 h" claim can be judged against sampling noise.

USAGE
    python eval_early_warning_loso.py \\
        --gka-grid    data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet \\
        --pregen-grid data/gse_panel_70m_slim.parquet \\
        --lead-grid   data/grid_slim_start.parquet \\
        --tracks      data/tracks/tracks_subset.parquet \\
        --out-dir     figures/early_warning_loso
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_spiral_genesis import auc_roc, fit_logistic, predict
from eval_early_warning import (
    load_storm_crops, fit_curated, add_scores,
    CURATED, LEADS, WINDOWS, LEAD_TOL,
)

FEATS_OUT = ["s"] + [f"A{W}" for W in WINDOWS]


def auc_pos_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    # eval_spiral_genesis.auc_roc already takes (positive_scores, negative_scores)
    # as a Mann-Whitney rank AUC — pass them straight through.
    return auc_roc(pos, neg)


def main():
    args = parse_args()
    feats = CURATED
    crops = load_storm_crops(args, feats)
    storms = list(crops.keys())
    if len(storms) < 4:
        raise SystemExit("[ew-loso] need >= 4 storms.")

    # Per storm, held-out arrays: neg[feature], pos[L][feature]
    held = {s: {"neg": {c: None for c in FEATS_OUT},
                "pos": {L: {c: None for c in FEATS_OUT} for L in LEADS}}
            for s in storms}
    rng = np.random.default_rng(1)

    for s in storms:
        train = {k: v for k, v in crops.items() if k != s}
        w, mu, sd = fit_curated(train, feats)          # fit WITHOUT held-out storm
        m = add_scores(crops[s].copy(), feats, w, mu, sd)
        sp = m[m["pregen"] == 1]
        fiz = sp[sp["near_storm"] == 0]
        if len(fiz) > args.max_neg:
            fiz = fiz.iloc[rng.choice(len(fiz), args.max_neg, replace=False)]
        for c in FEATS_OUT:
            held[s]["neg"][c] = fiz[c].to_numpy(float)
        tight = sp[sp["near_storm"] == 1]
        tt = pd.to_numeric(tight["t_to_storm_min_h"], errors="coerce").to_numpy(float)
        for L in LEADS:
            mask = (tt >= L - LEAD_TOL) & (tt <= L + LEAD_TOL)
            for c in FEATS_OUT:
                held[s]["pos"][L][c] = tight[c].to_numpy(float)[mask]
        npos = {L: len(held[s]['pos'][L]['s']) for L in LEADS}
        print(f"[ew-loso] held-out {s:12s} fit on {len(train)} storms  "
              f"neg={len(fiz):,}  pos@L={npos}")

    # Per-storm AUC is the correct unit: each held-out storm is scored by its OWN
    # leave-out model, so AUC (scale-invariant) is valid within a storm. Pooling
    # raw scores across different models is NOT valid. We therefore compute an AUC
    # per held-out storm (only where that storm has >= min_bin positives at the
    # lead), then aggregate across storms with a storm-block bootstrap.
    MIN_BIN = args.min_bin

    def per_storm_aucs(L, c):
        """List of (storm, auc) for storms with enough positives at lead L."""
        out = []
        for s in storms:
            p = held[s]["pos"][L][c]
            if len(p[np.isfinite(p)]) >= MIN_BIN:
                out.append((s, auc_pos_neg(p, held[s]["neg"][c])))
        return out

    B = args.n_boot
    rows = []
    for L in LEADS:
        for c in FEATS_OUT:
            sa = per_storm_aucs(L, c)
            names = [s for s, _ in sa]; vals = np.array([a for _, a in sa], float)
            point = float(np.nanmean(vals)) if len(vals) else float("nan")
            if len(vals) >= 2:
                bmeans = [np.nanmean(vals[rng.integers(0, len(vals), size=len(vals))])
                          for _ in range(B)]
                lo, hi = float(np.percentile(bmeans, 2.5)), float(np.percentile(bmeans, 97.5))
            else:
                lo = hi = float("nan")
            rows.append({"lead_h": L, "feature": c, "auc_mean": point,
                         "ci_lo": lo, "ci_hi": hi, "n_storms": len(vals),
                         "contributing": ",".join(names)})
    res = pd.DataFrame(rows)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "early_warning_loso.csv", index=False)

    print("\n[ew-loso] per-storm-mean held-out AUC (storm-block bootstrap 95% CI); "
          f"n = #storms with >= {MIN_BIN} positives at that lead:")
    for L in LEADS:
        line = f"  lead {L:2d}h: "
        for c in FEATS_OUT:
            r = res[(res.lead_h == L) & (res.feature == c)].iloc[0]
            line += f"{c}={r.auc_mean:.3f}[{r.ci_lo:.3f},{r.ci_hi:.3f}]  "
        n = res[(res.lead_h == L) & (res.feature == 's')].iloc[0].n_storms
        print(line + f"(n={n} storms)")

    # paired A48 vs instantaneous, per storm (only storms with both defined)
    print("\n[ew-loso] A48 - instantaneous advantage, paired per storm (mean, 95% CI):")
    adv_rows = []
    for L in LEADS:
        pairs = []
        for s in storms:
            ps, pa = held[s]["pos"][L]["s"], held[s]["pos"][L]["A48"]
            if len(ps[np.isfinite(ps)]) >= MIN_BIN:
                d = auc_pos_neg(pa, held[s]["neg"]["A48"]) - auc_pos_neg(ps, held[s]["neg"]["s"])
                pairs.append(d)
        pairs = np.array(pairs, float)
        if len(pairs) >= 2:
            point = float(np.nanmean(pairs))
            bm = [np.nanmean(pairs[rng.integers(0, len(pairs), size=len(pairs))]) for _ in range(B)]
            lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
            beats = bool(lo > 0)
            verdict = "A48 WINS" if beats else "n.s. (CI spans 0)"
        else:
            point = float(np.nanmean(pairs)) if len(pairs) else float("nan")
            lo = hi = float("nan"); beats = False
            verdict = f"insufficient storms (n={len(pairs)})"
        adv_rows.append({"lead_h": L, "delta_auc_mean": point, "ci_lo": lo, "ci_hi": hi,
                         "n_storms": int(len(pairs)), "A48_beats_instant": beats})
        print(f"  lead {L:2d}h: dAUC={point:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  n={len(pairs)}  {verdict}")

    # ---- plot: AUC vs lead with CI bands ----
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
    colors = {"s": "#4c72b0", "A24": "#dd8452", "A48": "#55a868"}
    labels = {"s": "instantaneous", "A24": "trailing 24h", "A48": "trailing 48h"}
    for c in FEATS_OUT:
        sub = res[res.feature == c].sort_values("lead_h")
        ax.plot(sub.lead_h, sub.auc_mean, "-o", color=colors[c], label=labels[c], lw=2)
        ax.fill_between(sub.lead_h, sub.ci_lo, sub.ci_hi, color=colors[c], alpha=0.18)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("warning lead (hours before storm)")
    ax.set_ylabel("held-out AUC vs fizzle (LOSO)")
    ax.set_title("Hardened early-warning: per-storm LOSO AUC (storm-block-bootstrap 95% CI)")
    ax.grid(alpha=0.3); ax.legend()
    for L in LEADS:
        n = res[(res.lead_h == L) & (res.feature == 's')].iloc[0].n_storms
        ax.annotate(f"n={n}", (L, 0.505), fontsize=7, ha="center", color="grey")
    fig.tight_layout(); fig.savefig(out / "early_warning_loso.png"); plt.close(fig)
    print(f"\n[ew-loso] wrote {out/'early_warning_loso.png'}")

    (out / "early_warning_loso.json").write_text(json.dumps({
        "n_storms": len(storms), "storms": storms,
        "auc_by_lead_feature": res.to_dict(orient="records"),
        "a48_vs_instant": adv_rows,
        "windows_h": WINDOWS, "lead_tol_h": LEAD_TOL, "n_boot": B,
        "validation": "leave-one-storm-out; storm-block bootstrap",
    }, indent=2, default=str))
    print(f"[ew-loso] wrote {out/'early_warning_loso.json'}")


def parse_args():
    ap = argparse.ArgumentParser(description="LOSO-hardened trailing-accumulation early-warning trigger.")
    ap.add_argument("--gka-grid", required=True)
    ap.add_argument("--pregen-grid", required=True)
    ap.add_argument("--lead-grid", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--max-neg", type=int, default=40000)
    ap.add_argument("--min-bin", type=int, default=30,
                    help="min positives at a lead for a storm to contribute to that lead's AUC.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default="figures/early_warning_loso")
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
