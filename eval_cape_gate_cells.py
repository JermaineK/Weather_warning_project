#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cape_gate_cells.py — does the CAPE gate help at CELL level, at a MATCHED
operating point?

The gate was validated at 5deg sector level (RECOMMENDED_CONFIGURATION.md). The
pipeline trigger is per-cell, so the gate must be re-validated at that unit before
it can be made a default.

This also fixes a confound the sector comparison had: there, gated and ungated
configurations fired at different rates, so the precision gain and the lead loss
were entangled with threshold placement. Here every configuration is evaluated at
the SAME alert rate, so precision differences are attributable to the gate rather
than to firing less often.

Design
    strict genesis labels, leave-one-season-out scoring (model never sees the
    season it scores), cells restricted to spiral cells (pregen == 1).

    For each target alert rate r:
      ungated : fire on the top r fraction by score
      gated   : require cape >= c*, then fire on the top-scoring cells until the
                SAME number of alerts is reached
    Compare precision, recall and median lead. Bootstrap over storms.

USAGE
    python eval_cape_gate_cells.py \\
        --panels "data/genesis_*_slim_cape.parquet" \\
        --tracks "data/tracks/tracks_2021.parquet,...,tracks_geomval.parquet" \\
        --out-dir results/metrics
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_spiral_genesis import fit_logistic, predict                       # noqa: E402
from geomval_seasons import (load_storm_crops, read_tracks, resolve_files,   # noqa: E402
                             genesis_events, add_strict_labels)

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
ALERT_RATES = [0.005, 0.01, 0.02, 0.05, 0.10]


def boot_ci(v, rng, n_boot=2000):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (float(np.nanmean(v)) if len(v) else np.nan, np.nan, np.nan, len(v))
    bm = [np.nanmean(v[rng.integers(0, len(v), size=len(v))]) for _ in range(n_boot)]
    return (float(np.nanmean(v)), float(np.percentile(bm, 2.5)),
            float(np.percentile(bm, 97.5)), len(v))


def main() -> int:
    a = parse_args()
    feats = CURATED
    crops = load_storm_crops(a.panels, a.tracks, feats, extra_cols=["cape"],
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)
    ev_all = genesis_events(read_tracks(resolve_files(a.tracks)), a.genesis_thresh_kt)
    seasons = sorted({v["season"] for v in crops.values()})

    # strict labels + season-blocked scores
    labelled = {}
    for k, v in crops.items():
        d, _ = add_strict_labels(v["df"], ev_all, a.genesis_radius, a.genesis_max_lead)
        sp = d[d["pregen"] == 1]
        pos = sp[sp["gen_pos"] == 1]
        neg = sp[(sp["gen_pos"] == 0) & (sp["near_storm"] == 0)]
        if len(pos) < a.min_bin or len(neg) < a.min_bin:
            continue
        dd = pd.concat([pos, neg], ignore_index=True)
        dd["y"] = (dd["gen_pos"] == 1).astype(int)
        labelled[k] = {"df": dd, "season": v["season"]}
    print(f"[cape-gate] {len(labelled)} storms across {len(seasons)} seasons")

    scored = {}
    for s in seasons:
        tr = [k for k, v in labelled.items() if v["season"] != s]
        te = [k for k, v in labelled.items() if v["season"] == s]
        if not tr or not te:
            continue
        X = np.vstack([labelled[k]["df"][feats].apply(pd.to_numeric, errors="coerce")
                       .to_numpy(float) for k in tr])
        y = np.concatenate([labelled[k]["df"]["y"].to_numpy(float) for k in tr])
        fin = np.all(np.isfinite(X), axis=1); X, y = X[fin], y[fin]
        mu, sd = X.mean(0), X.std(0) + 1e-9
        w = fit_logistic((X - mu) / sd, y, l2=1.0)
        for k in te:
            d = labelled[k]["df"].copy()
            Xa = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            m = np.all(np.isfinite(Xa), axis=1)
            sc = np.full(len(d), np.nan); sc[m] = predict(w, (Xa[m] - mu) / sd)
            d["s"] = sc
            scored[k] = {"df": d, "season": labelled[k]["season"]}
    print(f"[cape-gate] scored {len(scored)} storms (season-blocked)")

    # CAPE threshold from the training seasons only, to avoid using test cape
    fit_cape = np.concatenate([
        pd.to_numeric(v["df"]["cape"], errors="coerce").dropna().to_numpy(float)
        for k, v in scored.items() if v["season"] in a.fit_seasons])
    rows = []
    for cq in a.cape_quantiles:
        c_star = float(np.nanquantile(fit_cape, cq))
        for r in ALERT_RATES:
            per_storm = []
            for k, v in scored.items():
                d = v["df"].dropna(subset=["s"])
                if d.empty:
                    continue
                n_fire = max(1, int(round(r * len(d))))
                y = d["y"].to_numpy(int)
                s = d["s"].to_numpy(float)
                cape = pd.to_numeric(d["cape"], errors="coerce").to_numpy(float)
                lead = pd.to_numeric(d["gen_lead_h"], errors="coerce").to_numpy(float)

                # ungated: top-n by score
                ui = np.argsort(-s)[:n_fire]
                # gated: eligible only where cape >= c*, then top-n by score
                elig = np.where(cape >= c_star)[0]
                gi = elig[np.argsort(-s[elig])][:n_fire] if len(elig) else np.array([], int)
                if len(gi) == 0:
                    continue
                per_storm.append({
                    "storm": k,
                    "prec_ungated": float(y[ui].mean()),
                    "prec_gated": float(y[gi].mean()),
                    "lead_ungated": float(np.nanmedian(lead[ui][y[ui] == 1]))
                    if (y[ui] == 1).any() else np.nan,
                    "lead_gated": float(np.nanmedian(lead[gi][y[gi] == 1]))
                    if (y[gi] == 1).any() else np.nan,
                    "n_gated": len(gi), "n_target": n_fire,
                })
            if not per_storm:
                continue
            ps = pd.DataFrame(per_storm)
            rng = np.random.default_rng(1)
            d_prec = (ps["prec_gated"] - ps["prec_ungated"]).to_numpy(float)
            mp, lop, hip, npair = boot_ci(d_prec, rng, a.n_boot)
            d_lead = (ps["lead_gated"] - ps["lead_ungated"]).to_numpy(float)
            ml, lol, hil, _ = boot_ci(d_lead, rng, a.n_boot)
            rows.append({
                "cape_quantile": cq, "c_star": c_star, "alert_rate": r,
                "prec_ungated": float(ps["prec_ungated"].mean()),
                "prec_gated": float(ps["prec_gated"].mean()),
                "delta_precision": mp, "dp_lo": lop, "dp_hi": hip,
                "delta_lead_h": ml, "dl_lo": lol, "dl_hi": hil,
                "n_storms": npair,
                "gate_shortfall": float((ps["n_gated"] < ps["n_target"]).mean()),
            })
    res = pd.DataFrame(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "cape_gate_cells.csv", index=False)

    print("\n[cape-gate] matched-alert-rate comparison (paired over storms):")
    print(f"{'capeQ':>6} {'c*':>8} {'rate':>6} {'ungated':>8} {'gated':>8} "
          f"{'dPrec':>8} {'95% CI':>18} {'dLead(h)':>9} {'n':>3}")
    for _, x in res.iterrows():
        sig = "*" if (np.isfinite(x.dp_lo) and x.dp_lo > 0) else " "
        print(f"{x.cape_quantile:6.2f} {x.c_star:8.0f} {x.alert_rate:6.3f} "
              f"{x.prec_ungated:8.3f} {x.prec_gated:8.3f} {x.delta_precision:+8.4f}{sig} "
              f"[{x.dp_lo:+.4f},{x.dp_hi:+.4f}] {x.delta_lead_h:+9.1f} {int(x.n_storms):3d}")

    wins = res[(res.dp_lo > 0)]
    verdict = ("CAPE GATE HELPS at cell level" if len(wins) else
               "NO cell-level benefit from the CAPE gate")
    print(f"\n[cape-gate] {verdict}"
          + (f" — {len(wins)}/{len(res)} configurations with CI excluding zero"
             if len(res) else ""))
    if len(wins):
        b = wins.sort_values("delta_precision", ascending=False).iloc[0]
        print(f"[cape-gate] best: cape>={b.c_star:.0f} J/kg at alert_rate={b.alert_rate:.3f} "
              f"-> dPrecision={b.delta_precision:+.4f} [{b.dp_lo:+.4f},{b.dp_hi:+.4f}], "
              f"dLead={b.delta_lead_h:+.1f}h")
    (out / "cape_gate_cells.json").write_text(json.dumps(
        {"verdict": verdict, "rows": res.to_dict("records"),
         "note": "paired over storms at matched alert rate; season-blocked scores; "
                 "c* from fit seasons only"}, indent=2, default=str))

    if not res.empty:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        for cq, g in res.groupby("cape_quantile"):
            g = g.sort_values("alert_rate")
            ax.errorbar(g.alert_rate, g.delta_precision,
                        yerr=[g.delta_precision - g.dp_lo, g.dp_hi - g.delta_precision],
                        marker="o", capsize=3, label=f"cape q={cq:.2f}")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("alert rate (matched)")
        ax.set_ylabel("precision gain from CAPE gate")
        ax.set_title("Cell-level CAPE gate at matched operating points")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "cape_gate_cells.png"); plt.close(fig)
    print(f"[cape-gate] wrote {out/'cape_gate_cells.csv'} / .json / .png")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="Cell-level CAPE gate at matched operating points.")
    ap.add_argument("--panels", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--fit-seasons", type=int, nargs="+", default=[2021, 2022])
    ap.add_argument("--cape-quantiles", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--genesis-thresh-kt", type=float, default=34.0)
    ap.add_argument("--genesis-radius", type=float, default=3.0)
    ap.add_argument("--genesis-max-lead", type=float, default=72.0)
    ap.add_argument("--min-bin", type=int, default=30)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
    ap.add_argument("--n-boot", type=int, default=2000)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
