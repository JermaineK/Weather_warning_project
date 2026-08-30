#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4_1_detector.py — V4.1 fitting/test procedure (see preregistration_v4_1.md).

Stage logic is unchanged from V4.0 and is imported, not reimplemented, so the
detector definition is identical and only the *procedure* differs:

  FIXED (not tuned):  D = 6 h, W = 24 h, sector 5deg, genesis 34 kt / 72 h
  FREE (three only):  c*, g*, h*
  OBJECTIVE        :  maximise genesis events caught, subject to a false-alarm
                      ceiling of 5% of sector-episodes. Unlike F1, a detector
                      that never fires scores WORST here, not best — which is
                      the specific pathology that broke V4.0.
  SELECTION        :  leave-one-event-out across fit-season genesis events;
                      the modal winning candidate is frozen, and fold agreement
                      is reported so instability is visible.
  FAILED-FIT GATE  :  < 5 fired episodes on the fit seasons -> refuse to freeze,
                      do not touch the test seasons (preregistration section 5).
  INCONCLUSIVE GATE:  on test, < 5 fires or < 2 catchable events -> report
                      INCONCLUSIVE, explicitly not a falsification.

USAGE
    python v4_1_detector.py --mode fit
    python v4_1_detector.py --mode test
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v4_sector_detector import (episodes_for, score_episodes,          # noqa: E402
                                FIT_SEASONS, TEST_SEASONS, SECTOR_DEG,
                                GENESIS_KT, GENESIS_WINDOW_H)

D_FIXED = 6      # persistence, hours   — fixed from the event-centred composite
W_FIXED = 24     # build window, hours  — fixed from the event-centred composite
FA_CEILING = 0.05
MIN_FIRES_FIT = 5
MIN_FIRES_TEST = 5
MIN_EVENTS_TEST = 2


def load(cache):
    sec = pd.read_parquet(cache)
    ev = pd.read_parquet(Path(cache).with_name("v4_genesis_events.parquet"))
    return sec, ev


def candidate_grid(sec):
    return {
        "c_star": [float(x) for x in np.nanquantile(sec["C"], [0.3, 0.4, 0.5, 0.6, 0.7])],
        "g_star": [0.0, 0.001, 0.002, 0.005],
        "h_star": [float(x) for x in np.nanquantile(sec["H"], [0.3, 0.4, 0.5, 0.6])],
    }


def evaluate(sec, ev, p):
    """Episode-level metrics for one candidate, with fixed D/W."""
    full = {**p, "D": D_FIXED, "W": W_FIXED, "g_only": 0.0}
    ep = episodes_for(sec, ev, full)
    m = score_episodes(ep)
    m["fa_rate"] = (m["n_fired"] / m["n_episodes"]) if m["n_episodes"] else np.nan
    m["caught_events"] = sorted(
        ep.loc[ep["detected"] & (ep["y"] == 1), "event"].replace("", np.nan).dropna().unique())
    return m, ep


def objective(m):
    """Events caught, subject to the FA ceiling. Non-firing scores worst."""
    if not np.isfinite(m.get("fa_rate", np.nan)) or m["fa_rate"] > FA_CEILING:
        return (-1, -np.inf)                     # violates the alarm budget
    lead = m.get("median_lead", np.nan)
    return (m["n_events_caught"], lead if np.isfinite(lead) else -np.inf)


def mode_fit(a):
    sec, ev = load(a.cache)
    sec = sec[sec["season"].isin(FIT_SEASONS)]
    ev = ev[ev["season"].isin(FIT_SEASONS)].reset_index(drop=True)
    ev["key"] = ev["season"].astype(str) + ":" + ev["name"].astype(str)
    print(f"[v4.1-fit] FIT {FIT_SEASONS}: {len(sec):,} sector-hours, {len(ev)} events")
    print(f"[v4.1-fit] FIXED D={D_FIXED}h W={W_FIXED}h; objective = events caught "
          f"s.t. FA <= {FA_CEILING:.0%}")

    grid = candidate_grid(sec)
    combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]
    print(f"[v4.1-fit] {len(combos)} candidates over 3 free parameters")

    # cache each candidate's metrics once
    cache_m = {}
    for i, p in enumerate(combos):
        m, _ = evaluate(sec, ev, p)
        cache_m[i] = m

    # ---- leave-one-event-out selection ----
    folds = []
    for _, e in ev.iterrows():
        held = e["key"]
        best_i, best_obj = None, (-2, -np.inf)
        for i, p in enumerate(combos):
            m = cache_m[i]
            caught_wo = [c for c in m["caught_events"] if c != held]
            mm = {**m, "n_events_caught": len(caught_wo)}
            o = objective(mm)
            if o > best_obj:
                best_obj, best_i = o, i
        held_caught = held in cache_m[best_i]["caught_events"] if best_i is not None else False
        folds.append({"held_event": held, "chosen": best_i, "held_caught": bool(held_caught)})
    fold_df = pd.DataFrame(folds)
    agree = Counter(fold_df["chosen"]).most_common()
    modal_i = agree[0][0]
    print(f"[v4.1-fit] LOEO folds: {len(fold_df)}; held-out events caught: "
          f"{int(fold_df['held_caught'].sum())}/{len(fold_df)}")
    print(f"[v4.1-fit] fold agreement on winning candidate: "
          f"{agree[0][1]}/{len(fold_df)} chose candidate {modal_i}"
          + ("" if len(agree) == 1 else f"  (alternatives: {agree[1:]})"))

    p_final = combos[modal_i]
    m_final, ep_final = evaluate(sec, ev, p_final)
    print(f"[v4.1-fit] frozen candidate: " + ", ".join(f"{k}={v:.4g}" for k, v in p_final.items()))
    print(f"[v4.1-fit] fit metrics: fired={m_final['n_fired']}/{m_final['n_episodes']} "
          f"(FA={m_final['fa_rate']:.2%}) precision={m_final['precision']:.3f} "
          f"events={m_final['n_events_caught']}/{m_final['n_events_total']} "
          f"median_lead={m_final['median_lead']:.1f}h")

    # ---- preregistered FAILED-FIT gate ----
    if m_final["n_fired"] < MIN_FIRES_FIT:
        print(f"\n[v4.1-fit] FAILED FIT: only {m_final['n_fired']} fired episodes "
              f"(< {MIN_FIRES_FIT}). Per preregistration section 5 the test seasons are "
              f"NOT evaluated and no parameters are frozen. Revise under V4.2.")
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.out_dir) / "v4_1_failed_fit.json").write_text(json.dumps(
            {"reason": "fewer than MIN_FIRES_FIT fired episodes on fit seasons",
             "min_fires_fit": MIN_FIRES_FIT, "n_fired": int(m_final["n_fired"]),
             "candidate": {k: float(v) for k, v in p_final.items()},
             "fit_metrics": {k: (v if not isinstance(v, list) else v)
                             for k, v in m_final.items()},
             "loeo_folds": fold_df.to_dict("records")}, indent=2, default=str))
        return 2

    payload = {"preregistration": "preregistration_v4_1.md",
               "fixed": {"D": D_FIXED, "W": W_FIXED, "sector_deg": SECTOR_DEG,
                         "genesis_kt": GENESIS_KT, "genesis_window_h": GENESIS_WINDOW_H},
               "fa_ceiling": FA_CEILING, "grid": grid,
               "params": {k: float(v) for k, v in p_final.items()},
               "loeo": {"n_folds": int(len(fold_df)),
                        "held_out_caught": int(fold_df["held_caught"].sum()),
                        "fold_agreement": f"{agree[0][1]}/{len(fold_df)}",
                        "folds": fold_df.to_dict("records")},
               "fit_metrics": {k: v for k, v in m_final.items()}}
    blob = json.dumps(payload, indent=2, default=str)
    payload["self_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    Path(a.params).parent.mkdir(parents=True, exist_ok=True)
    Path(a.params).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[v4.1-fit] FROZEN -> {a.params}  sha256={payload['self_sha256'][:16]}...")
    return 0


def mode_test(a):
    pf = Path(a.params)
    if not pf.exists():
        raise SystemExit(f"[v4.1-test] no frozen parameters at {pf}. Run --mode fit first "
                         "(and it must not have hit the failed-fit gate).")
    payload = json.loads(pf.read_text())
    p = payload["params"]
    print(f"[v4.1-test] frozen params (sha256={payload.get('self_sha256','?')[:16]}...): "
          + ", ".join(f"{k}={v:.4g}" for k, v in p.items())
          + f"   FIXED D={D_FIXED} W={W_FIXED}")

    sec, ev = load(a.cache)
    sect = sec[sec["season"].isin(TEST_SEASONS)]
    evt = ev[ev["season"].isin(TEST_SEASONS)]
    print(f"[v4.1-test] TEST {TEST_SEASONS}: {len(sect):,} sector-hours, {len(evt)} events")

    res = {}
    m_seq, ep_seq = evaluate(sect, evt, p)
    res["sequential"] = {k: v for k, v in m_seq.items()}

    full = {**p, "D": D_FIXED, "W": W_FIXED, "g_only": 0.0}
    # gate-only control: CAPE gate + plain threshold on G, threshold from fit seasons
    secf = sec[sec["season"].isin(FIT_SEASONS)]
    g_thr = float(np.nanquantile(secf["G"], 0.90))
    res["gate_only"] = score_episodes(episodes_for(sect, evt, {**full, "g_only": g_thr},
                                                   mode="gate_only"))
    # baseline: max-pool G over sector (no CAPE gate), threshold swept on fit seasons
    evf = ev[ev["season"].isin(FIT_SEASONS)]
    best_thr, best_obj = None, (-2, -np.inf)
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        thr = float(np.nanquantile(secf["G"], q))
        mm = score_episodes(episodes_for(secf, evf,
                                         {**full, "c_star": -np.inf, "g_only": thr},
                                         mode="gate_only"))
        mm["fa_rate"] = (mm["n_fired"] / mm["n_episodes"]) if mm["n_episodes"] else np.nan
        o = objective(mm)
        if o > best_obj:
            best_obj, best_thr = o, thr
    res["baseline_maxpool"] = score_episodes(
        episodes_for(sect, evt, {**full, "c_star": -np.inf, "g_only": best_thr},
                     mode="gate_only"))
    res["baseline_threshold"] = best_thr
    # permuted-order control
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(a.n_perm_orders):
        order = tuple(rng.permutation(["S1", "S2", "S3"]))
        if order == ("S1", "S2", "S3"):
            continue
        perm.append(score_episodes(episodes_for(sect, evt, full, order=order)))
    res["permuted_order"] = ({k: float(np.nanmean([q[k] for q in perm]))
                              for k in ("precision", "recall", "f1", "median_lead")}
                             if perm else {})

    print("\n[v4.1-test] RESULTS (single evaluation):")
    for k in ("sequential", "baseline_maxpool", "gate_only", "permuted_order"):
        v = res.get(k) or {}
        if not v:
            continue
        print(f"  {k:18s} fired={v.get('n_fired','-'):>5} "
              f"precision={v.get('precision', float('nan')):.3f} "
              f"recall={v.get('recall', float('nan')):.3f} "
              f"median_lead={v.get('median_lead', float('nan')):.1f}h "
              f"events={v.get('n_events_caught','-')}/{v.get('n_events_total','-')}")

    # ---- preregistered INCONCLUSIVE gate (declared before running) ----
    n_fired = m_seq["n_fired"]
    n_catchable = m_seq.get("n_events_total", 0)
    if n_fired < MIN_FIRES_TEST or n_catchable < MIN_EVENTS_TEST:
        verdict = "INCONCLUSIVE"
        print(f"\n[v4.1-test] VERDICT: INCONCLUSIVE (insufficient power) — "
              f"fired={n_fired} (min {MIN_FIRES_TEST}), catchable events={n_catchable} "
              f"(min {MIN_EVENTS_TEST}).")
        print("[v4.1-test] Per preregistration section 5 this is a failure of the study "
              "design, NOT evidence against the hypothesis.")
        crit = {}
    else:
        base = res["baseline_maxpool"]
        crit = {
            "1_precision_gain>=0.05": bool(
                np.isfinite(m_seq["precision"]) and np.isfinite(base["precision"])
                and (m_seq["precision"] - base["precision"]) >= 0.05),
            "2_median_lead>=24h_and_beats_baseline": bool(
                np.isfinite(m_seq["median_lead"]) and m_seq["median_lead"] >= 24
                and m_seq["median_lead"] >= base.get("median_lead", -np.inf)),
            "3_order_matters": bool(
                res["permuted_order"] and np.isfinite(m_seq["f1"])
                and m_seq["f1"] > res["permuted_order"].get("f1", np.inf)),
            "4_beats_gate_only": bool(
                np.isfinite(m_seq["f1"]) and m_seq["f1"] > res["gate_only"]["f1"]),
        }
        print("\n[v4.1-test] preregistered criteria:")
        for k, v in crit.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        verdict = "SUPPORTED" if all(crit.values()) else "NOT SUPPORTED"
        print(f"\n[v4.1-test] VERDICT: V4.1 {verdict}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "v4_1_test_results.json").write_text(json.dumps(
        {"params_sha256": payload.get("self_sha256"), "params": p,
         "fixed": {"D": D_FIXED, "W": W_FIXED}, "results": res,
         "criteria": crit, "verdict": verdict,
         "gates": {"min_fires_test": MIN_FIRES_TEST, "min_events_test": MIN_EVENTS_TEST,
                   "n_fired": int(n_fired), "n_catchable_events": int(n_catchable)},
         "scope_caveat": "panels are storm-windowed; negatives are storm-adjacent "
                         "sectors, not a global background"},
        indent=2, default=str))
    print(f"[v4.1-test] wrote {out/'v4_1_test_results.json'}")
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="V4.1 sequential sector detector: fit / test.")
    ap.add_argument("--mode", choices=["fit", "test"], required=True)
    ap.add_argument("--cache", default="data/v4_sector_series.parquet")
    ap.add_argument("--params", default="results/metrics/v4_1_params.json")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--n-perm-orders", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    try:
        a = parse_args()
        sys.exit(mode_fit(a) if a.mode == "fit" else mode_test(a))
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
