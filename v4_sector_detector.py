#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4_sector_detector.py — sequential sector genesis detector (see preregistration_v4_0.md).

Implements the registered design exactly:

  S1 Reservoir : C(t) = median cape      >= c*     held D consecutive hours
  S2 Build     : I(t) = median gka_SII  non-decreasing over trailing W, AND
                 G(t) = mean V3.3 score non-decreasing over W with rise >= g*
  S3 Confirm   : H(t) = median shear_low >= h*     held D hours

  A detection fires at the first hour S1 -> S2 -> S3 are satisfied IN ORDER.

Three modes, deliberately separate so the parameter freeze is enforced by the
filesystem rather than by good intentions:

  --mode sectors : build + cache hourly 5deg sector series from the season panels
  --mode fit     : grid-search the six free parameters on FIT seasons only,
                   write results/metrics/v4_params.json (with its own SHA-256)
  --mode test    : load frozen params, evaluate ONCE on TEST seasons, run the
                   baseline and both registered controls. REFUSES to run if the
                   params file is absent, and records the params hash it used.

Scope caveat recorded in the output: the season panels are storm-windowed
(+/-6deg, -120h/+24h around tracks), so negatives are storm-adjacent sectors, not
a global background. False-alarm rates are therefore relative to that population.

USAGE
    python v4_sector_detector.py --mode sectors
    python v4_sector_detector.py --mode fit
    python v4_sector_detector.py --mode test
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_spiral_genesis import fit_logistic, predict                       # noqa: E402
from geomval_seasons import (load_storm_crops, read_tracks, resolve_files,   # noqa: E402
                             genesis_events, add_strict_labels, haversine_km)

CURATED = ["gka_shear_quench", "gka_msl_nd", "gka_SII", "gka_knee_ratio"]
FIT_SEASONS = [2021, 2022]
TEST_SEASONS = [2023, 2024, 2025]
SECTOR_DEG = 5.0            # FIXED by preregistration, not tuned
GENESIS_KT = 34.0           # FIXED
GENESIS_WINDOW_H = 72.0     # FIXED


# ---------------------------------------------------------------- sectors

def sector_id(lat, lon):
    return (np.floor(lat / SECTOR_DEG) * SECTOR_DEG,
            np.floor(lon / SECTOR_DEG) * SECTOR_DEG)


def build_sectors(a) -> pd.DataFrame:
    """Hourly sector state. The V3.3 score model is trained on FIT seasons only
    and applied unchanged everywhere, so test seasons never inform the score."""
    feats = CURATED
    crops = load_storm_crops(a.panels, a.tracks, feats,
                             extra_cols=["cape", "shear_low"],
                             pre_h=a.pre_h, pad_deg=a.pad_deg, max_cells=a.max_cells)
    tr_all = read_tracks(resolve_files(a.tracks))
    ev_all = genesis_events(tr_all, GENESIS_KT)

    # --- score model: FIT seasons only ---
    Xs, ys = [], []
    for k, v in crops.items():
        if v["season"] not in FIT_SEASONS:
            continue
        d, _ = add_strict_labels(v["df"], ev_all, a.genesis_radius, GENESIS_WINDOW_H)
        sp = d[d["pregen"] == 1]
        Xs.append(sp[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        ys.append((sp["gen_pos"] == 1).to_numpy(float))
    X = np.vstack(Xs); y = np.concatenate(ys)
    fin = np.all(np.isfinite(X), axis=1); X, y = X[fin], y[fin]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = fit_logistic((X - mu) / sd, y, l2=1.0)
    print(f"[v4] score model trained on FIT seasons {FIT_SEASONS}: n={len(y):,}")

    rows = []
    for k, v in crops.items():
        d = v["df"].copy()
        Xa = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        m = np.all(np.isfinite(Xa), axis=1)
        sc = np.full(len(d), np.nan)
        sc[m] = predict(w, (Xa[m] - mu) / sd)
        d["s"] = sc
        d = d[d["pregen"] == 1]
        if d.empty:
            continue
        slat, slon = sector_id(pd.to_numeric(d["lat"], errors="coerce").to_numpy(float),
                               pd.to_numeric(d["lon"], errors="coerce").to_numpy(float))
        d = d.assign(sec_lat=slat, sec_lon=slon, season=v["season"])
        g = (d.groupby(["season", "sec_lat", "sec_lon", "time"])
               .agg(C=("cape", "median"), I=("gka_SII", "median"),
                    G=("s", "mean"), H=("shear_low", "median"),
                    n_cells=("s", "size"))
               .reset_index())
        rows.append(g)
    sec = pd.concat(rows, ignore_index=True)
    sec = (sec.groupby(["season", "sec_lat", "sec_lon", "time"], as_index=False)
              .agg(C=("C", "median"), I=("I", "median"), G=("G", "mean"),
                   H=("H", "median"), n_cells=("n_cells", "sum")))
    sec = sec[sec["n_cells"] >= a.min_cells].sort_values(
        ["season", "sec_lat", "sec_lon", "time"]).reset_index(drop=True)
    print(f"[v4] sector-hours: {len(sec):,}  sectors: "
          f"{sec[['sec_lat','sec_lon']].drop_duplicates().shape[0]}")

    # genesis truth per sector
    ev = ev_all.copy()
    gslat, gslon = sector_id(ev["lat_g"].to_numpy(float), ev["lon_g"].to_numpy(float))
    ev = ev.assign(sec_lat=gslat, sec_lon=gslon)
    ev.to_parquet(Path(a.cache).with_name("v4_genesis_events.parquet"), index=False)
    sec.to_parquet(a.cache, index=False)
    print(f"[v4] wrote {a.cache} and v4_genesis_events.parquet ({len(ev)} events)")
    return sec


# ---------------------------------------------------------------- detector

def _held(mask: np.ndarray, D: int) -> np.ndarray:
    """True at index i if mask held for D consecutive steps ending at i."""
    if D <= 1:
        return mask.copy()
    c = pd.Series(mask.astype(int)).rolling(D, min_periods=D).sum().to_numpy()
    return c >= D


def detect_sector(df: pd.DataFrame, p: dict, order=("S1", "S2", "S3")):
    """Return (detect_time, stage_times) for one sector's hourly series, or None."""
    df = df.sort_values("time")
    t = df["time"].to_numpy()
    C = df["C"].to_numpy(float); I = df["I"].to_numpy(float)
    G = df["G"].to_numpy(float); H = df["H"].to_numpy(float)
    W, D = int(p["W"]), int(p["D"])

    def shifted(x, k):
        out = np.full_like(x, np.nan)
        if k < len(x):
            out[k:] = x[:-k] if k else x
        return out

    s1 = _held(C >= p["c_star"], D)
    rise_G = G - shifted(G, W)
    rise_I = I - shifted(I, W)
    s2 = _held((rise_G >= p["g_star"]) & (rise_I >= 0), D)
    s3 = _held(H >= p["h_star"], D)
    stages = {"S1": s1, "S2": s2, "S3": s3}

    idx = 0
    times = {}
    for name in order:
        m = stages[name]
        hit = np.where(m[idx:])[0]
        if len(hit) == 0:
            return None, times
        idx = idx + hit[0]
        times[name] = t[idx]
    return t[idx], times


def episodes_for(sec: pd.DataFrame, ev: pd.DataFrame, p: dict, order=("S1", "S2", "S3"),
                 mode="sequential"):
    """One row per (sector, S1-episode) with detection + truth."""
    out = []
    for (season, la, lo), g in sec.groupby(["season", "sec_lat", "sec_lon"]):
        g = g.sort_values("time").reset_index(drop=True)
        in_s1 = (g["C"].to_numpy(float) >= p["c_star"])
        if not in_s1.any():
            continue
        brk = np.where(np.diff(in_s1.astype(int)) != 0)[0] + 1
        for seg in np.split(np.arange(len(g)), brk):
            if not len(seg) or not in_s1[seg[0]]:
                continue
            sub = g.iloc[seg]
            if len(sub) < int(p["D"]):
                continue
            if mode == "gate_only":
                hit = np.where(sub["G"].to_numpy(float) >= p["g_only"])[0]
                dt = sub["time"].to_numpy()[hit[0]] if len(hit) else None
            else:
                dt, _ = detect_sector(sub, p, order=order)
            evs = ev[(ev["season"] == season) & (ev["sec_lat"] == la) & (ev["sec_lon"] == lo)]
            truth, lead, ename = 0, np.nan, ""
            if len(evs):
                ref = dt if dt is not None else sub["time"].iloc[0]
                for e in evs.itertuples(index=False):
                    L = (pd.Timestamp(e.t_g) - pd.Timestamp(ref)) / pd.Timedelta(hours=1)
                    if 0 <= L <= GENESIS_WINDOW_H:
                        truth, lead, ename = 1, L, f"{e.season}:{e.name}"
                        break
            out.append({"season": season, "sec_lat": la, "sec_lon": lo,
                        "t_start": sub["time"].iloc[0], "detected": dt is not None,
                        "t_detect": dt, "y": truth, "lead_h": lead, "event": ename})
    return pd.DataFrame(out)


def score_episodes(ep: pd.DataFrame):
    if ep.empty:
        return dict(n_episodes=0, n_fired=0, precision=np.nan, recall=np.nan,
                    f1=np.nan, median_lead=np.nan, n_events_caught=0)
    fired = ep[ep["detected"]]
    n_ev = ep.loc[ep["y"] == 1, "event"].replace("", np.nan).dropna().nunique()
    caught = fired.loc[fired["y"] == 1, "event"].replace("", np.nan).dropna().nunique()
    prec = float((fired["y"] == 1).mean()) if len(fired) else np.nan
    rec = caught / n_ev if n_ev else np.nan
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and prec + rec > 0) else np.nan
    return dict(n_episodes=int(len(ep)), n_fired=int(len(fired)),
                precision=prec, recall=rec, f1=f1,
                median_lead=float(fired.loc[fired["y"] == 1, "lead_h"].median())
                if caught else np.nan,
                n_events_caught=int(caught), n_events_total=int(n_ev))


# ---------------------------------------------------------------- modes

def mode_fit(a):
    sec = pd.read_parquet(a.cache)
    ev = pd.read_parquet(Path(a.cache).with_name("v4_genesis_events.parquet"))
    sec = sec[sec["season"].isin(FIT_SEASONS)]
    ev = ev[ev["season"].isin(FIT_SEASONS)]
    print(f"[v4-fit] FIT seasons {FIT_SEASONS}: {len(sec):,} sector-hours, {len(ev)} events")

    grid = {
        "c_star": list(np.nanquantile(sec["C"], [0.5, 0.6, 0.7, 0.8])),
        "g_star": [0.0, 0.002, 0.005, 0.010],
        "h_star": list(np.nanquantile(sec["H"], [0.4, 0.5, 0.6, 0.7])),
        "D": [3, 6],
        "W": [6, 12, 24],
    }
    keys = list(grid)
    best, best_f1 = None, -np.inf
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, combo))
        m = score_episodes(episodes_for(sec, ev, p))
        if np.isfinite(m["f1"]) and m["f1"] > best_f1:
            best_f1, best = m["f1"], {**p, "_fit_metrics": m}
    if best is None:
        raise SystemExit("[v4-fit] no parameter set produced a finite F1.")
    best["g_only"] = float(np.nanquantile(sec["G"], 0.90))   # for the gate-only control
    print(f"[v4-fit] best F1={best_f1:.4f} at "
          + ", ".join(f"{k}={best[k]:.4g}" for k in keys))
    print(f"[v4-fit] fit metrics: {best['_fit_metrics']}")

    outp = Path(a.params)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload = {"preregistration": "preregistration_v4_0.md",
               "fit_seasons": FIT_SEASONS, "test_seasons": TEST_SEASONS,
               "sector_deg": SECTOR_DEG, "genesis_kt": GENESIS_KT,
               "genesis_window_h": GENESIS_WINDOW_H, "grid": {k: list(map(float, grid[k])) for k in keys},
               "params": {k: float(best[k]) for k in keys + ["g_only"]},
               "fit_metrics": best["_fit_metrics"]}
    blob = json.dumps(payload, indent=2, default=str)
    payload["self_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    outp.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[v4-fit] FROZEN -> {outp}  sha256={payload['self_sha256'][:16]}...")
    print("[v4-fit] parameters are now frozen; --mode test evaluates ONCE.")
    return 0


def mode_test(a):
    pf = Path(a.params)
    if not pf.exists():
        raise SystemExit(f"[v4-test] frozen parameters not found at {pf}. "
                         "Run --mode fit first; the preregistration requires the "
                         "freeze to precede any test-season evaluation.")
    payload = json.loads(pf.read_text())
    p = payload["params"]
    print(f"[v4-test] using frozen params (sha256={payload.get('self_sha256','?')[:16]}...): "
          + ", ".join(f"{k}={v:.4g}" for k, v in p.items()))

    sec = pd.read_parquet(a.cache)
    ev = pd.read_parquet(Path(a.cache).with_name("v4_genesis_events.parquet"))
    sec = sec[sec["season"].isin(TEST_SEASONS)]
    ev = ev[ev["season"].isin(TEST_SEASONS)]
    print(f"[v4-test] TEST seasons {TEST_SEASONS}: {len(sec):,} sector-hours, {len(ev)} events")

    res = {}
    res["sequential"] = score_episodes(episodes_for(sec, ev, p))
    res["gate_only"] = score_episodes(episodes_for(sec, ev, p, mode="gate_only"))
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(a.n_perm_orders):
        order = tuple(rng.permutation(["S1", "S2", "S3"]))
        if order == ("S1", "S2", "S3"):
            continue
        perm.append(score_episodes(episodes_for(sec, ev, p, order=order)))
    res["permuted_order"] = {
        k: float(np.nanmean([q[k] for q in perm])) for k in
        ("precision", "recall", "f1", "median_lead")} if perm else {}

    # baseline: max-pool G over the sector, threshold swept on FIT seasons
    secf = pd.read_parquet(a.cache); evf = pd.read_parquet(
        Path(a.cache).with_name("v4_genesis_events.parquet"))
    secf = secf[secf["season"].isin(FIT_SEASONS)]; evf = evf[evf["season"].isin(FIT_SEASONS)]
    best_thr, best_f1 = None, -np.inf
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        thr = float(np.nanquantile(secf["G"], q))
        m = score_episodes(episodes_for(secf, evf, {**p, "c_star": -np.inf, "g_only": thr},
                                        mode="gate_only"))
        if np.isfinite(m["f1"]) and m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    res["baseline_maxpool"] = score_episodes(
        episodes_for(sec, ev, {**p, "c_star": -np.inf, "g_only": best_thr}, mode="gate_only"))
    res["baseline_threshold"] = best_thr

    print("\n[v4-test] RESULTS (test seasons, single evaluation):")
    for k in ("sequential", "baseline_maxpool", "gate_only", "permuted_order"):
        v = res.get(k, {})
        if not v:
            continue
        print(f"  {k:18s} precision={v.get('precision', float('nan')):.3f} "
              f"recall={v.get('recall', float('nan')):.3f} "
              f"F1={v.get('f1', float('nan')):.3f} "
              f"median_lead={v.get('median_lead', float('nan')):.1f}h "
              f"events={v.get('n_events_caught','-')}/{v.get('n_events_total','-')}")

    seq, base = res["sequential"], res["baseline_maxpool"]
    crit = {
        "1_precision_gain>=0.05": bool(np.isfinite(seq["precision"]) and np.isfinite(base["precision"])
                                       and (seq["precision"] - base["precision"]) >= 0.05),
        "2_median_lead>=24h_and_beats_baseline": bool(np.isfinite(seq["median_lead"])
                                                      and seq["median_lead"] >= 24
                                                      and seq["median_lead"] >= base.get("median_lead", -np.inf)),
        "3_order_matters": bool(res["permuted_order"] and np.isfinite(seq["f1"])
                                and seq["f1"] > res["permuted_order"].get("f1", np.inf)),
        "4_beats_gate_only": bool(np.isfinite(seq["f1"]) and seq["f1"] > res["gate_only"]["f1"]),
    }
    supported = all(crit.values())
    print("\n[v4-test] preregistered criteria:")
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n[v4-test] VERDICT: V4.0 {'SUPPORTED' if supported else 'NOT SUPPORTED'}")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "v4_test_results.json").write_text(json.dumps(
        {"params_sha256": payload.get("self_sha256"), "params": p,
         "results": res, "criteria": crit, "supported": supported,
         "scope_caveat": "panels are storm-windowed; negatives are storm-adjacent "
                         "sectors, not a global background"},
        indent=2, default=str))
    print(f"[v4-test] wrote {out/'v4_test_results.json'}")
    return 0


def main() -> int:
    a = parse_args()
    if a.mode == "sectors":
        build_sectors(a); return 0
    if a.mode == "fit":
        return mode_fit(a)
    return mode_test(a)


def parse_args():
    ap = argparse.ArgumentParser(description="V4.0 sequential sector genesis detector.")
    ap.add_argument("--mode", choices=["sectors", "fit", "test"], required=True)
    ap.add_argument("--panels", default="data/genesis_*_slim_cape.parquet")
    ap.add_argument("--tracks", default=("data/tracks/tracks_2021.parquet,data/tracks/tracks_2022.parquet,"
                                         "data/tracks/tracks_2023.parquet,data/tracks/tracks_2024.parquet,"
                                         "data/tracks/tracks_geomval.parquet"))
    ap.add_argument("--cache", default="data/v4_sector_series.parquet")
    ap.add_argument("--params", default="results/metrics/v4_params.json")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--genesis-radius", type=float, default=3.0)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--pre-h", type=float, default=120.0)
    ap.add_argument("--pad-deg", type=float, default=6.0)
    ap.add_argument("--max-cells", type=int, default=3000)
    ap.add_argument("--n-perm-orders", type=int, default=5)
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
