#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
storm_geometric_validation.py — v3.3

A validation harness for path/loop-area cyclogenesis features, built to
address the OVERFITTING/VALIDATION roadblock rather than to add more
features.  Preregistered: see preregistration_v3_3.md.

WHY THIS SHAPE
    When a rare-event forecasting project stalls on validation, adding
    features makes it worse. Two things fix it:
      1. A protocol that cannot leak (block CV by season AND basin) with
         rare-event-appropriate metrics and an explicit PERMUTATION NULL,
         so "apparent skill" is always compared against the skill this
         pipeline produces on shuffled labels.
      2. PHYSICS INVARIANCE GUARDS. The geometric-pumping reading makes
         three structured predictions that an overfit model cannot fake:
            (a) hemisphere SIGN FLIP of the signed-area effect,
            (b) effect scales with |enclosed area|,
            (c) effect is invariant to traversal SPEED (adiabatic).
         Noise fitted to one basin/season will not reproduce three
         independent structured invariances. These are free confirmatory
         tests that do not cost held-out data.

    The harness ships with a SELF-TEST that plants a known signal in
    synthetic data and also runs pure noise — so you can verify the
    validator itself before trusting its verdict on ERA5.

DATA UNIT
    The signed loop-area feature is computed per storm track: for each
    track (grouped by storm_id) the ordered (lon, lat) path is treated as
    a closed loop and its signed shoelace area encodes the handedness of
    the pre-cyclone motion. One sample == one track.

NO SKLEARN REQUIRED (numpy/scipy/pandas only): the classifier is a
regularised logistic regression fitted by IRLS with ridge penalty.

USAGE
    # validate the validator (no data needed):
    python storm_geometric_validation.py --self-test

    # run on a real tracks file:
    python storm_geometric_validation.py \\
        --infile data/seed_tracks.parquet \\
        --out-dir results/geom_val \\
        --label-col genesis \\
        --n-perm 200

    # or import the pieces:
    from storm_geometric_validation import signed_loop_area, evaluate
    res = evaluate(X, y, blocks, n_perm=200)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ───────────────────────── I/O (repo convention) ───────────────────────
def read_any(path, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        print(f"[geom-val] reading Parquet: {path}")
        return pd.read_parquet(path, **kw)
    print(f"[geom-val] reading CSV: {path}")
    return pd.read_csv(path, low_memory=False, **kw)


def _first_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ───────────────────────── geometric features ──────────────────────────
def signed_loop_area(u1, u2):
    """Signed area enclosed by the trajectory (u1,u2) — shoelace formula.
    Sign encodes traversal orientation (the parity-odd/handedness part).
    Trajectory is closed implicitly (last point joined to first)."""
    u1 = np.asarray(u1, float); u2 = np.asarray(u2, float)
    if u1.size < 3:
        return 0.0
    return 0.5 * float(np.sum(u1 * np.roll(u2, -1) - np.roll(u1, -1) * u2))


def loop_features(u1, u2):
    """Small FIXED feature set (anti-overfitting: no free knobs)."""
    A = signed_loop_area(u1, u2)
    du = np.diff(np.asarray(u1, float)); dv = np.diff(np.asarray(u2, float))
    path_len = float(np.sum(np.hypot(du, dv)))
    # speed-normalised area: geometric pumping is traversal-speed
    # independent, so A should NOT depend on how fast the loop is walked
    A_norm = A / path_len if path_len > 0 else 0.0
    return dict(area_signed=A, area_abs=abs(A), area_norm=A_norm,
                path_len=path_len)


# ───────────────────────── model (no sklearn) ──────────────────────────
def _fit_logistic(X, y, l2=1.0, iters=50):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        R = np.eye(X.shape[1]) * l2; R[0, 0] = 0.0
        H = X.T @ (X * W[:, None]) + R
        g = X.T @ (y - p) - R @ w
        try:
            w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return w


def _predict(w, X):
    X = np.c_[np.ones(len(X)), X]
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


# ───────────────────────── rare-event metrics ──────────────────────────
def pr_auc(y, p):
    """Average precision (area under precision-recall)."""
    o = np.argsort(-p); y = np.asarray(y)[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def brier_skill(y, p):
    """Brier skill score vs climatology (>0 means better than base rate)."""
    base = np.mean(y)
    bs = np.mean((p - y) ** 2)
    bs_ref = np.mean((base - y) ** 2)
    return float(1 - bs / bs_ref) if bs_ref > 0 else 0.0


# ───────────────────────── block CV + null test ────────────────────────
def block_cv(X, y, blocks, l2=1.0):
    """Leave-one-block-out CV. Blocks should encode season AND basin so
    neither temporal nor spatial autocorrelation can leak."""
    blocks = np.asarray(blocks)
    oof = np.zeros(len(y), float)
    for b in np.unique(blocks):
        te = blocks == b; tr = ~te
        if y[tr].sum() < 2 or y[te].sum() < 1:
            oof[te] = y[tr].mean() if tr.sum() else 0.5
            continue
        w = _fit_logistic(X[tr], y[tr], l2=l2)
        oof[te] = _predict(w, X[te])
    return oof


def evaluate(X, y, blocks, n_perm=200, l2=1.0, seed=0, verbose=True):
    """Out-of-fold skill vs a permutation null built with the SAME pipeline.
    This is the core anti-overfitting test: shuffle labels WITHIN blocks and
    re-run everything; if real skill sits inside the null distribution, the
    apparent signal is pipeline artefact."""
    X = np.asarray(X, float); y = np.asarray(y).astype(float)
    oof = block_cv(X, y, blocks, l2=l2)
    obs = dict(pr_auc=pr_auc(y, oof), bss=brier_skill(y, oof))
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks)
    null = {"pr_auc": [], "bss": []}
    for _ in range(n_perm):
        yp = y.copy()
        for b in np.unique(blocks):          # permute within block
            m = blocks == b
            yp[m] = rng.permutation(yp[m])
        o = block_cv(X, yp, blocks, l2=l2)
        null["pr_auc"].append(pr_auc(yp, o))
        null["bss"].append(brier_skill(yp, o))
    out = {}
    for k in obs:
        nl = np.array(null[k])
        p = float((np.sum(nl >= obs[k]) + 1) / (len(nl) + 1))
        out[k] = dict(observed=obs[k], null_mean=float(nl.mean()),
                      null_p95=float(np.percentile(nl, 95)), p_value=p)
    if verbose:
        for k, v in out.items():
            verdict = "ABOVE NULL" if v["p_value"] < 0.05 else "inside null"
            print(f"  {k:7s} obs={v['observed']:+.4f}  null_mean={v['null_mean']:+.4f} "
                  f"p95={v['null_p95']:+.4f}  p={v['p_value']:.3f}  {verdict}")
    return out


# ───────────────────────── physics invariance guards ───────────────────
def physics_guards(area_signed, y, hemisphere, speed=None, verbose=True):
    """Three structured predictions an overfit model cannot fake.
    Returns dict of pass/fail + effect sizes. Uses simple, assumption-light
    statistics (rank correlation / rate differences)."""
    from scipy.stats import spearmanr
    A = np.asarray(area_signed, float); y = np.asarray(y).astype(float)
    hem = np.asarray(hemisphere)
    res = {}

    # (a) hemisphere sign flip: correlation of signed area with genesis
    #     should REVERSE sign between hemispheres
    rN = spearmanr(A[hem == "N"], y[hem == "N"]).statistic if (hem == "N").sum() > 10 else np.nan
    rS = spearmanr(A[hem == "S"], y[hem == "S"]).statistic if (hem == "S").sum() > 10 else np.nan
    res["hemisphere_flip"] = dict(rho_N=float(rN), rho_S=float(rS),
                                  passes=bool(np.sign(rN) != np.sign(rS)
                                              and np.isfinite(rN) and np.isfinite(rS)))

    # (b) area scaling: within a hemisphere, genesis rate should increase
    #     monotonically with |signed area| in the favourable orientation
    fav = A * np.where(hem == "N", 1, -1)          # orient-corrected
    q = np.quantile(fav, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(fav, q)
    rates = [float(y[bins == i].mean()) if (bins == i).sum() > 5 else np.nan
             for i in range(5)]
    mono = np.all(np.diff([r for r in rates if np.isfinite(r)]) >= -1e-9)
    res["area_scaling"] = dict(rates_by_quintile=rates, monotonic=bool(mono))

    # (c) traversal-speed independence: skill should NOT depend on speed
    if speed is not None:
        sp = np.asarray(speed, float)
        med = np.median(sp)
        rs_slow = spearmanr(fav[sp <= med], y[sp <= med]).statistic
        rs_fast = spearmanr(fav[sp > med], y[sp > med]).statistic
        rel = abs(rs_slow - rs_fast) / max(abs(rs_slow), abs(rs_fast), 1e-9)
        res["speed_independence"] = dict(rho_slow=float(rs_slow),
                                         rho_fast=float(rs_fast),
                                         rel_diff=float(rel),
                                         passes=bool(rel < 0.5))
    if verbose:
        h = res["hemisphere_flip"]
        print(f"  (a) hemisphere sign flip: rho_N={h['rho_N']:+.3f} "
              f"rho_S={h['rho_S']:+.3f} -> {'PASS' if h['passes'] else 'FAIL'}")
        print(f"  (b) area scaling monotonic: "
              f"{'PASS' if res['area_scaling']['monotonic'] else 'FAIL'} "
              f"rates={[round(r,4) if np.isfinite(r) else None for r in rates]}")
        if speed is not None:
            s = res["speed_independence"]
            print(f"  (c) speed independence: rho_slow={s['rho_slow']:+.3f} "
                  f"rho_fast={s['rho_fast']:+.3f} reldiff={s['rel_diff']:.2f} "
                  f"-> {'PASS' if s['passes'] else 'FAIL'}")
    return res


# ───────────────────────── real-data track loader ──────────────────────
_MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                    6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def _lon_sector(lon_mean: float) -> str:
    """Coarse longitudinal basin sector when no basin column is present."""
    lon = ((lon_mean + 180) % 360) - 180
    if -100 <= lon < -20:
        return "ATL"
    if -180 <= lon < -100:
        return "EPAC"
    if 100 <= lon <= 180 or -180 <= lon < -140:
        return "WPAC"
    return "IO"


def load_tracks_features(path, label_col="genesis",
                         id_col=None, time_col=None,
                         lat_col="lat", lon_col="lon",
                         basin_col=None, min_points=3):
    """Build one sample per storm track.

    Returns (X, y, blocks, hemisphere, speed, area_signed, meta_df).
    X columns are the FIXED loop feature set [area_signed, area_abs, area_norm].
    """
    df = read_any(path)

    id_col = id_col or _first_col(df, ["storm_id", "track_id", "sid", "seed_id", "id"])
    if id_col is None:
        raise SystemExit("[geom-val] no track-id column found (tried storm_id/track_id/sid/seed_id/id).")
    time_col = time_col or _first_col(df, ["time", "obs_time"])
    basin_col = basin_col or _first_col(df, ["basin"])

    label_present = label_col in df.columns
    if not label_present:
        raise SystemExit(f"[geom-val] label column '{label_col}' not in data; columns={list(df.columns)[:20]}")

    for req in (lat_col, lon_col):
        if req not in df.columns:
            raise SystemExit(f"[geom-val] missing required column '{req}'.")

    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    if time_col:
        df["_t"] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values([id_col, "_t"])
    df = df.dropna(subset=[lat_col, lon_col])

    rows = []
    for sid, g in df.groupby(id_col, sort=False):
        if len(g) < min_points:
            continue
        lat = g[lat_col].to_numpy(float)
        lon = g[lon_col].to_numpy(float)
        feats = loop_features(lon, lat)                       # (u1,u2) = (lon,lat)
        lat_mean = float(np.nanmean(lat))
        hem = "N" if lat_mean >= 0 else "S"
        if basin_col and pd.notna(g[basin_col].iloc[0]) and str(g[basin_col].iloc[0]).strip():
            basin = str(g[basin_col].iloc[0]).strip()
        else:
            basin = _lon_sector(float(np.nanmean(lon)))
        if time_col and g["_t"].notna().any():
            month = int(g["_t"].dropna().iloc[0].month)
        else:
            month = 1
        season = _MONTH_TO_SEASON.get(month, 0)
        # label: any positive along the track == genesis
        y_val = float(pd.to_numeric(g[label_col], errors="coerce").fillna(0).max() > 0)
        n_steps = max(len(g) - 1, 1)
        speed = feats["path_len"] / n_steps                  # mean step distance
        rows.append(dict(storm_id=sid, area_signed=feats["area_signed"],
                         area_abs=feats["area_abs"], area_norm=feats["area_norm"],
                         path_len=feats["path_len"], hemisphere=hem, basin=basin,
                         season=season, speed=speed, y=y_val))

    if not rows:
        raise SystemExit("[geom-val] no usable tracks after grouping/filtering.")

    meta = pd.DataFrame(rows)
    X = meta[["area_signed", "area_abs", "area_norm"]].to_numpy(float)
    y = meta["y"].to_numpy(float)
    blocks = (meta["basin"].astype(str) + "_" + meta["season"].astype(str)).to_numpy()
    hemisphere = meta["hemisphere"].to_numpy()
    speed = meta["speed"].to_numpy(float)
    area_signed = meta["area_signed"].to_numpy(float)
    print(f"[geom-val] tracks={len(meta):,}  positives={int(y.sum())}  "
          f"blocks={len(np.unique(blocks))}  base_rate={y.mean():.4f}")
    return X, y, blocks, hemisphere, speed, area_signed, meta


def run_on_data(args) -> dict:
    X, y, blocks, hemisphere, speed, area_signed, meta = load_tracks_features(
        args.infile, label_col=args.label_col, min_points=args.min_points,
    )

    print("\n[geom-val] block-CV + permutation null")
    skill = evaluate(X, y, blocks, n_perm=args.n_perm, l2=args.l2, seed=args.seed)

    print("\n[geom-val] physics invariance guards")
    guards = physics_guards(area_signed, y, hemisphere, speed=speed)

    # ── preregistered verdict ──
    # SUPPORTED requires: skill ABOVE the within-block permutation null on
    # PR-AUC (p < 0.05) AND the hemisphere sign-flip guard passes AND the
    # area-scaling guard is monotonic. (Speed independence is confirmatory
    # but does not gate the verdict — see preregistration_v3_3.md.)
    skill_above_null = skill["pr_auc"]["p_value"] < 0.05
    hemi_ok = bool(guards["hemisphere_flip"]["passes"])
    area_ok = bool(guards["area_scaling"]["monotonic"])
    supported = skill_above_null and hemi_ok and area_ok
    verdict = "GEOMETRIC PUMPING: SUPPORTED" if supported else "GEOMETRIC PUMPING: NOT SUPPORTED"
    print(f"\n[geom-val] {verdict}")
    print(f"[geom-val]   PR-AUC above null (p<0.05): {skill_above_null}  (p={skill['pr_auc']['p_value']:.3f})")
    print(f"[geom-val]   hemisphere sign-flip passes: {hemi_ok}")
    print(f"[geom-val]   area-scaling monotonic:      {area_ok}")

    return {
        "verdict": verdict,
        "supported": supported,
        "skill": skill,
        "guards": guards,
        "n_tracks": int(len(y)),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
        "n_blocks": int(len(np.unique(blocks))),
        "n_perm": args.n_perm,
    }


# ───────────────────────── self-test ───────────────────────────────────
def self_test(n=4000, seed=1, n_perm=60):
    """Validate the VALIDATOR: (1) pure noise must show no skill and fail
    the guards; (2) planted geometric signal must show skill and pass.

    Returns (noise_result, signal_result) dicts for programmatic checking."""
    rng = np.random.default_rng(seed)
    hem = rng.choice(["N", "S"], n)
    basin = rng.choice(["ATL", "EPAC", "WPAC", "SIO"], n)
    season = rng.integers(0, 4, n)
    blocks = np.array([f"{b}_{s}" for b, s in zip(basin, season)])
    # trajectories: random loops with controlled signed area
    area = rng.normal(0, 1, n)
    speed = rng.uniform(0.5, 2.0, n)
    confound = rng.normal(0, 1, n)          # e.g. SST-like nuisance

    print("=" * 62)
    print("SELF-TEST 1 - PURE NOISE (harness must find NOTHING)")
    y_noise = (rng.random(n) < 0.05).astype(float)
    X = np.c_[area, abs(area), confound]
    noise_skill = evaluate(X, y_noise, blocks, n_perm=n_perm, seed=2)
    noise_guards = physics_guards(area, y_noise, hem, speed=speed)

    print("\n" + "=" * 62)
    print("SELF-TEST 2 - PLANTED GEOMETRIC SIGNAL (must be detected)")
    # genesis probability rises with orientation-corrected area, sign flips
    # by hemisphere, independent of traversal speed
    fav = area * np.where(hem == "N", 1, -1)
    logit = -3.0 + 1.2 * fav + 0.4 * confound
    p = 1 / (1 + np.exp(-logit))
    y_sig = (rng.random(n) < p).astype(float)
    signal_skill = evaluate(X, y_sig, blocks, n_perm=n_perm, seed=3)
    signal_guards = physics_guards(area, y_sig, hem, speed=speed)
    print("\nInterpretation: the harness is trustworthy only if test 1 is")
    print("'inside null' + guards FAIL, and test 2 is 'ABOVE NULL' + guards")
    print("PASS. Run this before believing any verdict on ERA5 data.")

    return (
        {"skill": noise_skill, "guards": noise_guards},
        {"skill": signal_skill, "guards": signal_guards},
    )


# ───────────────────────── output writers ──────────────────────────────
def _write_outputs(out_dir: Path, results: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "geometric_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[geom-val] wrote {json_path}")

    txt_path = out_dir / "geometric_validation_summary.txt"
    sk = results["skill"]; gd = results["guards"]
    lines = [
        "PREREGISTERED — see preregistration_v3_3.md",
        "",
        results["verdict"],
        "",
        f"PR-AUC  observed : {sk['pr_auc']['observed']:+.4f}  "
        f"null_mean={sk['pr_auc']['null_mean']:+.4f}  p={sk['pr_auc']['p_value']:.3f}",
        f"BSS     observed : {sk['bss']['observed']:+.4f}  "
        f"null_mean={sk['bss']['null_mean']:+.4f}  p={sk['bss']['p_value']:.3f}",
        "",
        f"(a) hemisphere flip : rho_N={gd['hemisphere_flip']['rho_N']:+.3f}  "
        f"rho_S={gd['hemisphere_flip']['rho_S']:+.3f}  "
        f"passes={gd['hemisphere_flip']['passes']}",
        f"(b) area scaling    : monotonic={gd['area_scaling']['monotonic']}",
    ]
    if "speed_independence" in gd:
        s = gd["speed_independence"]
        lines.append(
            f"(c) speed indep.    : rho_slow={s['rho_slow']:+.3f}  "
            f"rho_fast={s['rho_fast']:+.3f}  rel_diff={s['rel_diff']:.2f}  "
            f"passes={s['passes']}"
        )
    lines += [
        "",
        f"n_tracks   : {results['n_tracks']}",
        f"n_positive : {results['n_positive']}",
        f"base_rate  : {results['base_rate']:.4f}",
        f"n_blocks   : {results['n_blocks']}",
        f"n_perm     : {results['n_perm']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[geom-val] wrote {txt_path}")


# ───────────────────────── CLI ─────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Preregistered geometric-pumping validation harness.")
    ap.add_argument("--self-test", action="store_true",
                    help="Validate the validator on synthetic noise + planted signal, then exit.")
    ap.add_argument("--infile", default=None, help="Tracks file (parquet or CSV) for real-data mode.")
    ap.add_argument("--out-dir", default="results/geom_val", help="Output directory (real-data mode).")
    ap.add_argument("--label-col", default="genesis", help="Binary genesis label column (default: genesis).")
    ap.add_argument("--n-perm", type=int, default=200, help="Permutation-null resamples (default: 200).")
    ap.add_argument("--l2", type=float, default=1.0, help="Ridge penalty for the logistic model (default: 1.0).")
    ap.add_argument("--min-points", type=int, default=3, help="Minimum track points to form a loop (default: 3).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the permutation null (default: 0).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.infile:
        print(__doc__)
        print("[geom-val] no --infile given and --self-test not set; nothing to do.", file=sys.stderr)
        return
    results = run_on_data(args)
    _write_outputs(Path(args.out_dir), results)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
