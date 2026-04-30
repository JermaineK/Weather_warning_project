#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_phi_threshold.py — preregistered evaluation of the gka_phi composite threshold.

See preregistration_v3_2.md for the falsification criterion.

Usage
-----
python eval_phi_threshold.py \\
    --infile  data/grid_labelled_FMA_gka.parquet \\
    --out-dir results/phi_eval \\
    --label-col pregen \\
    --lead-window 24,48 \\
    --train-end 2025-03-31 \\
    --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit
from scipy.stats import bootstrap as scipy_bootstrap


# ---------------- I/O ----------------

def read_any(path, **kw) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".parq", ".pq")):
        print(f"[phi-eval] reading Parquet: {path}")
        return pd.read_parquet(path, **kw)
    print(f"[phi-eval] reading CSV: {path}")
    return pd.read_csv(path, low_memory=False, **kw)


# ---------------- logistic helpers ----------------

def _sigmoid(x):
    return expit(np.clip(x, -500, 500))


def _loglik(params, phi, y):
    a, b = params
    p = _sigmoid(a + b * phi)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_smooth(phi: np.ndarray, y: np.ndarray):
    """Fit standard logistic P = sigmoid(a + b*phi). Returns (a, b, ll)."""
    res = minimize(_loglik, [0.0, 0.0], args=(phi, y), method="L-BFGS-B")
    a, b = res.x
    return a, b, -res.fun


def _loglik_break(params, phi, y, phi_c):
    a, b1, b2 = params
    lin = a + b1 * phi + b2 * np.maximum(0.0, phi - phi_c)
    p = _sigmoid(lin)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_break(phi: np.ndarray, y: np.ndarray, phi_c: float):
    """Fit broken logistic P = sigmoid(a + b1*phi + b2*max(0, phi-phi_c)). Returns (a,b1,b2,ll)."""
    res = minimize(_loglik_break, [0.0, 0.0, 0.0], args=(phi, y, phi_c), method="L-BFGS-B")
    a, b1, b2 = res.x
    return a, b1, b2, -res.fun


def grid_search_phi_c(phi_train: np.ndarray, y_train: np.ndarray,
                      quantiles: np.ndarray) -> tuple[float, float, float, float, float]:
    """Return (phi_c, a, b1, b2, ll) for the phi_c that maximises train log-likelihood."""
    candidates = np.nanquantile(phi_train, quantiles)
    best_ll = -np.inf
    best = (candidates[0], 0.0, 0.0, 0.0, -np.inf)
    for phi_c in candidates:
        a, b1, b2, ll = fit_break(phi_train, y_train, phi_c)
        if ll > best_ll:
            best_ll = ll
            best = (phi_c, a, b1, b2, ll)
    return best


# ---------------- AUC / Brier ----------------

def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    return float(np.trapz(tpr, fpr))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------- GPI baseline ----------------

def build_gpi_features(df: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """Return (X, description) for GPI logistic baseline, or (None, reason)."""
    # Primary: vmax_potential, mid_humidity, shear_proxy
    primary = {"vmax_potential": None, "mid_humidity": None, "shear_proxy": None}
    fallback = {"humidity_mid_proxy": None, "shear_proxy": None}

    for col in primary:
        if col in df.columns:
            primary[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

    if all(v is not None for v in primary.values()):
        X = np.column_stack([primary["vmax_potential"],
                             primary["mid_humidity"],
                             primary["shear_proxy"]])
        return X, "vmax_potential + mid_humidity + shear_proxy"

    for col in fallback:
        if col in df.columns:
            fallback[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

    if all(v is not None for v in fallback.values()):
        X = np.column_stack([fallback["humidity_mid_proxy"],
                             fallback["shear_proxy"]])
        return X, "humidity_mid_proxy + shear_proxy (fallback)"

    return None, "no GPI proxy columns found"


def _loglik_multi(params, X, y):
    beta = params
    lin = X @ beta[1:] + beta[0]
    p = _sigmoid(lin)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_gpi_logistic(X_train, y_train, X_test, y_test):
    n_feat = X_train.shape[1]
    res = minimize(_loglik_multi, np.zeros(n_feat + 1),
                   args=(X_train, y_train), method="L-BFGS-B")
    beta = res.x
    p_test = _sigmoid(X_test @ beta[1:] + beta[0])
    return auc_roc(y_test, p_test), brier(y_test, p_test)


# ---------------- bootstrap CI ----------------

def _bootstrap_slope_diff(phi: np.ndarray, y: np.ndarray,
                           phi_c: float, n_boot: int, rng: np.random.Generator):
    diffs = np.empty(n_boot)
    n = len(phi)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            _, b1, b2, _ = fit_break(phi[idx], y[idx], phi_c)
            diffs[i] = b2  # slope change at phi_c
        except Exception:
            diffs[i] = np.nan
    valid = diffs[np.isfinite(diffs)]
    if len(valid) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def _bootstrap_phi_c(phi: np.ndarray, y: np.ndarray,
                     quantiles: np.ndarray, n_boot: int, rng: np.random.Generator):
    phi_cs = np.empty(n_boot)
    n = len(phi)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            phi_c_i, *_ = grid_search_phi_c(phi[idx], y[idx], quantiles)
            phi_cs[i] = phi_c_i
        except Exception:
            phi_cs[i] = np.nan
    valid = phi_cs[np.isfinite(phi_cs)]
    if len(valid) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ---------------- main evaluation ----------------

def evaluate(args) -> dict:
    df = read_any(args.infile)

    # filter: lead window
    lead_vals = [int(x) for x in str(args.lead_window).split(",")]
    if "lead_h" in df.columns:
        df = df[(df["lead_h"] >= lead_vals[0]) & (df["lead_h"] <= lead_vals[1])]
        print(f"[phi-eval] lead_h filter {lead_vals} -> {len(df):,} rows")
    else:
        print(f"[phi-eval] 'lead_h' not found; skipping lead-window filter", file=sys.stderr)

    # filter: finite gka_phi
    if "gka_phi" not in df.columns:
        raise SystemExit("[phi-eval] 'gka_phi' column not found in input; run compute_gka_features.py first.")
    df = df[np.isfinite(df["gka_phi"].astype(float))].copy()
    print(f"[phi-eval] finite gka_phi rows: {len(df):,}")

    label_col = args.label_col
    if label_col not in df.columns:
        raise SystemExit(f"[phi-eval] label column '{label_col}' not in data.")

    # train / test split
    train_end = pd.Timestamp(args.train_end)
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce")
        train_mask = t <= train_end
        test_mask  = t  > train_end
    else:
        raise SystemExit("[phi-eval] 'time' column required for train/test split.")

    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()
    print(f"[phi-eval] train={len(df_train):,}  test={len(df_test):,}")

    if len(df_train) < 30 or len(df_test) < 10:
        raise SystemExit("[phi-eval] insufficient data in train or test split.")

    phi_train = df_train["gka_phi"].to_numpy(float)
    y_train   = pd.to_numeric(df_train[label_col], errors="coerce").to_numpy(float)
    phi_test  = df_test["gka_phi"].to_numpy(float)
    y_test    = pd.to_numeric(df_test[label_col],  errors="coerce").to_numpy(float)

    # drop rows with NaN labels
    m_tr = np.isfinite(y_train); phi_train, y_train = phi_train[m_tr], y_train[m_tr]
    m_te = np.isfinite(y_test);  phi_test,  y_test  = phi_test[m_te],  y_test[m_te]

    quantiles = np.linspace(0.1, 0.9, 17)

    # ---------- fit models ----------
    a_s, b_s, _ = fit_smooth(phi_train, y_train)
    phi_c, a_b, b1, b2, _ = grid_search_phi_c(phi_train, y_train, quantiles)
    print(f"[phi-eval] M_smooth: a={a_s:.4f} b={b_s:.4f}")
    print(f"[phi-eval] M_break:  phi_c={phi_c:.4f} a={a_b:.4f} b1={b1:.4f} b2={b2:.4f}")

    # ---------- test evaluation ----------
    p_smooth = _sigmoid(a_s + b_s * phi_test)
    p_break  = _sigmoid(a_b + b1 * phi_test + b2 * np.maximum(0.0, phi_test - phi_c))

    auc_smooth = auc_roc(y_test, p_smooth)
    auc_break  = auc_roc(y_test, p_break)
    brier_smooth = brier(y_test, p_smooth)
    brier_break  = brier(y_test, p_break)
    print(f"[phi-eval] AUC  smooth={auc_smooth:.4f}  break={auc_break:.4f}")
    print(f"[phi-eval] Brier smooth={brier_smooth:.4f}  break={brier_break:.4f}")

    # ---------- bootstrap CIs ----------
    rng = np.random.default_rng(seed=42)
    ci_lo_diff, ci_hi_diff = _bootstrap_slope_diff(phi_test, y_test, phi_c, args.n_boot, rng)
    ci_lo_phic, ci_hi_phic = _bootstrap_phi_c(phi_test, y_test, quantiles, args.n_boot, rng)
    print(f"[phi-eval] slope-diff 95% CI: [{ci_lo_diff:.4f}, {ci_hi_diff:.4f}]")
    print(f"[phi-eval] phi_c 95% CI:      [{ci_lo_phic:.4f}, {ci_hi_phic:.4f}]")

    # ---------- GPI baseline ----------
    X_tr, gpi_desc = build_gpi_features(df_train)
    print(f"[phi-eval] GPI columns: {gpi_desc}")
    auc_gpi = float("nan")
    brier_gpi = float("nan")
    if X_tr is not None:
        X_te, _ = build_gpi_features(df_test)
        if X_te is not None and X_te.shape[1] == X_tr.shape[1]:
            m_tr2 = np.all(np.isfinite(X_tr), axis=1) & np.isfinite(y_train)
            m_te2 = np.all(np.isfinite(X_te), axis=1) & np.isfinite(y_test)
            if m_tr2.sum() > 10 and m_te2.sum() > 5:
                try:
                    auc_gpi, brier_gpi = fit_gpi_logistic(
                        X_tr[m_tr2], y_train[m_tr2],
                        X_te[m_te2], y_test[m_te2],
                    )
                    print(f"[phi-eval] GPI AUC={auc_gpi:.4f}  Brier={brier_gpi:.4f}")
                except Exception as exc:
                    print(f"[phi-eval] GPI fit failed: {exc}", file=sys.stderr)

    # ---------- verdict (preregistered falsification criterion) ----------
    # slope-diff CI must not contain zero AND auc_break - auc_gpi >= 0.02
    auc_gain = auc_break - auc_gpi if np.isfinite(auc_gpi) else float("nan")
    slope_diff_ci_excludes_zero = (
        np.isfinite(ci_lo_diff) and np.isfinite(ci_hi_diff)
        and not (ci_lo_diff <= 0.0 <= ci_hi_diff)
    )
    auc_gain_sufficient = np.isfinite(auc_gain) and auc_gain >= 0.02
    supported = slope_diff_ci_excludes_zero and auc_gain_sufficient
    verdict = "PHI THRESHOLD: SUPPORTED" if supported else "PHI THRESHOLD: NOT SUPPORTED"
    print(f"\n[phi-eval] {verdict}")
    print(f"[phi-eval]   slope_diff CI excludes 0: {slope_diff_ci_excludes_zero}")
    print(f"[phi-eval]   AUC gain over GPI >= 0.02: {auc_gain_sufficient}  (gain={auc_gain:.4f})")

    results = {
        "verdict": verdict,
        "supported": supported,
        "phi_c": phi_c,
        "phi_c_ci_95": [ci_lo_phic, ci_hi_phic],
        "slope_diff_ci_95": [ci_lo_diff, ci_hi_diff],
        "slope_diff_ci_excludes_zero": slope_diff_ci_excludes_zero,
        "auc_smooth": auc_smooth,
        "auc_break": auc_break,
        "auc_gpi": auc_gpi,
        "auc_gain_over_gpi": auc_gain,
        "auc_gain_sufficient": auc_gain_sufficient,
        "brier_smooth": brier_smooth,
        "brier_break": brier_break,
        "brier_gpi": brier_gpi,
        "gpi_columns": gpi_desc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_end": args.train_end,
        "smooth_params": {"a": float(a_s), "b": float(b_s)},
        "break_params": {"a": float(a_b), "b1": float(b1), "b2": float(b2)},
    }

    # ---------- phi_curve.csv ----------
    n_bins = 20
    bin_edges = np.nanpercentile(phi_test, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rows = []
    for lo, hi, ctr in zip(bin_edges[:-1], bin_edges[1:], bin_centres):
        mask = (phi_test >= lo) & (phi_test < hi)
        n = int(mask.sum())
        n_pos = int(y_test[mask].sum()) if n > 0 else 0
        obs = float(y_test[mask].mean()) if n > 0 else float("nan")
        sp = float(_sigmoid(a_s + b_s * ctr))
        bp = float(_sigmoid(a_b + b1 * ctr + b2 * max(0.0, ctr - phi_c)))
        rows.append({"phi_bin_centre": ctr, "n": n, "n_pos": n_pos,
                     "observed_rate": obs, "smooth_pred": sp, "break_pred": bp})
    curve_df = pd.DataFrame(rows)

    return results, curve_df


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate preregistered gka_phi threshold.")
    ap.add_argument("--infile",      required=True, help="Labelled grid file (parquet or CSV)")
    ap.add_argument("--label-col",   default="pregen", help="Binary label column (default: pregen)")
    ap.add_argument("--lead-window", default="24,48",  help="Comma-separated lead hours to include (default: 24,48)")
    ap.add_argument("--train-end",   default="2025-03-31", help="Last date of train split (default: 2025-03-31)")
    ap.add_argument("--n-boot",      type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    ap.add_argument("--out-dir",     default="results/phi_eval", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, curve_df = evaluate(args)

    # phi_threshold_results.json
    json_path = out_dir / "phi_threshold_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[phi-eval] wrote {json_path}")

    # phi_threshold_summary.txt
    txt_path = out_dir / "phi_threshold_summary.txt"
    lines = [
        "PREREGISTERED — see preregistration_v3_2.md",
        "",
        results["verdict"],
        "",
        f"phi_c           : {results['phi_c']:.4f}  95% CI [{results['phi_c_ci_95'][0]:.4f}, {results['phi_c_ci_95'][1]:.4f}]",
        f"slope diff 95CI : [{results['slope_diff_ci_95'][0]:.4f}, {results['slope_diff_ci_95'][1]:.4f}]"
        f"  excludes zero: {results['slope_diff_ci_excludes_zero']}",
        f"AUC smooth      : {results['auc_smooth']:.4f}",
        f"AUC break       : {results['auc_break']:.4f}",
        f"AUC GPI         : {results['auc_gpi']:.4f}  ({results['gpi_columns']})",
        f"AUC gain        : {results['auc_gain_over_gpi']:.4f}  (>= 0.02 required: {results['auc_gain_sufficient']})",
        f"Brier smooth    : {results['brier_smooth']:.4f}",
        f"Brier break     : {results['brier_break']:.4f}",
        f"n_train         : {results['n_train']}",
        f"n_test          : {results['n_test']}",
        f"train_end       : {results['train_end']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[phi-eval] wrote {txt_path}")

    # phi_curve.csv
    csv_path = out_dir / "phi_curve.csv"
    curve_df.to_csv(csv_path, index=False)
    print(f"[phi-eval] wrote {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\nERROR [{type(e).__name__}]: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)
