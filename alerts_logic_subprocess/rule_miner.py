#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rule_miner.py
Find sparse (L1) logistic drivers and propose a simple threshold rule.

- Label options:
    • label_mode = "future_window":
        strict future window (t, t+H] per (lat,lon) based on a binary label column
        (storm / near_storm / pregen) — robust to gaps

    • label_mode = "t_to_storm":
        use continuous t_to_storm_min_h and define positives as
        0 < t_to_storm_min_h <= lead_hours

- NaN/Inf handling: column-wise median impute; Infs -> NaN
- Outputs: AUC/PRAUC/Brier, best-F1 (test), sorted non-zero L1 coefs,
           and a 1-feature threshold rule evaluated on the test set
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


# ---- strict-future label: for each (lat,lon), any(target==1) in (t, t+H] ----
def future_max_label(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    win = f"{int(hours)}H"

    def _one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("time")
        # reverse -> rolling future max; shift(1) to exclude current instant
        rev = g.set_index("time")[label_col].astype(int).iloc[::-1]
        fut = rev.rolling(win, min_periods=1).max().shift(1)
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        fut.index = g.index
        return fut

    out = df.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_one)
    return out.reindex(df.index).astype(int)


def best_f1_threshold(y_true, p):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = (2 * pr * rc) / (pr + rc + 1e-9)
    i = int(np.nanargmax(f1))
    thr = float(th[max(i - 1, 0)]) if len(th) else 0.5
    return thr, float(f1[i]), float(pr[i]), float(rc[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labelled",
        required=True,
        help="Path to grid-labelled CSV/Parquet with features+labels",
    )
    ap.add_argument(
        "--target",
        default="pregen",
        help="Base label column when label_mode='future_window' "
             "(e.g. 'storm', 'near_storm', 'pregen')",
    )
    ap.add_argument(
        "--label-mode",
        choices=["future_window", "t_to_storm"],
        default="future_window",
        help=(
            "Label construction:\n"
            "  future_window: strict future (t,t+H] on binary target column\n"
            "  t_to_storm   : 0 < t_to_storm_min_h <= lead_hours"
        ),
    )
    ap.add_argument("--lead-hours", type=int, default=24)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--max-features",
        type=int,
        default=8,
        help="Report top-K |coef| features",
    )
    ap.add_argument(
        "--C",
        type=float,
        default=0.5,
        help="Inverse regularization strength (smaller -> sparser)",
    )
    ap.add_argument("--out", default="models/rule_l1.pkl")
    args = ap.parse_args()

    # ---- load ----
    pth = str(args.labelled).lower()
    if pth.endswith((".parquet", ".pq", ".pqt")):
        df = pd.read_parquet(args.labelled)
    else:
        df = pd.read_csv(args.labelled, low_memory=False)

    # time normalize & order
    if "time" not in df.columns or "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("Need columns: time, lat, lon.")
    df["time"] = (
        pd.to_datetime(df["time"], errors="coerce", utc=True)
        .dt.tz_localize(None)
    )
    df = (
        df.dropna(subset=["time", "lat", "lon"])
          .sort_values(["lat", "lon", "time"])
          .reset_index(drop=True)
    )

    # candidate features (take intersect gracefully)
    preferred = [
        # original low-level / dynamical
        "S", "relax", "agree", "msl", "msl_grad", "zeta", "div",
        "zeta_mean", "div_mean",
        "dS_dt", "drelax_dt", "dagree_dt",
        "S_mean3h", "S_std3h",
        "zeta_mean3h", "zeta_std3h",
        "div_mean3h", "div_std3h",
        "wspd", "u10", "v10",
        # GKA family
        "gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap",
        "gka_F", "gka_msl_nd", "gka_knee_ratio", "gka_chirality",
        "gka_Q", "gka_dir_var", "gka_vortdiv_ratio",
        # spherical / thermo feedback
        "sph_center", "sph_radial_signed", "sph_radial_abs", "sph_vdr_std",
        "t2m_anom_local", "pdrop_nd", "thermo_shear",
        "SFI", "SFI2",
    ]
    present = [
        c
        for c in preferred
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not present:
        raise SystemExit(
            "No usable numeric features found from preferred set "
            f"(none of {len(preferred)} candidates present)."
        )

    # ---- label construction ----
    if args.label_mode == "t_to_storm":
        if "t_to_storm_min_h" not in df.columns:
            raise SystemExit(
                "label_mode='t_to_storm' requires column 't_to_storm_min_h' "
                "in the labelled dataset."
            )
        tts = pd.to_numeric(
            df["t_to_storm_min_h"], errors="coerce"
        ).fillna(np.inf)
        # 0 < t_to_storm <= lead_hours
        y = ((tts > 0) & (tts <= float(args.lead_hours))).astype(int).to_numpy()
        print(
            f"Rows: {len(df):,}  Positives(t_to_storm<=+{args.lead_hours}h): "
            f"{int(y.sum()):,}"
        )
    else:
        # future_window mode: strict future label from an existing binary column
        if args.target not in df.columns:
            raise SystemExit(
                f"Target '{args.target}' not found in labelled file."
            )
        base_lab = (
            pd.to_numeric(df[args.target], errors="coerce")
              .fillna(0)
              .astype(int)
        )
        df = df.copy()
        df[args.target] = base_lab
        y = future_max_label(df, args.target, hours=args.lead_hours).to_numpy()
        print(
            f"Rows: {len(df):,}  Positives(+{args.lead_hours}h via '{args.target}'): "
            f"{int(y.sum()):,}"
        )

    if y.sum() == 0:
        raise SystemExit("No positive examples for the chosen label/lead window.")

    # feature matrix with robust cleanup
    Xdf = (
        df[present]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)
    X = Xdf.to_numpy()

    # split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    # L1-sparse logistic on standardized space
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        C=args.C,
        max_iter=1000,
        random_state=args.random_state,
    )
    clf.fit(Xtr_s, ytr)
    p = clf.predict_proba(Xte_s)[:, 1]

    # metrics
    auc = roc_auc_score(yte, p)
    ap_ = average_precision_score(yte, p)
    bri = brier_score_loss(yte, p)
    thr, f1, prec, rec = best_f1_threshold(yte, p)
    print("\n== L1-Logistic (test) ==")
    print(f"AUC={auc:.3f}  PRAUC={ap_:.3f}  Brier={bri:.3f}")
    print(f"Best-F1={f1:.3f} at thr={thr:.3f}  (P={prec:.3f}, R={rec:.3f})")

    # non-zero standardized coefs
    w = clf.coef_.ravel()
    nz = [
        (feat, float(coef))
        for feat, coef in zip(present, w)
        if abs(coef) > 1e-9
    ]
    nz_sorted = sorted(nz, key=lambda t: abs(t[1]), reverse=True)
    print("\nNon-zero weights (standardized; sorted by |coef|):")
    for f, c in nz_sorted[: args.max_features]:
        print(f"  {f:18s} {c:+.4f}")

    # --- propose a 1-feature rule on TEST SPLIT using raw feature scale ---
    if nz_sorted:
        top_feat, top_coef = nz_sorted[0]
        # rebuild test column for the same rows:
        # need to extract the same rows used in Xte; we don't have indices directly,
        # so re-split the DataFrame to align. We can reuse train_test_split with the same RNG.
        idx = np.arange(len(df))
        _, idx_te, _, _ = train_test_split(
            idx,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        v_test = df.loc[idx_te, top_feat].to_numpy()
        y_test = y[idx_te]

        # scan percentiles on TEST to avoid train leakage
        grid = np.unique(
            np.nanpercentile(v_test, np.linspace(5, 95, 19))
        )
        best = (None, -1.0, 0.0, 0.0)
        sgn = 1 if top_coef >= 0 else -1
        for t in grid:
            yhat = (v_test >= t) if sgn >= 0 else (v_test <= t)
            pr, rc, _th = precision_recall_curve(
                y_test, yhat.astype(int)
            )
            f1s = (2 * pr * rc) / (pr + rc + 1e-9)
            j = int(np.nanargmax(f1s))
            if f1s[j] > best[1]:
                best = (
                    float(t),
                    float(f1s[j]),
                    float(pr[j]),
                    float(rc[j]),
                )
        if best[0] is not None:
            op = ">=" if sgn >= 0 else "<="
            print("\n== Proposed 1-feature rule (TEST) ==")
            print(
                f"IF {top_feat} {op} {best[0]:.6g}  "
                f"THEN positive-within-{args.lead_hours}h"
            )
            print(
                f"F1={best[1]:.3f}  (P={best[2]:.3f}, R={best[3]:.3f})"
            )

    # save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": sc,
            "model": clf,
            "features": present,
            "lead_hours": int(args.lead_hours),
            "C": float(args.C),
            "label_mode": args.label_mode,
            "target": args.target,
            "best_thr_test": float(thr),
            "metrics_test": {
                "AUC": float(auc),
                "PRAUC": float(ap_),
                "Brier": float(bri),
                "F1": float(f1),
                "P": float(prec),
                "R": float(rec),
            },
        },
        args.out,
    )
    print(f"\nSaved sparse rule model -> {args.out}")


if __name__ == "__main__":
    main()