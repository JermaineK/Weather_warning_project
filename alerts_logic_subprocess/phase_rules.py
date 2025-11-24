#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_rules.py  — Phase comparison + simple rule mining (time-aware windows, strict-future or t_to_storm labels)
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

warnings.filterwarnings("ignore", message="is_categorical_dtype is deprecated")
pd.options.mode.copy_on_write = True

# ---------- Utils ----------


def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    # enforce time parsing + deterministic order
    if "time" in df.columns:
        df["time"] = (
            pd.to_datetime(df["time"], utc=True, errors="coerce")
            .dt.tz_localize(None)
        )
    must = {"lat", "lon", "time"}
    if not must.issubset(df.columns):
        missing = sorted(must - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")
    return (
        df.dropna(subset=["time", "lat", "lon"])
        .sort_values(["lat", "lon", "time"], kind="mergesort")
        .reset_index(drop=True)
    )


def impute_then_scale(df: pd.DataFrame, cols, scaler):
    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return scaler.transform(X.to_numpy(float))


# --- Time-aware rolling helpers (strict-future / past-excluding-current) ---


def future_max_timeaware(df: pd.DataFrame, label_col: str, hours: int) -> pd.Series:
    """
    For each (lat,lon,time), 1 iff any label==1 occurs in (t, t+hours].
    Reverse → rolling(time) → shift(1) → reverse. Returns Series aligned to df.index.
    """

    def _lead(g: pd.DataFrame) -> pd.Series:
        s = pd.Series(g[label_col].astype(int).to_numpy(), index=g["time"])
        rev = s.iloc[::-1]
        fut = rev.rolling(f"{hours}h", min_periods=1).max().shift(1)  # exclude current
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        return pd.Series(fut.to_numpy(), index=g.index)

    return df.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_lead)


def past_max_timeaware(df: pd.DataFrame, prob_col: str, hours: int) -> pd.Series:
    """
    For each (lat,lon,time), max(prob) over [t-hours, t) — past window excluding current.
    Returns Series aligned to df.index.
    """

    def _past(g: pd.DataFrame) -> pd.Series:
        s = pd.Series(g[prob_col].to_numpy(float), index=g["time"])
        r = s.rolling(f"{hours}h", min_periods=1).max().shift(1)  # exclude current
        return pd.Series(
            r.reindex(g["time"]).fillna(0).to_numpy(), index=g.index
        )

    return df.groupby(["lat", "lon"], sort=False, group_keys=False).apply(_past)


# ---------- Rule miner bits ----------


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray):
    pr, rc, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i = int(np.nanargmax(f1))
    use_thr = float(thr[max(i - 1, 0)]) if len(thr) else 0.5
    return float(f1[i]), use_thr, float(pr[i]), float(rc[i])


def rule_from_feature(
    df: pd.DataFrame,
    feature: str,
    y: np.ndarray,
    dir_hint: str = "pos",
    steps: int = 25,
):
    """
    1D threshold rule on a feature. dir_hint='pos' → (x >= thr), 'neg' → (x <= thr).
    Returns best (F1, thr, P, R).
    """
    x = df[feature].to_numpy(float)
    q = np.nanpercentile(x, np.linspace(5, 95, steps))
    best = (-1.0, np.nan, 0.0, 0.0)
    for thr in q:
        if dir_hint == "pos":
            pred = (x >= thr).astype(int)
        else:
            pred = (x <= thr).astype(int)
        if pred.sum() == 0:
            continue
        f1 = f1_score(y, pred, zero_division=0)
        tp = int(((pred == 1) & (y == 1)).sum())
        pp = int(pred.sum())
        pos = int((y == 1).sum())
        precision = tp / pp if pp else 0.0
        recall = tp / pos if pos else 0.0
        if f1 > best[0]:
            best = (float(f1), float(thr), float(precision), float(recall))
    return best


# ---------- Main ----------


def main():
    ap = argparse.ArgumentParser(
        description="Phase comparison + rule mining (time-aware, strict-future or t_to_storm labels)"
    )
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", default="pregen")
    ap.add_argument(
        "--label-mode",
        choices=["future_window", "t_to_storm"],
        default="future_window",
        help=(
            "Label construction:\n"
            "  future_window: strict future (t,t+H] from binary target column\n"
            "  t_to_storm   : 0 < t_to_storm_min_h <= lead_h for each lead"
        ),
    )
    ap.add_argument(
        "--leads",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[24, 48],
    )
    ap.add_argument("--phase-top-frac", type=float, default=0.10)
    ap.add_argument("--subsample-hours", type=float, default=0.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Normalise leads (handles default list vs CLI string)
    if isinstance(args.leads, str):
        leads = [int(x) for x in args.leads.split(",") if x.strip()]
    else:
        leads = list(args.leads)

    # Load + normalize
    df = pd.read_csv(args.labelled, parse_dates=["time"])

    must = {"lat", "lon", "time"}
    if args.label_mode == "future_window":
        must.add(args.target)
    else:  # t_to_storm
        must.add("t_to_storm_min_h")

    if not must.issubset(df.columns):
        missing = sorted(must - set(df.columns))
        raise ValueError(f"Labelled file missing columns: {missing}")

    df = ensure_sorted(df)

    # Optional hour subsample (by distinct hours)
    if 0 < args.subsample_hours < 1.0:
        hrs = df["time"].dt.floor("h").drop_duplicates()
        keep = set(hrs.sample(frac=args.subsample_hours, random_state=42))
        df = df[df["time"].dt.floor("h").isin(keep)].reset_index(drop=True)

    # Models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)
    feats_b, sc_b = mb["features"], mb["scaler"]
    feats_r, sc_r = mr["features"], mr["scaler"]

    # Probabilities (robust impute)
    Xb = impute_then_scale(df, feats_b, sc_b)
    Xr = impute_then_scale(df, feats_r, sc_r)
    df["p_build"] = mb["model"].predict_proba(Xb)[:, 1]
    df["p_relax"] = mr["model"].predict_proba(Xr)[:, 1]

    # Phase slices (top fraction by each prob)
    qb = df["p_build"].quantile(1.0 - args.phase_top_frac)
    qr = df["p_relax"].quantile(1.0 - args.phase_top_frac)
    top_build_idx = df["p_build"] >= qb
    top_relax_idx = df["p_relax"] >= qr

    # ---------- Labels per lead ----------
    labels_by_lead = {}

    if args.label_mode == "future_window":
        base = df[["time", "lat", "lon", args.target]].copy()
        base[args.target] = (
            pd.to_numeric(base[args.target], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        for h in leads:
            y = future_max_timeaware(base, args.target, h).to_numpy()
            labels_by_lead[h] = y
            print(
                f"Lead {h:3d}h  positives(future_window on '{args.target}'): "
                f"{int(y.sum()):,}"
            )
    else:
        # label_mode = t_to_storm
        tts = (
            pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
            .fillna(np.inf)
            .to_numpy()
        )
        for h in leads:
            y = ((tts > 0) & (tts <= float(h))).astype(int)
            labels_by_lead[h] = y
            print(
                f"Lead {h:3d}h  positives(0 < t_to_storm_min_h <= {h}): "
                f"{int(y.sum()):,}"
            )

    # ---------- Feature influence per phase ----------

    # Base feature pool from the two models
    feature_pool_raw = list(dict.fromkeys(list(feats_b) + list(feats_r)))

    # Extend with extra spiral / spherical / thermo candidates if present
    extra_candidates = [
        "SFI",
        "SFI2",
        "thermo_shear",
        "pdrop_nd",
        "sph_center",
        "sph_radial_signed",
        "sph_radial_abs",
        "sph_vdr_std",
        "t2m_anom_local",
    ]
    feature_pool_raw.extend(extra_candidates)
    # Deduplicate while preserving order
    feature_pool_raw = list(dict.fromkeys(feature_pool_raw))
    # Keep only numeric columns that exist in df
    feature_pool = [
        f
        for f in feature_pool_raw
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    rows = []
    from scipy.stats import spearmanr

    for h in leads:
        y = labels_by_lead[h]
        for phase_tag, mask in [("build", top_build_idx), ("relax", top_relax_idx)]:
            sub = df.loc[mask, feature_pool]
            yy = y[mask.to_numpy()]
            for col in sub.columns:
                xc = sub[col].to_numpy(float)
                ok = np.isfinite(xc) & np.isfinite(yy)
                if ok.sum() < 200:  # require a bit more support
                    continue
                r, _ = spearmanr(xc[ok], yy[ok])
                rows.append(
                    {
                        "phase": phase_tag,
                        "lead_h": h,
                        "feature": col,
                        "spearman_r": float(r),
                        "abs_r": float(abs(r)),
                        "n": int(ok.sum()),
                    }
                )

    feat_out = outdir / "phase_feature_influence.csv"
    pd.DataFrame(rows).sort_values(
        ["lead_h", "phase", "abs_r"], ascending=[True, True, False]
    ).to_csv(feat_out, index=False)

    # ---------- Simple 1-feature rule mining ----------

    dir_hints = {}
    for f in feature_pool:
        if f in {
            "S",
            "S_mean3h",
            "S_std3h",
            "zeta_mean",
            "zeta_std3h",
            "agree",
            "relax",
            "SFI",
            "SFI2",
            "thermo_shear",
        }:
            dir_hints[f] = "pos"
        elif f in {"div_mean", "div_std3h", "msl_grad", "pdrop_nd"}:
            dir_hints[f] = "neg"
        else:
            # default assumption: "higher is worse / more stormy"
            dir_hints[f] = "pos"

    rule_rows = []
    for h in leads:
        y = labels_by_lead[h]
        for f in feature_pool:
            if f not in df.columns:
                continue
            F1, thr, P, R = rule_from_feature(
                df, f, y, dir_hint=dir_hints.get(f, "pos"), steps=31
            )
            rule_rows.append(
                {
                    "lead_h": h,
                    "feature": f,
                    "direction": dir_hints.get(f, "pos"),
                    "thr": thr,
                    "F1": F1,
                    "Precision": P,
                    "Recall": R,
                }
            )

    rules_out = outdir / "rules_candidates.csv"
    pd.DataFrame(rule_rows).sort_values(
        ["lead_h", "F1"], ascending=[True, False]
    ).to_csv(rules_out, index=False)

    # ---------- Quick console summary ----------

    def fmt_top(df_rules, lead):
        d = df_rules[df_rules["lead_h"] == lead].head(8)
        return "\n".join(
            f"  • {lead:>2}h: {r['feature']:>16} ({r['direction']}) "
            f"thr={r['thr']:.4f}  F1={r['F1']:.3f}  "
            f"P={r['Precision']:.3f} R={r['Recall']:.3f}"
            for _, r in d.iterrows()
        )

    print(f"\nSaved phase feature influence → {feat_out}")
    print(f"Saved rule candidates        → {rules_out}\n")
    rules_df = pd.read_csv(rules_out)
    for h in leads:
        print(f"Top 1-feature rules @ lead {h}h")
        print(fmt_top(rules_df, h))
        print("")


if __name__ == "__main__":
    main()