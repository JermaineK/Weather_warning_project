#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# rule_applier.py
# Apply single-feature rules from rules_candidates.csv and evaluate metrics.
# Two modes:
#   (A) legacy: time-aware future labels using a binary target (e.g. pregen)
#   (B) lead-band mode: use continuous t_to_storm_min_h and evaluate rules
#       against a time-band label: (lead_h - band_width, lead_h] hours.

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

pd.options.mode.copy_on_write = True

# ---------- helpers (robust, time-aware) ----------

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = (
            pd.to_datetime(df["time"], utc=True, errors="coerce")
              .dt.tz_localize(None)
        )
    need = {"time", "lat", "lon"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Labelled file missing required columns: {sorted(miss)}")
    return (
        df.dropna(subset=["time", "lat", "lon"])
          .sort_values(["lat", "lon", "time"], kind="mergesort")
          .reset_index(drop=True)
    )

def future_max_label_per_point_timeaware(df: pd.DataFrame, target_col: str, hours: int) -> np.ndarray:
    """
    Strictly future: for each (lat,lon,time), 1 if any target==1 occurs in (t, t+hours].
    Works even with missing hours / irregular cadence.
    Returns an array aligned to df.index (int 0/1).
    """
    def _lead(g: pd.DataFrame) -> pd.Series:
        s = pd.Series(g[target_col].astype(int).to_numpy(), index=g["time"])
        rev = s.iloc[::-1]
        fut = rev.rolling(f"{hours}h", min_periods=1).max().shift(1)   # exclude current hour
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        return pd.Series(fut.to_numpy(), index=g.index)

    return (
        df.groupby(["lat", "lon"], sort=False, group_keys=False)
          .apply(_lead)
          .to_numpy()
          .astype(int)
    )

def build_leadband_labels(
    df: pd.DataFrame,
    t_col: str,
    lead_h: int,
    band_width_h: float,
) -> np.ndarray:
    """
    Lead-band label from continuous t_to_storm_min_h.

    For a given lead_h and band_width_h, positives are rows with:
        0 < t_to_storm_min_h <= lead_h
        and t_to_storm_min_h > lead_h - band_width_h

    So with band_width_h = 24:
      lead_h = 24  -> (0, 24] h
      lead_h = 48  -> (24, 48] h
      lead_h = 120 -> (96, 120] h
    """
    if t_col not in df.columns:
        raise ValueError(f"t_to_storm column '{t_col}' not found in labelled data.")

    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)

    # base sanity: NaNs and non-positive -> no storm in future or outside horizon
    mask = np.isfinite(t) & (t > 0.0)

    lo = float(lead_h - band_width_h)
    hi = float(lead_h)

    band = (
        mask &
        (t > lo) &
        (t <= hi)
    )

    return band.astype(int)

def parse_rules_csv(path: Path) -> pd.DataFrame:
    """
    Flexible schema:
      - lead_h / lead / lead_hours
      - feature
      - sign / direction  (pos/neg, + / -)
      - thr / threshold
    """
    r = pd.read_csv(path)
    cols = {c.lower(): c for c in r.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"Missing expected column among: {names}")

    lead_col = pick("lead_h", "lead", "lead_hours")
    feat_col = pick("feature",)
    sign_col = pick("sign", "direction")
    thr_col  = pick("thr", "threshold")

    out = r[[lead_col, feat_col, sign_col, thr_col]].copy()
    out.columns = ["lead_h", "feature", "sign", "thr"]

    out["lead_h"] = pd.to_numeric(out["lead_h"], errors="coerce").astype("Int64").astype(int)
    out["sign"] = (
        out["sign"].astype(str).str.lower()
           .map({"pos": "pos", "positive": "pos", "+": "pos",
                 "neg": "neg", "negative": "neg", "-": "neg"})
    )
    if out["sign"].isna().any():
        raise ValueError("Could not parse rule sign (expected 'pos'/'neg'). Check rules CSV.")
    out["thr"] = pd.to_numeric(out["thr"], errors="coerce")
    if out["thr"].isna().any():
        raise ValueError("Could not parse numeric thresholds in rules CSV.")
    return out

def score_binary(y_true: np.ndarray, y_hat: np.ndarray):
    """Return Precision, Recall, F1, AUC, PRAUC, Brier, Coverage."""
    P, R, F1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    try:
        AUC = roc_auc_score(y_true, y_hat)
    except Exception:
        AUC = np.nan
    try:
        PRAUC = average_precision_score(y_true, y_hat)
    except Exception:
        PRAUC = np.nan
    try:
        Brier = brier_score_loss(y_true, y_hat.astype(float))
    except Exception:
        Brier = np.nan
    cov = float(y_hat.mean())
    return P, R, F1, AUC, PRAUC, Brier, cov

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Apply single-feature rules and evaluate against either:\n"
            "  (A) time-aware future labels from a binary target (legacy mode), or\n"
            "  (B) lead-band labels from t_to_storm_min_h (lead-band mode)."
        )
    )
    ap.add_argument("--labelled", required=True, help="Path to labelled grid CSV")
    ap.add_argument("--rules", default="results/rules_candidates.csv",
                    help="Rules CSV produced by phase_rules.py")
    # Legacy target (mode A)
    ap.add_argument("--target", default="pregen",
                    help="Binary target column name for legacy future-label mode")
    # Lead-band mode
    ap.add_argument("--t-to-storm-col", default="t_to_storm_min_h",
                    help="Continuous time-to-storm column for lead-band mode "
                         "(if present, lead-band mode is used).")
    ap.add_argument("--band-width-h", type=float, default=24.0,
                    help="Lead band width in hours (default: 24).")
    # Generic rule filtering
    ap.add_argument("--leads", default="",
                    help="Comma list of lead hours to evaluate (e.g., 24,48). "
                         "If empty, use all unique leads in rules CSV.")
    ap.add_argument("--min-f1", type=float, default=0.0,
                    help="Filter rules by an existing F1 column in rules CSV (if present).")
    ap.add_argument("--top-n", type=int, default=0,
                    help="Take only top-N rules per lead by F1 (if present). 0=all.")
    ap.add_argument("--out", default="results/rule_eval_summary.csv",
                    help="Output CSV with per-rule metrics.")
    args = ap.parse_args()

    labelled_path = Path(args.labelled)
    rules_path    = Path(args.rules)
    out_path      = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("== Rule Applier ==")
    print(f"Data    : {labelled_path}")
    print(f"Rules   : {rules_path}")

    # Load & normalize labelled grid
    df = pd.read_csv(labelled_path, parse_dates=["time"])
    df = ensure_sorted(df)

    # Decide mode
    leadband_mode = args.t_to_storm_col in df.columns
    if leadband_mode:
        mode_str = f"LEAD-BAND (t_to_storm = '{args.t_to_storm_col}', band_width={args.band_width_h} h)"
        print(f"Mode    : {mode_str}")
    else:
        if args.target not in df.columns:
            raise ValueError(
                f"Labelled file missing required column for legacy mode: {args.target} "
                f"and no t_to_storm column '{args.t_to_storm_col}' found."
            )
        mode_str = f"LEGACY FUTURE-LABEL (target='{args.target}')"
        print(f"Mode    : {mode_str}")

    N = len(df)
    if leadband_mode:
        # Just show how many rows have a finite t_to_storm
        t = pd.to_numeric(df[args.t_to_storm_col], errors="coerce")
        finite = np.isfinite(t) & (t > 0)
        print(f"Rows: {N:,}  finite t_to_storm_min_h: {int(finite.sum()):,}")
    else:
        pos = int(pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int).sum())
        print(f"Rows: {N:,}  Pos({args.target}): {pos:,}")

    # Load & normalize rules
    rules = parse_rules_csv(rules_path)

    # Optional filtering by min-f1 if the CSV has an F1-like column
    raw_hdr = pd.read_csv(rules_path, nrows=0)
    f1_col = next((c for c in ["F1", "f1", "F1_est", "F1_fix"] if c in raw_hdr.columns), None)
    if f1_col is not None and args.min_f1 > 0:
        raw = pd.read_csv(rules_path)
        cols = {c.lower(): c for c in raw.columns}
        key_lead = cols.get("lead_h") or cols.get("lead") or cols.get("lead_hours")
        key_feat = cols["feature"]
        key_sign = cols.get("sign") or cols.get("direction")
        key_thr  = cols.get("thr") or cols.get("threshold")
        rules = rules.merge(
            raw[[key_lead, key_feat, key_sign, key_thr, f1_col]],
            left_on=["lead_h", "feature", "sign", "thr"],
            right_on=[key_lead, key_feat, key_sign, key_thr],
            how="left",
        )
        rules = rules[rules[f1_col] >= args.min_f1].copy()

    # Optional top-N per lead by F1 (if present)
    if f1_col is not None and args.top_n > 0:
        rules = (
            rules.sort_values(["lead_h", f1_col], ascending=[True, False])
                 .groupby("lead_h", as_index=False, sort=False)
                 .head(args.top_n)
                 .reset_index(drop=True)
        )

    leads = sorted(rules["lead_h"].unique().tolist())
    if args.leads.strip():
        leads = sorted({int(x) for x in args.leads.split(",") if x.strip()})
        rules = rules[rules["lead_h"].isin(leads)].reset_index(drop=True)

    if not len(leads):
        print("No leads found in rules after filtering.", file=sys.stderr)
        sys.exit(2)

    # Precompute label arrays per lead, depending on mode
    labels_by_lead: dict[int, np.ndarray] = {}

    if leadband_mode:
        print("Preparing lead-band labels from t_to_storm_min_h …", flush=True)
        for h in leads:
            y = build_leadband_labels(df, args.t_to_storm_col, h, args.band_width_h)
            labels_by_lead[h] = y
            print(f"  lead +{h}h: band positives={int(y.sum()):,}  coverage={y.mean():.4%}", flush=True)
    else:
        print("Preparing strictly-future labels for legacy mode …", flush=True)
        base = df[["time", "lat", "lon", args.target]].copy()
        for h in leads:
            print(f"  lead +{h}h …", flush=True)
            labels_by_lead[h] = future_max_label_per_point_timeaware(base, args.target, h)

    # Evaluate each rule
    rows = []
    for i, r in rules.iterrows():
        h   = int(r["lead_h"])
        f   = r["feature"]
        sgn = r["sign"]
        thr = float(r["thr"])

        if f not in df.columns:
            print(f"  ! Skipping rule {i} (lead {h}h): feature '{f}' not in data.", flush=True)
            continue

        x = pd.to_numeric(df[f], errors="coerce")
        alert = (x >= thr) if sgn == "pos" else (x <= thr)
        alert = alert.fillna(False).to_numpy().astype(int)

        y = labels_by_lead[h]
        P, R, F1, AUC, PRAUC, Brier, cov = score_binary(y, alert)

        rows.append({
            "lead_h": h,
            "feature": f,
            "sign": sgn,
            "thr": thr,
            "Precision": P,
            "Recall": R,
            "F1": F1,
            "AUC": AUC,
            "PRAUC": PRAUC,
            "Brier": Brier,
            "Coverage": cov,
            "mode": "leadband" if leadband_mode else "legacy",
            "band_width_h": args.band_width_h if leadband_mode else np.nan,
        })

        if (i % 20) == 0:
            print(
                f"  • [{i+1}/{len(rules)}] lead {h:>3}h | {f} ({sgn}) thr={thr:.4f}  "
                f"F1={F1:.3f}  P={P:.3f}  R={R:.3f}  Cov={cov:.4f}",
                flush=True,
            )

    if not rows:
        print("No rules evaluated (empty selection or missing features).", file=sys.stderr)
        sys.exit(2)

    outdf = pd.DataFrame(rows).sort_values(["lead_h", "F1"], ascending=[True, False])
    outdf.to_csv(out_path, index=False)
    print(f"\nSaved rule evaluation -> {out_path}  rows={len(outdf)}", flush=True)

    # Pretty print top-5 per lead
    for h in leads:
        top = outdf[outdf["lead_h"] == h].head(5)
        if len(top):
            print(f"\nTop rules @ lead +{h}h ({'leadband' if leadband_mode else 'legacy'})")
            for _, rr in top.iterrows():
                print(
                    f"  • {rr['feature']:>14s} ({rr['sign']}) thr={rr['thr']:.4f}  "
                    f"F1={rr['F1']:.3f}  P={rr['Precision']:.3f}  R={rr['Recall']:.3f}  "
                    f"Cov={rr['Coverage']:.3f}"
                )

if __name__ == "__main__":
    main()