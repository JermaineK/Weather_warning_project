# rule_applier.py
# Apply single-feature rules from rules_candidates.csv and evaluate metrics.
# Uses only groupby.transform (no deprecated groupby.apply patterns).

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

# ---------- helpers (no deprecated patterns) ----------

def future_max_label_per_point(df: pd.DataFrame, target_col: str, hours: int) -> np.ndarray:
    """
    For each (lat,lon) point, compute the future OR-current max of target over the next `hours`
    EXCLUDING the current hour (i.e., look-ahead window).
    Assumes regular hourly sampling per point.
    Returns an array aligned to df.index of 0/1 ints.
    """
    # Reverse-roll trick per point via transform (no apply):
    def _roll_future_max_excl_current(s: pd.Series) -> pd.Series:
        # reverse time order for that point
        rev = s.iloc[::-1]
        # rolling over 'hours' *rows* (we are hourly sampled), max, then shift by +1 to exclude "current"
        fut = rev.rolling(window=hours, min_periods=1).max().shift(1)
        return fut.iloc[::-1].fillna(0)

    return (
        df.groupby(["lat", "lon"], sort=False)[target_col]
          .transform(_roll_future_max_excl_current)
          .to_numpy()
          .astype(int)
    )

def parse_rules_csv(path: Path) -> pd.DataFrame:
    """
    Accepts a flexible schema. Expected columns (any reasonable variation is OK):
      - lead_h  or  lead or lead_hours
      - feature
      - sign  or  direction  (values like 'pos' / 'neg')
      - thr   or  threshold
      (others will be ignored)
    """
    r = pd.read_csv(path)
    cols = {c.lower(): c for c in r.columns}

    # Map to canonical names
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        raise KeyError(f"Missing expected column among: {names}")

    lead_col = pick("lead_h", "lead", "lead_hours")
    feat_col = pick("feature",)
    sign_col = pick("sign", "direction")
    thr_col  = pick("thr", "threshold")

    out = r[[lead_col, feat_col, sign_col, thr_col]].copy()
    out.columns = ["lead_h", "feature", "sign", "thr"]

    # Normalize
    out["lead_h"] = out["lead_h"].astype(float).astype(int)
    out["sign"] = out["sign"].astype(str).str.lower().map(
        {"pos": "pos", "positive": "pos", "+": "pos", "neg": "neg", "negative": "neg", "-": "neg"}
    )
    if out["sign"].isna().any():
        raise ValueError("Could not parse rule sign (expected 'pos'/'neg'). Check rules CSV.")
    out["thr"] = pd.to_numeric(out["thr"], errors="coerce")
    if out["thr"].isna().any():
        raise ValueError("Could not parse numeric thresholds in rules CSV.")
    return out

def score_binary(y_true: np.ndarray, y_hat: np.ndarray):
    """Return Precision, Recall, F1, AUC, PRAUC, Brier, Coverage."""
    # y_hat is binary 0/1; for AUC and Brier we’ll use it as score/prob 0/1
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
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
        description="Apply single-feature rules and evaluate against future storm labels."
    )
    ap.add_argument("--labelled", required=True, help="Path to labelled grid CSV (e.g., data\\grid_labelled_FMA_phasecols.csv.gz)")
    ap.add_argument("--rules", default="results/rules_candidates.csv", help="Rules CSV produced by phase_rules.py")
    ap.add_argument("--target", default="pregen", help="Target column name (default: pregen)")
    ap.add_argument("--leads", default="", help="Comma list of lead hours to evaluate (e.g., 24,48). If empty, use all in rules.")
    ap.add_argument("--min-f1", type=float, default=0.0, help="Optional: filter rules by an existing F1 column in rules CSV (if present).")
    ap.add_argument("--top-n", type=int, default=0, help="Optional: take only top-N rules per lead by F1 column in rules CSV (if present). 0=all.")
    ap.add_argument("--out", default="results/rule_eval_summary.csv", help="Output CSV with per-rule metrics.")
    args = ap.parse_args()

    labelled_path = Path(args.labelled)
    rules_path    = Path(args.rules)
    out_path      = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("== Rule Applier ==")
    print(f"Data    : {labelled_path}")
    print(f"Rules   : {rules_path}")
    print(f"Target  : {args.target}")

    # Load labelled grid
    usecols = ["time", "lat", "lon", args.target]  # features will be looked up on demand
    df = pd.read_csv(labelled_path, parse_dates=["time"])
    for c in ["time", "lat", "lon", args.target]:
        if c not in df.columns:
            raise ValueError(f"Labelled file missing required column: {c}")

    N = len(df)
    pos = int(df[args.target].sum())
    print(f"Rows: {N:,}  Pos({args.target}): {pos:,}")

    # Load & normalize rules
    rules = parse_rules_csv(rules_path)

    # Optional filtering by min-f1 if the CSV has an F1 column
    f1_col = None
    for cand in ["F1", "f1", "F1_est", "F1_fix"]:
        if cand in pd.read_csv(rules_path, nrows=0).columns:
            f1_col = cand
            break
    if f1_col is not None and args.min_f1 > 0:
        raw = pd.read_csv(rules_path)
        raw_cols = {c.lower(): c for c in raw.columns}
        # bring F1 alongside normalized rules
        rules = rules.merge(
            raw[[raw_cols.get("lead_h","lead_h") if "lead_h" in raw_cols else raw_cols.get("lead","lead"),
                 raw_cols.get("feature","feature"),
                 raw_cols.get("sign","sign") if "sign" in raw_cols else raw_cols.get("direction","direction"),
                 raw_cols.get("thr","thr") if "thr" in raw_cols else raw_cols.get("threshold","threshold"),
                 f1_col]],
            left_on=["lead_h","feature","sign","thr"],
            right_on=[
                raw_cols.get("lead_h","lead_h") if "lead_h" in raw_cols else raw_cols.get("lead","lead"),
                raw_cols.get("feature","feature"),
                raw_cols.get("sign","sign") if "sign" in raw_cols else raw_cols.get("direction","direction"),
                raw_cols.get("thr","thr") if "thr" in raw_cols else raw_cols.get("threshold","threshold"),
            ],
            how="left"
        )
        rules = rules[rules[f1_col] >= args.min_f1].copy()

    # Optional top-N per lead by F1 (if present)
    if f1_col is not None and args.top_n > 0:
        rules = (
            rules.sort_values([ "lead_h", f1_col ], ascending=[True, False])
                 .groupby("lead_h", as_index=False, sort=False)
                 .head(args.top_n)
                 .reset_index(drop=True)
        )

    leads = sorted(rules["lead_h"].unique().tolist())
    if args.leads.strip():
        leads = sorted(list({int(x) for x in args.leads.split(",") if x.strip()}))
        rules = rules[rules["lead_h"].isin(leads)].reset_index(drop=True)

    # Precompute labels for each lead (future max per point; excludes current hour)
    labels_by_lead = {}
    for h in leads:
        print(f"Preparing labels for lead +{h}h …", flush=True)
        labels_by_lead[h] = future_max_label_per_point(df, args.target, h)

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

        # Build the binary alert according to sign
        if sgn == "pos":
            alert = (df[f].to_numpy() >= thr)
        else:  # "neg"
            alert = (df[f].to_numpy() <= thr)

        y = labels_by_lead[h]
        P, R, F1, AUC, PRAUC, Brier, cov = score_binary(y, alert.astype(int))

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
            "Coverage": cov
        })

        if (i % 20) == 0:
            print(f"  • [{i+1}/{len(rules)}] lead {h:>2}h | {f} ({sgn}) thr={thr:.4f}  F1={F1:.3f}  P={P:.3f}  R={R:.3f}", flush=True)

    if not rows:
        print("No rules evaluated (empty selection or missing features).", file=sys.stderr)
        sys.exit(2)

    outdf = pd.DataFrame(rows).sort_values(["lead_h", "F1"], ascending=[True, False])
    outdf.to_csv(out_path, index=False)
    print(f"\nSaved rule evaluation → {out_path}  rows={len(outdf)}", flush=True)

    # Pretty print top-5 per lead
    for h in leads:
        top = outdf[outdf["lead_h"] == h].head(5)
        if len(top):
            print(f"\nTop rules @ lead +{h}h")
            for _, rr in top.iterrows():
                print(f"  • {rr['feature']:>10s} ({rr['sign']}) thr={rr['thr']:.4f}  "
                      f"F1={rr['F1']:.3f}  P={rr['Precision']:.3f}  R={rr['Recall']:.3f}  "
                      f"Cov={rr['Coverage']:.3f}")

if __name__ == "__main__":
    main()