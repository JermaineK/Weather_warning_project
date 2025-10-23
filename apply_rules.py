# apply_rules.py
# Take one or more logical rules (from rule_combo_summary.csv)
# and generate an alert CSV consistent with the rest of the pipeline.

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

def make_mask(df, rule_str):
    """Parse 'feature|sign|thr' -> boolean mask"""
    try:
        feat, sign, thr = rule_str.split("|")
        thr = float(thr)
    except Exception as e:
        raise ValueError(f"Invalid rule format: {rule_str}") from e
    if feat not in df.columns:
        raise ValueError(f"Missing feature '{feat}' in dataframe")
    x = df[feat].to_numpy()
    if sign.lower().startswith("pos"):
        return (x >= thr)
    else:
        return (x <= thr)

def main():
    ap = argparse.ArgumentParser(description="Apply selected logical rules to produce alerts")
    ap.add_argument("--labelled", required=True, help="grid-labelled CSV (same as training input)")
    ap.add_argument("--rules", required=True, nargs="+", help="one or more rule strings 'feature|sign|thr'")
    ap.add_argument("--ops", default="AND", choices=["AND","OR"], help="combine rule logic (default: AND)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    print(f"Loaded {len(df):,} rows")

    masks = [make_mask(df, r) for r in args.rules]
    if len(masks) == 1:
        alert = masks[0]
    elif args.ops == "AND":
        alert = np.logical_and.reduce(masks)
    else:
        alert = np.logical_or.reduce(masks)

    df["alert_rule"] = alert.astype(int)
    total = int(df["alert_rule"].sum())
    frac = total / len(df)
    print(f"Alerts: {total:,} / {len(df):,}  ({frac:.3%}) using {args.ops} of {len(masks)} rules")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["time","lat","lon","alert_rule"]].to_csv(out, index=False)
    print(f"Wrote → {out}")

if __name__ == "__main__":
    main()