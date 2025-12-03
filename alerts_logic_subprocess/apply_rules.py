# apply_rules.py
# Apply one or more logical rules ('feature|sign|thr') to a labelled grid
# and emit alert flags consistent with the pipeline.
# Adds: NaN-safe comparisons, deterministic sort, optional persistence and hourly throttle.

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

SIGN_MAP = {
    "pos": "pos", "positive": "pos", "+": "pos", ">=": "pos", ">": "pos",
    "neg": "neg", "negative": "neg", "-": "neg", "<=": "neg", "<": "neg"
}

def parse_rule(rule_str: str):
    """
    'feature|sign|thr'  (whitespace tolerated; sign in {pos,neg,+,-,>=,<=,>,<})
    Returns (feature:str, sign:'pos'|'neg', thr:float)
    """
    try:
        parts = [p.strip() for p in rule_str.split("|")]
        if len(parts) != 3:
            raise ValueError
        feat, sign_raw, thr_raw = parts
        sign_key = SIGN_MAP.get(sign_raw.lower())
        if sign_key is None:
            raise ValueError(f"bad sign '{sign_raw}'")
        thr = float(thr_raw)
        return feat, sign_key, thr
    except Exception as e:
        raise ValueError(f"Invalid rule format '{rule_str}'. Expected 'feature|sign|thr'.") from e

def make_mask(df: pd.DataFrame, feat: str, sign: str, thr: float) -> np.ndarray:
    if feat not in df.columns:
        raise ValueError(f"Missing feature '{feat}' in dataframe")
    x = pd.to_numeric(df[feat], errors="coerce").to_numpy()
    finite = np.isfinite(x)
    if sign == "pos":
        m = x >= thr
    else:
        m = x <= thr
    # NaNs/±Inf never trigger
    m &= finite
    return m

def apply_persistence(df: pd.DataFrame, base_mask: np.ndarray, hours: int) -> np.ndarray:
    """Require >=hours consecutive alert hours per (lat,lon). hours<=1 -> no-op."""
    if hours <= 1:
        return base_mask
    out = np.zeros(len(df), dtype=bool)
    # We assume hourly cadence; use rolling sum per group on the boolean array
    for _, idx in df.groupby(["lat","lon"], sort=False).indices.items():
        s = base_mask[idx].astype(int)
        if len(s) == 0:
            continue
        # cumulative sum trick for O(n)
        c = np.cumsum(s)
        win = c - np.r_[0, c[:-hours]]
        keep = np.zeros_like(s, dtype=bool)
        keep[hours-1:] = win[hours-1:] >= hours
        out[idx] = keep
    return out

def throttle_hourly(df: pd.DataFrame, scores: np.ndarray, base_mask: np.ndarray, q: float) -> np.ndarray:
    """
    Keep only top q per hour among rows where base_mask==True. q in (0,1].
    If an hour has no base_mask==True, keep none that hour.
    """
    if q is None:
        return base_mask
    kept = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(pd.to_datetime(df["time"]).dt.floor("H"), sort=False).indices.items():
        cand = base_mask[idx]
        if not np.any(cand):
            continue
        k = max(1, int(np.ceil(np.sum(cand) * (1.0 - (1.0 - q)))))  # effectively top-q; stable
        # compute threshold among candidates
        s = scores[idx][cand]
        if k >= len(s):
            thr = np.min(s)
        else:
            thr = np.partition(s, -k)[-k]
        subkeep = np.zeros_like(cand)
        subkeep[cand] = s >= thr
        kept[idx] = subkeep
    return kept

def main():
    ap = argparse.ArgumentParser(description="Apply selected logical rules to produce alerts.")
    ap.add_argument("--labelled", required=True, help="Grid-labelled CSV (training-compatible).")
    ap.add_argument("--rules", nargs="+", default=None,
                    help="One or more rule strings 'feature|sign|thr'.")
    ap.add_argument("--rules-file", default=None,
                    help="Optional file with one 'feature|sign|thr' per line.")
    ap.add_argument("--ops", default="AND", choices=["AND","OR"], help="Combine rule logic (default AND).")
    ap.add_argument("--persist-hours", type=int, default=1, help="Require consecutive hours per point (default 1).")
    ap.add_argument("--hourly-quantile", type=float, default=None,
                    help="Optional per-hour top-quantile throttle on a simple score (rule count). e.g. 0.90")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    # Load
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    for c in ("time","lat","lon"):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in labelled file.")
    # Normalize time and order for deterministic behavior
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time","lat","lon"]).sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    # Collect rules
    rule_strs = []
    if args.rules:
        rule_strs.extend(args.rules)
    if args.rules_file:
        with open(args.rules_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    rule_strs.append(s)
    if not rule_strs:
        raise ValueError("No rules provided. Use --rules and/or --rules-file.")

    parsed = [parse_rule(s) for s in rule_strs]

    # Build masks
    masks = [make_mask(df, feat, sign, thr) for (feat, sign, thr) in parsed]
    if len(masks) == 1:
        base_alert = masks[0]
    elif args.ops == "AND":
        base_alert = np.logical_and.reduce(masks)
    else:
        base_alert = np.logical_or.reduce(masks)

    # Optional persistence
    alert = apply_persistence(df, base_alert, hours=args.persist_hours)

    # Optional hourly throttle (score = number of satisfied rules at a row)
    if args.hourly_quantile is not None:
        rule_count = np.sum(np.vstack(masks), axis=0).astype(float)
        keep = throttle_hourly(df, rule_count, alert, q=args.hourly_quantile)
        alert = alert & keep

    # Emit
    out_df = df.loc[:, ["time","lat","lon"]].copy()
    out_df["alert_rule"] = alert.astype(int)

    total = int(out_df["alert_rule"].sum())
    frac = total / len(out_df)
    print(f"Rows: {len(out_df):,}  Alerts: {total:,}  Coverage={frac:.3%}  "
          f"Rules={len(parsed)} ({args.ops}), Persist={args.persist_hours}, "
          f"Throttle={'None' if args.hourly_quantile is None else args.hourly_quantile:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote -> {out}")

if __name__ == "__main__":
    main()