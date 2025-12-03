#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leadband_rule_miner.py  —  lead-aware rule mining on a rich labelled grid.

Adapted from apply_rules.py, but instead of just emitting a single alert flag,
this script:

  • Reads a rich grid CSV/Parquet (with time, lat, lon, t_to_storm_min_h, features)
  • Applies one or more threshold rules 'feature|sign|thr'
  • Defines target labels per lead band using t_to_storm_min_h
  • Computes simple metrics per (rule, band), e.g. coverage, precision, recall, F1
  • Optionally writes out per-row rule alerts for later reuse

This is the "slow-tick-aware" miner: your continuous lead lives in t_to_storm_min_h,
and bands can be chosen to respect a 22–23 h cycle (e.g. 0–24, 24–48, …).
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ---------------- rule parsing / masks (from apply_rules.py) ----------------

SIGN_MAP = {
    "pos": "pos", "positive": "pos", "+": "pos", ">=": "pos", ">": "pos",
    "neg": "neg", "negative": "neg", "-": "neg", "<=": "neg", "<": "neg"
}

def parse_rule(rule_str: str) -> Tuple[str, str, float]:
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
    """NaN-safe threshold mask for one feature."""
    if feat not in df.columns:
        raise ValueError(f"Missing feature '{feat}' in dataframe")
    x = pd.to_numeric(df[feat], errors="coerce").to_numpy()
    finite = np.isfinite(x)
    if sign == "pos":
        m = x >= thr
    else:
        m = x <= thr
    m &= finite
    return m

# ---------------- lead-band parsing ----------------

def parse_band_spec(spec: str) -> List[Tuple[float, float]]:
    """
    Parse lead-band spec like:
      "0-24,24-48,48-72,72-96"

    Returns list of (lo, hi) in hours.
    """
    out: List[Tuple[float, float]] = []
    if not spec:
        raise ValueError("--lead-bands is required (e.g. '0-24,24-48,48-72').")
    for part in spec.split(","):
        s = part.strip()
        if not s:
            continue
        if "-" not in s:
            raise ValueError(f"Bad band '{s}'; expected 'lo-hi'.")
        a, b = [p.strip() for p in s.split("-", 1)]
        lo = float(a); hi = float(b)
        if hi <= lo:
            raise ValueError(f"Bad band '{s}'; hi must be > lo.")
        out.append((lo, hi))
    if not out:
        raise ValueError("No valid bands parsed from --lead-bands.")
    return out

# ---------------- metrics ----------------

def compute_band_metrics(
    lead: np.ndarray,
    rule_mask: np.ndarray,
    band_lo: float,
    band_hi: float,
    margin_mult: float = 3.0,
) -> Dict[str, Any]:
    """
    Treat "in this lead band" as the target positive class.

    Positives:   band_lo <= lead < band_hi
    Negatives:   outside [band_lo, band_hi * margin_mult] but finite lead
                 (we ignore 'very far away' cells and NaNs for metric stability)

    rule_mask specifies which rows the rule fires on.

    Returns basic counts + precision/recall/F1.
    """
    finite = np.isfinite(lead)
    band = finite & (lead >= band_lo) & (lead < band_hi)

    # "negative pool" = reasonably close in time but not in this band
    max_lead = band_hi * margin_mult
    neg_pool = finite & (lead >= 0.0) & (lead <= max_lead) & (~band)

    if not (band.any() and neg_pool.any()):
        return dict(
            n_pos=int(band.sum()),
            n_neg=int(neg_pool.sum()),
            tp=0, fp=0, fn=int(band.sum()),
            precision=np.nan, recall=np.nan, f1=np.nan,
        )

    y_true = np.zeros(len(lead), dtype=np.int8)
    y_true[band] = 1
    # Only evaluate within band ∪ neg_pool
    eval_mask = band | neg_pool

    y = y_true[eval_mask]
    yhat = rule_mask[eval_mask].astype(bool)

    tp = int(np.sum(yhat & (y == 1)))
    fp = int(np.sum(yhat & (y == 0)))
    fn = int(np.sum((~yhat) & (y == 1)))

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if tp + fp == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = np.nan
    else:
        recall = tp / (tp + fn)
    if precision is np.nan or recall is np.nan or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(
        n_pos=n_pos,
        n_neg=n_neg,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float(precision) if precision == precision else np.nan,
        recall=float(recall) if recall == recall else np.nan,
        f1=float(f1) if f1 == f1 else np.nan,
    )

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Lead-aware rule miner: evaluate rules per t_to_storm lead band."
    )
    ap.add_argument("--labelled", required=True,
                    help="Rich grid CSV/Parquet (with time,lat,lon,t_to_storm_min_h,features).")
    ap.add_argument("--lead-col", default="t_to_storm_min_h",
                    help="Column with continuous lead in hours (default: t_to_storm_min_h).")
    ap.add_argument("--lead-bands", required=True,
                    help="Comma-separated bands 'lo-hi', e.g. '0-24,24-48,48-72,72-96'.")
    ap.add_argument("--rules", nargs="+", default=None,
                    help="One or more rule strings 'feature|sign|thr'.")
    ap.add_argument("--rules-file", default=None,
                    help="Optional file with one 'feature|sign|thr' per line.")
    ap.add_argument("--margin-mult", type=float, default=3.0,
                    help="Negatives are limited to lead <= hi * margin_mult (default 3.0).")
    ap.add_argument("--out-summary", required=True,
                    help="Output CSV with metrics per (rule, band).")
    ap.add_argument("--out-alerts", default=None,
                    help="Optional CSV(.gz) with time,lat,lon,lead + rule alerts (one col per rule).")
    args = ap.parse_args()

    path = Path(args.labelled)
    if not path.exists():
        raise SystemExit(f"[fatal] labelled grid not found: {path}")

    # Load (CSV or Parquet)
    low = str(path).lower()
    if low.endswith((".parquet", ".parq", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, compression="infer", low_memory=False)

    for c in ("time", "lat", "lon"):
        if c not in df.columns:
            raise SystemExit(f"[fatal] labelled grid missing column '{c}'.")

    if args.lead_col not in df.columns:
        raise SystemExit(f"[fatal] labelled grid missing lead column '{args.lead_col}'.")

    # Normalize time & basic types
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    lead = pd.to_numeric(df[args.lead_col], errors="coerce").to_numpy(dtype=float)

    mask_valid = np.isfinite(lead) & (lead >= 0.0)
    n_valid = int(mask_valid.sum())
    if n_valid == 0:
        raise SystemExit(f"[fatal] no finite non-negative values in {args.lead_col}.")

    print(f"[lead] using {n_valid:,} rows with finite {args.lead_col} >= 0.", flush=True)

    # Collect rules
    rule_strs: List[str] = []
    if args.rules:
        rule_strs.extend(args.rules)
    if args.rules_file:
        with open(args.rules_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    rule_strs.append(s)
    if not rule_strs:
        raise SystemExit("No rules provided. Use --rules and/or --rules-file.")

    parsed_rules = [parse_rule(s) for s in rule_strs]
    bands = parse_band_spec(args.lead_bands)

    # Precompute masks per rule (over all rows)
    rule_masks: List[np.ndarray] = []
    for (feat, sign, thr) in parsed_rules:
        m = make_mask(df, feat, sign, thr)
        rule_masks.append(m)
        cov = float(m[mask_valid].mean())
        print(f"[rule] {feat}|{sign}|{thr:g}  coverage={cov:.3%} on valid-lead rows.", flush=True)

    # Metrics per (rule, band)
    rows: List[Dict[str, Any]] = []
    for band_idx, (lo, hi) in enumerate(bands):
        for rule_idx, ((feat, sign, thr), m_rule) in enumerate(zip(parsed_rules, rule_masks)):
            stats = compute_band_metrics(
                lead=lead,
                rule_mask=m_rule,
                band_lo=lo,
                band_hi=hi,
                margin_mult=float(args.margin_mult),
            )
            row = dict(
                band_index=band_idx,
                band_lo_h=lo,
                band_hi_h=hi,
                rule_index=rule_idx,
                rule=f"{feat}|{sign}|{thr:g}",
                feature=feat,
                sign=sign,
                thr=thr,
            )
            row.update(stats)
            rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = Path(args.out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[ok] wrote lead-band rule summary -> {out_path} rows={len(summary)}", flush=True)

    # Optional: emit per-row rule alerts to drive later phases
    if args.out_alerts:
        alerts = df.loc[:, ["time", "lat", "lon", args.lead_col]].copy()
        for (feat, sign, thr), m_rule in zip(parsed_rules, rule_masks):
            colname = f"alert_{feat}_{sign}_{str(thr).replace('.','p')}"
            alerts[colname] = m_rule.astype(np.int8)
        out_a = Path(args.out_alerts)
        out_a.parent.mkdir(parents=True, exist_ok=True)
        comp = "gzip" if out_a.name.lower().endswith(".gz") else "infer"
        alerts.to_csv(out_a, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")
        print(f"[ok] wrote per-row alerts -> {out_a} rows={len(alerts)}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise