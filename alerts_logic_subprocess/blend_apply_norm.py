#!/usr/bin/env python
# blend_apply_norm.py  — probit blend (build×relax) with time-aware windows,
#                        optional t_to_storm labels, robust impute, persistence.

import argparse, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# ---------- numerics ----------

def erfinv(x):
    # Winitzki approximation (adequate for thresholds & ranking)
    a = 0.147
    ln = np.log(1 - x**2)
    s = np.sign(x)
    return s * np.sqrt(np.sqrt((2/(np.pi*a) + ln/2.0)**2 - ln/a) - (2/(np.pi*a) + ln/2.0))

def probit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)

# ---------- feature handling ----------

def impute_then_scale(df: pd.DataFrame, feats, scaler):
    # Select & coerce
    X = df[feats].copy()
    # Replace inf with NaN, then median-impute per column
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return scaler.transform(X.to_numpy(float))

# ---------- windows / labels ----------

def recent_build_window(df: pd.DataFrame, prob_col: str, hours: int) -> np.ndarray:
    """Past-window max per (lat,lon), EXCLUDING current hour."""
    def _roll(g):
        s = g.set_index("time")[prob_col]
        # rolling max over past window, then shift(1) to exclude current hour
        r = s.rolling(f"{hours}h", min_periods=1).max().shift(1)
        return pd.Series(r.reindex(g["time"]).to_numpy(), index=g.index)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_roll)
    return out.reindex(df.index).fillna(0.0).to_numpy()

def strict_future_max(df: pd.DataFrame, label_col: str, hours: int) -> np.ndarray:
    """
    Strict-future label: for each row at time t, 1 if any positive occurs in (t, t+hours].
    Implementation: reverse -> rolling max -> shift(1) -> reverse.
    """
    def _lead(g):
        s = g.set_index("time")[label_col].astype(int)
        rev = s.iloc[::-1]
        fut = rev.rolling(f"{hours}h", min_periods=1).max().shift(1)  # exclude current
        fut = fut.iloc[::-1].reindex(g["time"]).fillna(0).astype(int)
        return pd.Series(fut.to_numpy(), index=g.index)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)
    return out.reindex(df.index).to_numpy(dtype=int)

def t_to_storm_label(df: pd.DataFrame, lead_hours: int) -> np.ndarray:
    """
    Label = 1 iff 0 < t_to_storm_min_h <= lead_hours.
    Assumes t_to_storm_min_h is already the strict-future min time to storm (in hours).
    """
    tts = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce").fillna(np.inf).to_numpy()
    y = ((tts > 0) & (tts <= float(lead_hours))).astype(int)
    return y

def persistence_mask(df: pd.DataFrame, flag_col: str, hours: int) -> np.ndarray:
    """Require continuous presence for a time window per (lat,lon)."""
    if hours <= 1:
        return df[flag_col].to_numpy().astype(int)
    def _persist(g):
        s = g.set_index("time")[flag_col].astype(int)
        # mean==1.0 over the window ⇒ all ones in that window
        r = s.rolling(f"{hours}h", min_periods=hours).mean()
        keep = (r.reindex(g["time"]).fillna(0.0).to_numpy() >= 1.0).astype(int)
        return pd.Series(keep, index=g.index)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_persist)
    return out.reindex(df.index).to_numpy(dtype=int)

def throttle_by_hour_quantile_on_flagged(df: pd.DataFrame, score_col: str,
                                         base_flag_col: str, q: float) -> np.ndarray:
    """
    For each hour, among rows where base_flag==1, keep only those with score >= hourly q-quantile.
    If no base_flag rows in an hour, keep none.
    """
    def _keep(g):
        sub = g[g[base_flag_col] == 1]
        if sub.empty:
            return pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        thr = sub[score_col].quantile(q)
        return pd.Series((g[score_col] >= thr).astype(int), index=g.index)
    out = df.groupby(df["time"].dt.floor("H"), sort=False, group_keys=False).apply(_keep)
    return out.reindex(df.index).to_numpy(dtype=int)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Apply probit-blended build+relax to produce alerts (future_window or t_to_storm labels)."
    )
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", default="pregen",
                    help="Binary target column for label-mode=future_window.")
    ap.add_argument("--label-mode", choices=["future_window", "t_to_storm"],
                    default="t_to_storm",
                    help=("Label construction:\n"
                          "  future_window: strict future (t,t+L] from --target column\n"
                          "  t_to_storm   : 0 < t_to_storm_min_h <= lead_hours"))
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on build (0..1)")
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--lead-hours", type=int, default=24)
    ap.add_argument("--thr", type=float, required=True, help="probability threshold after blending")
    ap.add_argument("--persist-hours", type=int, default=1)
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="hourly keep-quantile among flagged (on blended risk)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load & normalize time
    df = pd.read_csv(args.labelled, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    need = {"time","lat","lon"}
    if not need.issubset(df.columns):
        missing = sorted(need - set(df.columns))
        raise SystemExit(f"Missing base columns: {missing}")
    df = (
        df.dropna(subset=["time","lat","lon"])
          .sort_values(["lat","lon","time"], kind="mergesort")
          .reset_index(drop=True)
    )

    # Sanity for labels
    if args.label_mode == "future_window":
        if args.target not in df.columns:
            raise SystemExit(f"Label-mode=future_window but target column '{args.target}' not found.")
    else:
        if "t_to_storm_min_h" not in df.columns:
            raise SystemExit("Label-mode=t_to_storm but column 't_to_storm_min_h' not found.")

    # Models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)

    # Probabilities with robust imputation
    p_build_raw = mb["model"].predict_proba(impute_then_scale(df, mb["features"], mb["scaler"]))[:, 1]
    p_relax_raw = mr["model"].predict_proba(impute_then_scale(df, mr["features"], mr["scaler"]))[:, 1]

    # Recent build window (exclude current hour)
    df["_p_build"] = p_build_raw
    build_recent = recent_build_window(df, "_p_build", args.build_window)
    df.drop(columns=["_p_build"], inplace=True)

    # Probit blend
    z = args.alpha * probit(build_recent) + (1.0 - args.alpha) * probit(p_relax_raw)
    p_blend = 1.0 / (1.0 + np.exp(-z))

    # Labels for evaluation
    if args.label_mode == "future_window":
        # strict future (t, t+L] from binary target
        df_target = df[["time","lat","lon", args.target]].copy()
        df_target[args.target] = (
            pd.to_numeric(df_target[args.target], errors="coerce")
              .fillna(0)
              .astype(int)
        )
        y = strict_future_max(df_target, args.target, args.lead_hours)
        label_desc = f"strict-future (t,t+{args.lead_hours}h] of '{args.target}'"
    else:
        # t_to_storm window
        y = t_to_storm_label(df, args.lead_hours)
        label_desc = f"0 < t_to_storm_min_h <= {args.lead_hours}h"

    # Alerts
    out = df[["time","lat","lon"]].copy()
    out["risk"] = p_blend
    out["alert"] = (p_blend >= args.thr).astype(int)

    # Persistence
    if args.persist_hours > 1:
        keep = persistence_mask(out.assign(time=df["time"]), "alert", args.persist_hours)
        out["alert"] = (out["alert"] & keep).astype(int)

    # Throttle: keep top-q per hour among rows already flagged
    tmp = out.copy()
    tmp["time"] = df["time"]
    tmp["score"] = out["risk"]
    keep_top = throttle_by_hour_quantile_on_flagged(tmp, "score", "alert", q=args.quantile)
    out["alert_throttled"] = (out["alert"] & keep_top).astype(int)

    # Quick metrics (binary against chosen label)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, out["alert_throttled"].to_numpy(), average="binary", zero_division=0
    )
    cov = float(out["alert_throttled"].mean())
    print(f"Rows: {len(out)}  Positives({label_desc}): {int(y.sum())}")
    print(f"Lead +{args.lead_hours}h  Precision={pr:.3f}  Recall={rc:.3f}  "
          f"F1={f1:.3f}  Coverage={cov:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Wrote {args.out}  | alerts_throttled: {int(out['alert_throttled'].sum())}/{len(out)}")


if __name__ == "__main__":
    main()