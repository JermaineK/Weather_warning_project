# blend_apply_norm.py  (patched)
import argparse, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

def probit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.sqrt(2) * erfinv(2 * p - 1)

def erfinv(x):
    # Winitzki approximation
    a = 0.147
    ln = np.log(1 - x**2)
    s = np.sign(x)
    return s * np.sqrt(np.sqrt((2/(np.pi*a) + ln/2.0)**2 - ln/a) - (2/(np.pi*a) + ln/2.0))

def impute_then_scale(df, feats, scaler, fill_map=None):
    X = df[feats].copy()
    if fill_map:
        for k, v in fill_map.items():
            if k in X.columns:
                X[k] = X[k].fillna(v)
    return scaler.transform(X.to_numpy(float))

def recent_build_window(df, prob_col, hours):
    # compute, per grid point, the max over the past `hours`, shifted by 1 (so “recent up to previous hour”)
    def _roll(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)[prob_col].max().shift(1)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_roll)
    return out.reindex(df.index).fillna(0.0).to_numpy()

def future_max_label(df, label_col, hours):
    def _lead(g):
        return g.rolling(f"{hours}h", on="time", min_periods=1)[label_col].max().shift(-1)
    out = df.groupby(["lat","lon"], sort=False, group_keys=False).apply(_lead)
    return out.reindex(df.index).fillna(0).to_numpy().astype(int)

def throttle_by_quantile(df, score_col, q=0.90):
    def keep_top(g):
        thr = np.quantile(g[score_col].to_numpy(), q)
        return (g[score_col] >= thr).astype(int)
    return df.groupby("time", sort=False, group_keys=False).apply(keep_top).to_numpy()

def persist_mask(df, flag_col, hours):
    def _persist(g):
        h = g[flag_col].astype(int).to_numpy()
        if hours <= 1:
            return h
        W = hours
        csum = np.cumsum(h)
        win = np.zeros_like(h)
        for i in range(len(h)):
            a = max(0, i - W + 1)
            win[i] = csum[i] - (csum[a - 1] if a > 0 else 0)
        return (win >= W).astype(int)
    return df.groupby(["lat","lon"], sort=False).apply(_persist).reset_index(level=[0,1], drop=True).to_numpy()

def main():
    ap = argparse.ArgumentParser(description="Apply probit-blended build+relax to produce alerts.")
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--relax", required=True)
    ap.add_argument("--target", default="pregen", choices=["storm","near_storm","pregen"])
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on build (0..1)")
    ap.add_argument("--build-window", type=int, default=24)
    ap.add_argument("--lead-hours", type=int, default=24)
    ap.add_argument("--thr", type=float, required=True, help="probability threshold after calibration")
    ap.add_argument("--persist-hours", type=int, default=1)
    ap.add_argument("--quantile", type=float, default=0.90, help="hourly throttle keep-quantile")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])

    # load models
    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)

    # build probs
    Xb = impute_then_scale(df, mb["features"], mb["scaler"], fill_map={})
    p_build_raw = mb["model"].predict_proba(Xb)[:, 1]
    df["p_build"] = p_build_raw  # <-- add column so rolling can find it

    # relax probs (simple impute→scale)
    Xr = df[mr["features"]].copy()
    Xr = Xr.fillna(Xr.median(numeric_only=True))
    Xr = mr["scaler"].transform(Xr.to_numpy(float))
    p_relax_raw = mr["model"].predict_proba(Xr)[:, 1]

    # recent build window
    build_recent = recent_build_window(df, "p_build", args.build_window)
    # (optional) drop temp column to keep output clean
    df.drop(columns=["p_build"], inplace=True)

    # probit-blend and squash back with logistic
    z = args.alpha * probit(build_recent) + (1 - args.alpha) * probit(p_relax_raw)
    p_blend = 1.0 / (1.0 + np.exp(-z))

    # label for lead window
    y = future_max_label(df, args.target, args.lead_hours)

    # alerts
    out = df[["time", "lat", "lon"]].copy()
    out["risk"] = p_blend
    out["alert"] = (p_blend >= args.thr).astype(int)

    # persistence (optional)
    if args.persist_hours > 1:
        keep = persist_mask(out, "alert", args.persist_hours)
        out["alert"] = (out["alert"] & keep).astype(int)

    # throttle top-q by hour
    kept = throttle_by_quantile(out.assign(score=out["risk"]), "score", q=args.quantile)
    out["alert_throttled"] = (out["alert"] & kept).astype(int)

    # quick metrics
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, out["alert_throttled"].to_numpy(), average="binary", zero_division=0
    )
    cov = out["alert_throttled"].mean()
    print(f"Rows: {len(out)}  Pos(+{args.lead_hours}h): {int(y.sum())}")
    print(f"Lead +{args.lead_hours}h  Precision={pr:.3f} Recall={rc:.3f} F1={f1:.3f}  Coverage={cov:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  | alerts_throttled: {int(out['alert_throttled'].sum())}/{len(out)}")

if __name__ == "__main__":
    main()