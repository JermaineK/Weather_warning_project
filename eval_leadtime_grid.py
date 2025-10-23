# eval_leadtime_grid.py
import argparse, warnings, numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="CSV(.gz) with lat,lon,time, target")
    ap.add_argument("--model",    required=True, help="joblib dict: {'model','scaler','features'}")
    ap.add_argument("--target",   required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, default=[24,48], help="e.g. 6 12 24 48")
    return ap.parse_args()

def load_model(path):
    m = joblib.load(path)
    use = m["features"]
    sc  = m.get("scaler", None)
    clf = m["model"]
    return use, sc, clf

def prep_df(labelled, use, target):
    # Load
    df = pd.read_csv(labelled, parse_dates=["time"])
    # Keep only what we need
    keep_cols = list({*use, "lat","lon","time", target})
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Labelled file missing columns: {missing}")
    df = df[keep_cols].copy()

    # Sort for time-based rolling
    df.sort_values(["lat","lon","time"], inplace=True, kind="mergesort")
    # Drop rows with NaNs in model features/target
    df = df.dropna(subset=use + [target]).reset_index(drop=True)
    return df

def predict_probs(df, use, sc, clf):
    X = df[use].to_numpy(float)
    if sc is not None:
        X = sc.transform(X)
    p = clf.predict_proba(X)[:,1]
    return p

def future_max_within_hours(df, label_col, hours):
    """
    For each (lat,lon), compute max of label within the *future* hours window,
    aligned to the current row. Returns a 1D int array (0/1), length = len(df).
    Memory-safe: works per group, no wide reindex.
    """
    out_parts = []
    # Iterate groups in order; align result to original row index to allow concat without reorder
    for (_, _), g in df.groupby(["lat","lon"], sort=False):
        # Ensure time-sorted within group (it already is after prep_df)
        y = g[label_col].astype(int).reset_index(drop=True)
        t = g["time"].reset_index(drop=True)

        # Reverse to turn “future max” into a past-looking rolling
        y_rev = y.iloc[::-1]
        t_rev = t.iloc[::-1]

        # Use time-based rolling with a DatetimeIndex
        y_rev.index = t_rev
        fut_max_rev = y_rev.rolling(f"{hours}h", min_periods=1).max()

        fut_max = fut_max_rev.iloc[::-1].reset_index(drop=True).astype(int)

        s = pd.Series(fut_max.values, index=g.index)  # align to original row indices
        out_parts.append(s)

    out = pd.concat(out_parts).sort_index()
    return out.to_numpy(dtype=int)

def main():
    args = parse_args()
    use, sc, clf = load_model(args.model)
    df = prep_df(args.labelled, use, args.target)

    # Base truth at current time (coincident)
    base = df[args.target].astype(int).to_numpy()
    print(f"Rows evaluated: {len(df):,}  Positives (coincident {args.target}): {base.sum():,}")

    # Predict probabilities once
    p = predict_probs(df, use, sc, clf)

    # Report coincident metrics (informational)
    try:
        auc0   = roc_auc_score(base, p)
        prauc0 = average_precision_score(base, p)
        brier0 = brier_score_loss(base, p)
        print(f"[COINCIDENT] AUC={auc0:.3f}  PRAUC={prauc0:.3f}  Brier={brier0:.3f}")
    except Exception:
        # For extremely imbalanced slices, metrics can fail; ignore
        pass

    # Evaluate each lead independently (no giant reindex)
    for h in args.lead_hours:
        y = future_max_within_hours(df, args.target, hours=h)
        auc   = roc_auc_score(y, p)
        prauc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        pos   = int(y.sum())
        print(f"Lead +{h:>2}h  →  AUC={auc:.3f}  PRAUC={prauc:.3f}  Brier={brier:.3f}  Pos={pos:,}/{len(y):,}")

if __name__ == "__main__":
    main()