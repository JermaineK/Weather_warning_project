# eval_lead_quick.py  (windowed lead labels w/out time reindex)
import argparse, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def future_max_window(s: pd.Series, hours: int) -> pd.Series:
    # s is already per-grid-cell, hourly-regular, 0/1 integers
    y = s.astype(int)
    # Shift by 1 to EXCLUDE the current hour from the "future" window
    fut = y.iloc[::-1].shift(1).rolling(window=hours, min_periods=1).max()
    return fut.iloc[::-1].fillna(0).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="pregen", choices=["pregen","near_storm","storm"])
    ap.add_argument("--leads", nargs="+", type=int, default=[6,12,24,36,48])
    args = ap.parse_args()

    m = joblib.load(args.model)
    use = m["features"]

    df = (pd.read_csv(args.labelled, parse_dates=["time"])
            .dropna(subset=use+[args.target])
            .drop_duplicates(subset=["lat","lon","time"])
            .sort_values(["lat","lon","time"], kind="mergesort"))

    X = m["scaler"].transform(df[use].to_numpy(float))
    p = m["model"].predict_proba(X)[:,1]

    base = df[args.target].astype(int).to_numpy()
    print(f"Rows evaluated: {len(df)}  Positives (coincident {args.target}): {base.sum()}")

    g = df.groupby(["lat","lon"], sort=False)[args.target]
    for h in args.leads:
        y = g.transform(lambda s: future_max_window(s, h)).to_numpy()
        auc   = roc_auc_score(y, p)
        prauc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        pos   = int(y.sum())
        print(f"Lead +{h:>2}h  →  AUC={auc:.3f}  PRAUC={prauc:.3f}  Brier={brier:.3f}  Pos={pos}/{len(y)}")

if __name__ == "__main__":
    main()