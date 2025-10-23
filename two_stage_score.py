# two_stage_score.py
import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="CSV(.gz) with features superset for both models")
    ap.add_argument("--model-context", required=True, help="joblib pkl (e.g., near_storm)")
    ap.add_argument("--model-pregen",  required=True, help="joblib pkl (calibrated pregen)")
    ap.add_argument("--context-quantile", type=float, default=0.90, help="Keep top q per hour from context risk")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])

    # Load and score context
    mc = joblib.load(args.model_context)
    Xc = mc["scaler"].transform(df[mc["features"]].to_numpy(float))
    pc = mc["model"].predict_proba(Xc)[:,1]
    df["risk_ctx"] = pc

    # Keep top quantile per hour
    def keep_top(dfh):
        thr = dfh["risk_ctx"].quantile(args.context_quantile)
        return (dfh["risk_ctx"] >= thr).astype(int)
    df["ctx_keep"] = df.groupby("time", sort=False).apply(keep_top).reset_index(level=0, drop=True)

    # Score pregen only on kept rows; others get 0 risk_pregen
    mp = joblib.load(args.model_pregen)
    df["risk_pregen"] = 0.0
    kept = df["ctx_keep"] == 1
    if kept.any():
        Xp = mp["scaler"].transform(df.loc[kept, mp["features"]].to_numpy(float))
        pp = mp["model"].predict_proba(Xp)[:,1]
        df.loc[kept, "risk_pregen"] = pp

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} | kept (context): {int(kept.sum())}/{len(df)}")

if __name__ == "__main__":
    main()