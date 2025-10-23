import argparse, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def future_or(df, col, hours):
    def fwd_or(s, h):
        y = s.astype(int)
        f = y.iloc[::-1].shift(1).rolling(h, min_periods=1).max()
        return f.iloc[::-1].fillna(0).astype(int)
    return df.groupby(["lat","lon"], sort=False)[col].transform(lambda s: fwd_or(s, hours)).to_numpy()

def metrics(y, p):
    return dict(
        AUC=float(roc_auc_score(y, p)),
        PRAUC=float(average_precision_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
        Pos=int(y.sum()), N=int(len(y))
    )

def predict_probs(df, bundle):
    feats = bundle["features"]
    Xdf = df[feats].copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.to_numpy(float)
    Xs = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(Xs)[:,1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build",    required=True)
    ap.add_argument("--relax",    required=True)
    ap.add_argument("--target",   required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, default=[24,48])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"]).sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)

    p_build = predict_probs(df, mb)
    p_relax = predict_probs(df, mr)

    # train meta-model on coincident target
    y = df[args.target].astype(int).to_numpy()
    Xblend = np.c_[p_build, p_relax]

    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))])
    pipe.fit(Xblend, y)
    p_meta = pipe.predict_proba(Xblend)[:,1]

    base = metrics(y, p_meta)
    print(f"[{args.target}] BLEND COINCIDENT  AUC={base['AUC']:.3f} PRAUC={base['PRAUC']:.3f} Brier={base['Brier']:.3f} Pos={base['Pos']}/{base['N']}")

    for h in args.lead_hours:
        y_lead = future_or(df, args.target, h)
        m = metrics(y_lead, p_meta)
        print(f"[{args.target}] BLEND LEAD +{h:>2}h  AUC={m['AUC']:.3f} PRAUC={m['PRAUC']:.3f} Brier={m['Brier']:.3f} Pos={m['Pos']}/{m['N']}")

    bundle = dict(model=pipe, inputs=["p_build","p_relax"])
    joblib.dump(bundle, args.out)
    print(f"Saved blend → {args.out}")

if __name__ == "__main__":
    main()