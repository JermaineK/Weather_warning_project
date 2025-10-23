import argparse, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def predict_probs(df, bundle):
    feats = bundle["features"]
    Xdf = df[feats].copy()
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))
    X = Xdf.to_numpy(float)
    Xs = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(Xs)[:,1]

def future_or(df, col, hours):
    def fwd_or(s, h):
        y = s.astype(int)
        f = y.iloc[::-1].shift(1).rolling(h, min_periods=1).max()
        return f.iloc[::-1].fillna(0).astype(int)
    return df.groupby(["lat","lon"], sort=False)[col].transform(lambda s: fwd_or(s, hours)).to_numpy()

def metrics(y, a):
    # a is 0/1 alerts; also compute prob-like score using a.astype(float)
    p = a.astype(float)
    return dict(
        Precision=float(( (a==1) & (y==1) ).sum() / max(1, a.sum())),
        Recall=float(( (a==1) & (y==1) ).sum() / max(1, y.sum())),
        F1=float( (2* ( (a==1)&(y==1) ).sum()) / max(1, (a.sum()+y.sum())) ),
        AUC=float(roc_auc_score(y, p)),
        PRAUC=float(average_precision_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
        Alerts=int(a.sum()),
        Pos=int(y.sum()),
        N=int(len(y))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build",    required=True)
    ap.add_argument("--relax",    required=True)
    ap.add_argument("--target",   required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--build-window", type=int, default=24, help="hours for recent max of p_build")
    ap.add_argument("--thr-build", nargs="+", type=float, default=[0.05,0.07,0.10])
    ap.add_argument("--thr-relax", nargs="+", type=float, default=[0.05,0.07,0.10])
    ap.add_argument("--lead-hours", nargs="+", type=int, default=[24,48])
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"]).sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    mb = joblib.load(args.build)
    mr = joblib.load(args.relax)

    p_build = predict_probs(df, mb)
    p_relax = predict_probs(df, mr)

    # recent max of build per point on past window
    def past_max_build(s, h):
        # rolling on forward time → use classic rolling then shift(1)
        return s.rolling(h, min_periods=1).max().shift(1).fillna(0)
    build_recent = df.groupby(["lat","lon"], sort=False).apply(
        lambda g: past_max_build(pd.Series(p_build[g.index], index=g.index), args.build_window)
    ).reset_index(level=[0,1], drop=True).reindex(df.index).to_numpy()

    y0 = df[args.target].astype(int).to_numpy()
    print(f"Rows: {len(df)}  Pos(coincident {args.target}): {y0.sum()}  BuildWin={args.build_window}h")

    # coincident first
    for tb in args.thr_build:
        for tr in args.thr_relax:
            alerts = ((build_recent >= tb) & (p_relax >= tr)).astype(int)
            m0 = metrics(y0, alerts)
            print(f"[COINC] tb={tb:.3f} tr={tr:.3f} | F1={m0['F1']:.3f} P={m0['Precision']:.3f} R={m0['Recall']:.3f} "
                  f"AUC={m0['AUC']:.3f} PRAUC={m0['PRAUC']:.3f} Brier={m0['Brier']:.3f} "
                  f"Alerts={m0['Alerts']}/{m0['N']}")

    # lead windows
    for h in args.lead_hours:
        y = future_or(df, args.target, h)
        for tb in args.thr_build:
            for tr in args.thr_relax:
                alerts = ((build_recent >= tb) & (p_relax >= tr)).astype(int)
                m = metrics(y, alerts)
                print(f"[LEAD +{h:>2}h] tb={tb:.3f} tr={tr:.3f} | F1={m['F1']:.3f} P={m['Precision']:.3f} R={m['Recall']:.3f} "
                      f"AUC={m['AUC']:.3f} PRAUC={m['PRAUC']:.3f} Brier={m['Brier']:.3f} "
                      f"Alerts={m['Alerts']}/{m['N']}")
if __name__ == "__main__":
    main()