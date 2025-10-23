# score_overlap.py
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def lead_mask(df, target_col, hours):
    """For each (lat,lon) series, mark if target occurs in the next `hours` steps (future-OR)."""
    def future_max_window(s, h):
        y = s.astype(int)
        f = y.iloc[::-1].shift(1).rolling(h, min_periods=1).max()
        return f.iloc[::-1].fillna(0).astype(int)
    return df.groupby(["lat","lon"], sort=False)[target_col].transform(lambda s: future_max_window(s, hours)).to_numpy()

def score(y, p):
    return dict(
        AUC=float(roc_auc_score(y, p)),
        PRAUC=float(average_precision_score(y, p)),
        Brier=float(brier_score_loss(y, p)),
        Pos=int(np.sum(y)),
        N=int(len(y)),
    )

def predict_probs(df, model_bundle, tag):
    """
    Build feature matrix for one model, median-impute NaNs, scale, and predict proba.
    Returns probability vector (aligned to df.index).
    """
    feats = model_bundle["features"]
    scaler = model_bundle["scaler"]
    model  = model_bundle["model"]

    # Select ONLY the features the model expects (and keep order!)
    Xdf = df[feats].copy()

    # Median-impute per-column to kill any NaNs safely
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med)

    # Convert to float, scale, predict
    X = Xdf.to_numpy(float)
    Xs = scaler.transform(X)
    p  = model.predict_proba(Xs)[:, 1]

    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--build",    required=True)
    ap.add_argument("--relax",    required=True)
    ap.add_argument("--target",   required=True, choices=["storm","near_storm","pregen"])
    ap.add_argument("--lead-hours", nargs="+", type=int, required=True)
    ap.add_argument("--combine", choices=["product","min","max","mean"], default="product")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"]).sort_values(["time","lat","lon"], kind="mergesort").reset_index(drop=True)

    # Load phase models
    m_build = joblib.load(args.build)
    m_relax = joblib.load(args.relax)

    # Predict probabilities with NaN-safe path
    p_build = predict_probs(df, m_build, tag="build")
    p_relax = predict_probs(df, m_relax, tag="relax")

    # Combine
    if args.combine == "product":
        p = p_build * p_relax
    elif args.combine == "min":
        p = np.minimum(p_build, p_relax)
    elif args.combine == "max":
        p = np.maximum(p_build, p_relax)
    else:  # mean
        p = 0.5 * (p_build + p_relax)

    # Evaluate coincident and lead windows
    base = df[args.target].astype(int).to_numpy()
    base_metrics = score(base, p)
    print(f"[{args.target}] COINCIDENT  AUC={base_metrics['AUC']:.3f}  PRAUC={base_metrics['PRAUC']:.3f}  "
          f"Brier={base_metrics['Brier']:.3f}  Pos={base_metrics['Pos']}/{base_metrics['N']}")

    for h in args.lead_hours:
        y = lead_mask(df, args.target, h)
        m = score(y, p)
        print(f"[{args.target}] LEAD +{h:>2}h  AUC={m['AUC']:.3f}  PRAUC={m['PRAUC']:.3f}  "
              f"Brier={m['Brier']:.3f}  Pos={m['Pos']}/{m['N']}")

if __name__ == "__main__":
    main()