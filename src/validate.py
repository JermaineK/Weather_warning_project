import argparse, pandas as pd, numpy as np, joblib
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from pathlib import Path

def evaluate(features_path: str, model_path: str, out_dir: str):
    d = joblib.load(model_path)
    FEATURES = d['features']; target = d['target']
    df = pd.read_parquet(features_path).dropna(subset=[target])
    X = df[FEATURES].astype(float).values
    y = df[target].astype(int).values
    p = d['model'].predict_proba(X)[:,1]
    roc = roc_auc_score(y, p)
    br = brier_score_loss(y, p)
    pr, re, _ = precision_recall_curve(y, p)
    prauc = auc(re, pr)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'metric':['AUC','Brier','PRAUC'], 'value':[roc, br, prauc]}).to_csv(Path(out_dir)/'scores.csv', index=False)
    print('AUC', roc, 'Brier', br, 'PRAUC', prauc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    evaluate(args.features, args.model, args.out)

if __name__ == '__main__':
    main()
