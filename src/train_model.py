import argparse, pandas as pd, numpy as np, joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURES = ['S','zeta_mean','div_mean','relax','agree']

def train(features_path: str, out_path: str, target: str = 'label_genesis24'):
    df = pd.read_parquet(features_path).dropna(subset=[target])
    X = df[FEATURES].astype(float).values
    y = df[target].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]
    print('AUC:', roc_auc_score(yte, p))
    print('Brier:', brier_score_loss(yte, p))
    joblib.dump({'model': clf, 'features': FEATURES, 'target': target}, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target', default='label_genesis24')
    args = ap.parse_args()
    train(args.features, args.out, args.target)

if __name__ == '__main__':
    main()
