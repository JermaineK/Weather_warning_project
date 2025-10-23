# calibrate.py
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

LAB = "data/features_era5_au_labelled.csv"
BASE = "models/logit_labelled.pkl"
OUT  = "models/logit_labelled_calibrated.pkl"

df = pd.read_csv(LAB)
bundle = joblib.load(BASE)
sc, mdl, feats = bundle["scaler"], bundle["model"], bundle["features"]

X = df[feats].values
y = df["storm"].astype(int).values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

cal = CalibratedClassifierCV(mdl, method="isotonic", cv=3)
cal.fit(sc.transform(Xtr), ytr)
p = cal.predict_proba(sc.transform(Xte))[:,1]
print("Brier (before):", brier_score_loss(yte, mdl.predict_proba(sc.transform(Xte))[:,1]))
print("Brier (after ):", brier_score_loss(yte, p))

joblib.dump({"scaler": sc, "model": cal, "features": feats}, OUT)
print("Saved", OUT)