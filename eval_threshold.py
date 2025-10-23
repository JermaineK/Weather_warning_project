# lead_time_eval.py
import pandas as pd, numpy as np, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

LAB = "data/features_era5_au_labelled.csv"
MODEL = "models/logit_labelled.pkl"

df = pd.read_csv(LAB)
df["time"] = pd.to_datetime(df["time"], utc=True)
bundle = joblib.load(MODEL)
sc, mdl, feats = bundle["scaler"], bundle["model"], bundle["features"]

X = df[feats].values
p = mdl.predict_proba(sc.transform(X))[:,1]

def eval_shift(hours):
    # label a sample positive if ANY positive occurs within the next `hours`
    y0 = df["storm"].values.astype(int)
    y_shift = np.zeros_like(y0)
    # group by time; simpler with rolling max over time per location (you have single location now)
    # For multi-cell later, do this per (lat_c, lon_c)
    s = pd.Series(y0, index=df["time"])
    y_shift = s.rolling(f"{hours}H", min_periods=1).max().shift(-1).reindex(df["time"]).fillna(0).to_numpy().astype(int)

    auc = roc_auc_score(y_shift, p)
    pr  = average_precision_score(y_shift, p)
    br  = brier_score_loss(y_shift, p)
    print(f"Lead {hours}h -> AUC={auc:.3f} PRAUC={pr:.3f} Brier={br:.3f}")

for h in [3, 6, 12]:
    eval_shift(h)