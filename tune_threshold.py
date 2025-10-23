# tune_threshold.py
import argparse, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="inp",  default="results/nowcast_scores.csv")
ap.add_argument("--out", dest="out",  default="results/nowcast_scores_tuned.csv")
ap.add_argument("--top", type=float,  default=0.10, help="Top fraction to flag (e.g., 0.10 = top 10%)")
args = ap.parse_args()

df = pd.read_csv(args.inp)
q = df["risk"].quantile(1.0 - args.top)
df["flag"] = (df["risk"] >= q).astype(int)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(f"Threshold={q:.4f}  alerts={int(df.flag.sum())}/{len(df)}  -> wrote {args.out}")