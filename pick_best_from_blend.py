#!/usr/bin/env python
import pandas as pd

df = pd.read_csv("results/blend_sweep_quick.csv")
best = (
    df.groupby(["alpha","lead_h"], as_index=False)
      .apply(lambda g: g.loc[g["F1"].idxmax()])
      .reset_index(drop=True)
)
print("\n== Best per alpha/lead ==")
print(best[["alpha","lead_h","thr","persist_h","quantile","F1","Precision","Recall","AUC","PRAUC","Brier","Coverage"]])

# Also best overall
top = df.loc[df["F1"].idxmax()]
print("\n== Best overall ==")
print(top.to_string(index=False))