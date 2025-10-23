# corr_quickread.py
import pandas as pd

df = pd.read_csv("results/nowcast_scores.csv")
df["absS"]   = df["S"].abs()
df["cohInv"] = 1.0/(df["relax"]+1e-9)

print("corr(risk, |S|)     =", df["risk"].corr(df["absS"]))
print("corr(risk, 1/relax) =", df["risk"].corr(df["cohInv"]))
print("corr(risk, agree)   =", df["risk"].corr(df["agree"]))
print("corr(risk, dir_var) =", df["risk"].corr(df["dir_var"]))
print("corr(risk, shear_var) =", df["risk"].corr(df["shear_var"]))