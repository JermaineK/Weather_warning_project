# pick_best_from_sweep.py
import pandas as pd
from pathlib import Path
import numpy as np

CSV = Path("results/sweep_summary_quick.csv")  # change if needed

def to_num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    df = pd.read_csv(CSV)
    # make sure key columns are numeric
    num_cols = ["lead","thr","persist","neighbors","quantile",
                "Precision","Recall","Coverage","AUC","PRAUC","Brier","F1"]
    existing = [c for c in num_cols if c in df.columns]
    df = to_num(df, existing)

    # recompute F1 from P & R to avoid any CSV weirdness
    if {"Precision","Recall"}.issubset(df.columns):
        pr = df["Precision"].clip(lower=0, upper=1).fillna(0.0)
        rc = df["Recall"].clip(lower=0, upper=1).fillna(0.0)
        df["F1_fix"] = (2*pr*rc) / (pr + rc + 1e-9)
    else:
        df["F1_fix"] = df.get("F1", pd.Series(np.nan, index=df.index))

    # baseline constraints to avoid degenerate “1% recall but great F1” rows
    COVERAGE_MAX = 0.25
    RECALL_MIN   = 0.20
    PREC_MIN     = 0.10

    ok = df.copy()
    ok = ok[(ok["Coverage"] <= COVERAGE_MAX) &
            (ok["Recall"]   >= RECALL_MIN) &
            (ok["Precision"]>= PREC_MIN)]
    if ok.empty:
        print("No rows met constraints; relaxing to Coverage<=0.35 and Recall>=0.10 …")
        ok = df[(df["Coverage"] <= 0.35) & (df["Recall"] >= 0.10)].copy()
        if ok.empty:
            print("Still empty. Showing top by F1_fix within any coverage.")
            ok = df.copy()

    keep_cols = ["lead","thr","persist","neighbors","quantile",
                 "F1_fix","Precision","Recall","Coverage","AUC","PRAUC","Brier"]
    ok = ok.sort_values(["lead","F1_fix","Recall","Precision"],
                        ascending=[True, False, False, False])

    # best per lead
    best = ok.groupby("lead", as_index=False).head(1)[keep_cols]
    print("\n== Best per lead (constraints applied) ==\n")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # also show top-5 per lead for context
    print("\n== Top-5 per lead ==\n")
    top5 = ok.groupby("lead", as_index=False).head(5)[keep_cols]
    print(top5.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # print ready-to-run commands
    for _,r in best.iterrows():
        cmd = (f"python apply_thresholds.py --labelled data\\grid_labelled_FMA_plus.csv.gz "
               f"--model models\\grid_pregen_from_available_cal.pkl --lead-hours {int(r.lead)} "
               f"--thr {r.thr:.3f} --out results\\alerts_best_lead{int(r.lead)}.csv.gz")
        print("\nSuggested command:\n", cmd)

if __name__ == "__main__":
    main()