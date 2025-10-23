import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="grid_labelled_FMA_phasecols.csv.gz")
    ap.add_argument("--out", required=True, help="alerts CSV.gz")
    ap.add_argument("--use-msl-grad", action="store_true", help="extend rule with msl_grad<=0")
    ap.add_argument("--cols", default="dS_dt,drelax_dt,msl_grad", help="override column names if needed")
    args = ap.parse_args()

    cols = args.cols.split(",")
    need = ["time","lat","lon"] + cols[:2]
    df = pd.read_csv(args.labelled, usecols=lambda c: c in set(need) or c=="pregen" or c=="agree"
                    , parse_dates=["time"])
    # safety: if msl_grad missing but requested, add NaNs → False
    if args.use_msl_grad and "msl_grad" not in df.columns:
        df["msl_grad"] = np.nan

    dS = cols[0]
    dR = cols[1]
    rule = (df[dS] <= 0) | (df[dR] <= 0)
    if args.use_msl_grad:
        rule = rule | (df["msl_grad"] <= 0)

    out = df[["time","lat","lon"]].copy()
    out["alert"] = rule.astype(int)
    # keep a simple risk proxy for downstream throttling (not used by rules)
    out["risk"] = rule.astype(float)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} | alerts: {int(out['alert'].sum())}/{len(out)}")

if __name__ == "__main__":
    main()