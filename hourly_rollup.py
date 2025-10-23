# hourly_rollup.py
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True, help="CSV(.gz) with columns time,risk and any of alert_* flags")
    ap.add_argument("--flag-col", default="alert_throttled", help="Which final flag to count")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.alerts, parse_dates=["time"])
    flag = args.flag_col if args.flag_col in df.columns else None

    aggs = {"risk":["min","median","max","mean"],}
    if flag: aggs[flag] = ["sum"]

    out = (df
           .groupby("time", as_index=False)
           .agg(aggs))
    out.columns = ["time"] + ["_".join(c for c in cols if c) for cols in out.columns.to_flat_index()[1:]]

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} | rows: {len(out)}")

if __name__ == "__main__":
    main()