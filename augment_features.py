#!/usr/bin/env python
import argparse, pandas as pd
from pathlib import Path
from features_patch import add_all_feature_enhancements

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="Input CSV(.gz) from rebuild_grid_from_nc.py")
    ap.add_argument("--out", dest="outp", required=True, help="Output CSV(.gz) with augmented features")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["time"])
    df2 = add_all_feature_enhancements(df)

    Path(args.outp).parent.mkdir(parents=True, exist_ok=True)
    comp = "gzip" if str(args.outp).endswith(".gz") else None
    df2.to_csv(args.outp, index=False, compression=comp)
    print(f"[augment] wrote {args.outp} rows: {len(df2):,}")

if __name__ == "__main__":
    main()