import argparse, numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_recall_curve

def future_max_window(s: pd.Series, hours: int) -> pd.Series:
    # “Any storm in the next <hours>?” label, computed per grid cell.
    y = s.astype(int)
    f = y.iloc[::-1].shift(1).rolling(hours, min_periods=1).max()
    return f.iloc[::-1].fillna(0).astype(int)

def best_f1(y_true: np.ndarray, p: np.ndarray):
    pr, rc, th = precision_recall_curve(y_true, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i = int(np.nanargmax(f1))
    # sklearn returns len(th) = len(pr) - 1
    thr = float(th[max(i - 1, 0)]) if len(th) else 0.5
    return dict(best_f1=float(f1[i]), threshold=thr, precision=float(pr[i]), recall=float(rc[i]))

def main():
    ap = argparse.ArgumentParser(description="Find best-F1 threshold(s) for lead windows.")
    ap.add_argument("--labelled", required=True, help="Path to grid_labelled_*.csv(.gz)")
    ap.add_argument("--model", required=True, help="Path to joblib-calibrated model (dict with keys: model, scaler, features)")
    ap.add_argument("--leads", nargs="+", type=int, default=[24], help="Lead hours to evaluate (e.g., 6 12 24 36 48)")
    ap.add_argument("--target", default="pregen", choices=["pregen","storm","near_storm"], help="Which label column to use")
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap for quick runs")
    ap.add_argument("--save-csv", default=None, help="Optional path to write results CSV")
    args = ap.parse_args()

    m = joblib.load(args.model)
    USE = list(m["features"])
    cols_needed = list(set(USE + ["time", "lat", "lon", args.target]))
    df = pd.read_csv(args.labelled, parse_dates=["time"], usecols=lambda c: c in cols_needed)

    if args.limit:
        df = df.iloc[:args.limit].copy()

    # score
    X = m["scaler"].transform(df[USE].to_numpy(float))
    p = m["model"].predict_proba(X)[:, 1]
    print(f"Rows loaded: {len(df):,}  Positives (coincident {args.target}): {int(df[args.target].sum()):,}")

    out_rows = []
    g = df.groupby(["lat", "lon"], sort=False)[args.target]
    for h in args.leads:
        y = g.transform(lambda s: future_max_window(s, h)).to_numpy()
        res = best_f1(y, p)
        out_rows.append({"lead_h": h, **res})
        print(f"Lead +{h:>2}h → Best F1: {res['best_f1']:.3f}  @ thr {res['threshold']:.3f}  (P={res['precision']:.3f}, R={res['recall']:.3f})")

    if args.save_csv:
        pd.DataFrame(out_rows).to_csv(args.save_csv, index=False)
        print(f"Saved thresholds → {args.save_csv}")

if __name__ == "__main__":
    main()
