#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_per_lead_logit.py

Two modes:

1) DIRECT PER-HOUR LABELS (recommended if you already built them)
   You have columns like pregen_h1, pregen_h2, ..., pregen_h240.
   Use:
     --label-prefix pregen_h --hours 1..240

2) SHIFTED (time-aware by default)
   You have a single base label column (e.g., 'storm'), and you want
   to train a per-lead model using a future window:
     yL(t,cell) = 1 if any base_label==1 occurs in (t, t+L] for that cell.
   Use:
     --label-col storm --leads 1..240
   Optional: --step-shift to revert to legacy fixed-k shift (not recommended).

Extras:
  • Leakage-safe evaluation (--eval) with day-grouped split; writes metrics CSV.
  • Stable cell grouping via rounded lat/lon (--round-geo).
  • Reproducible seed, class-imbalance warning, odds-ratio dump.

Memory safety helpers:
  • --sample-frac: uniform subsampling of the full table.
  • --neg-frac (shifted mode): keep all positives, sample this fraction of negatives.
  • Hard guard: refuses to train on >10M rows unless you downscale.

Output bundle (joblib pickle):
  {
    "per_lead_models": { <lead_h>: sklearn Pipeline, ... },
    "features": [ ... ],
    "meta": { ... }
  }
Also writes .meta.json next to the pickle.
"""

import argparse, json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pa = None
    pq = None


# ---------- Diagnostics helpers ----------

def diag_header(df: pd.DataFrame, mode_str: str):
    tt = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("H")
    unique_hours = int(tt.nunique())
    cells = int(
        pd.DataFrame({
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lon": pd.to_numeric(df["lon"], errors="coerce")
        })
        .dropna()
        .drop_duplicates()
        .shape[0]
    )
    print(f"[diag] {mode_str}  rows={len(df):,}  cells={cells:,}  unique_hours={unique_hours:,}", flush=True)


def diag_per_lead(lead_h: int, pos: int, n_rows: int,
                  pos_prev: int | None = None, sample_same: float | None = None):
    frac = pos / max(1, n_rows)
    msg = f"[diag] lead={lead_h:>3d}h  pos={pos:,}  frac={frac:.4f}"
    if pos_prev is not None:
        msg += f"  Δpos_vs_prev={pos - pos_prev:+,}"
    if sample_same is not None:
        msg += f"  same={sample_same:.3%}"
    print(msg, flush=True)


# ---------- I/O ----------

def read_any(path):
    p = str(path)
    if p.lower().endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


# Agent: stream parquet to apply sampling before materializing to avoid OOM.
def _stream_parquet_with_sampling(path: str, columns: list[str] | None, batch_rows: int,
                                  seed: int, sample_frac: float,
                                  neg_frac: float, label_col: str | None,
                                  pandas_batch_rows: int = 20_000):
    """
    Stream parquet in batches and apply sampling to reduce peak memory use.

    Returns: (df, uniform_applied, neg_applied)
    If streaming is unavailable (missing pyarrow.dataset), returns (None, False, False).
    """
    if pq is None or pa is None:
        return None, False, False
    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception:
        return None, False, False

    rng = np.random.default_rng(seed)
    uniform_applied = bool(sample_frac and 0 < sample_frac < 1)
    neg_applied = bool(neg_frac and 0 < neg_frac < 1 and label_col)

    dataset = ds.dataset(path, format="parquet")
    scan_cols = columns if columns else None

    # Cap batch size to keep to_pandas manageable.
    BATCH_CAP = max(5_000, min(int(pandas_batch_rows), 20_000))
    batch_size = batch_rows if batch_rows and batch_rows > 0 else BATCH_CAP
    if batch_size > BATCH_CAP:
        batch_size = BATCH_CAP
        print(f"[stream-load] capping parquet batch size to {batch_size} rows to reduce memory", flush=True)

    scanner = None
    # pyarrow APIs vary: FileSystemDataset may not have .scan in older versions.
    if hasattr(dataset, "scan"):
        try:
            scanner = dataset.scan(columns=scan_cols)
        except Exception:
            scanner = None
    if scanner is None:
        try:
            scanner = ds.Scanner.from_dataset(dataset, columns=scan_cols, batch_size=batch_size)
        except Exception:
            return None, False, False

    dfs = []
    total_rows = 0
    kept_rows = 0

    try:
        batches = scanner.to_batches(batch_size=batch_size)
    except TypeError:
        # Some pyarrow versions don't support batch_size kwarg on to_batches.
        batches = scanner.to_batches()

    for batch in batches:
        # slice very large batches into pandas-sized chunks to avoid huge allocations
        n_batch = len(batch)
        slice_rows = min(batch_size, BATCH_CAP)
        for offset in range(0, n_batch, slice_rows):
            sub = batch.slice(offset, min(slice_rows, n_batch - offset))
            try:
                pdf = sub.to_pandas(ignore_metadata=True, split_blocks=True, self_destruct=True)
            except TypeError:
                pdf = sub.to_pandas(ignore_metadata=True)
            total_rows += len(pdf)

            if uniform_applied:
                mask = rng.random(len(pdf)) < sample_frac
                pdf = pdf.loc[mask]

            if neg_applied and label_col in pdf.columns:
                ybase = pd.to_numeric(pdf[label_col], errors="coerce").fillna(0)
                pos_mask = ybase > 0
                neg_mask = ~pos_mask
                if neg_mask.any():
                    neg_idx = np.where(neg_mask.to_numpy())[0]
                    neg_keep = rng.random(len(neg_idx)) < neg_frac
                    keep_mask = pos_mask.to_numpy()
                    keep_mask[neg_idx] = neg_keep
                    pdf = pdf.loc[keep_mask]

            kept_rows += len(pdf)
            if not pdf.empty:
                dfs.append(pdf)

    if not dfs:
        return pd.DataFrame(columns=columns or dataset.schema.names), uniform_applied, neg_applied

    df = pd.concat(dfs, ignore_index=True)
    print(
        f"[stream-load] read {total_rows:,} rows in batches "
        f"({batch_size}); kept {kept_rows:,} after sampling",
        flush=True,
    )
    return df, uniform_applied, neg_applied


def _parquet_numeric_columns(path: str, required: set[str]) -> list[str]:
    """
    Inspect parquet schema and pick numeric columns plus required ones to
    reduce memory footprint when loading huge files.

    Uses pyarrow's Arrow schema (pf.schema_arrow) so it is robust across
    pyarrow versions where pf.schema fields may be ColumnSchema without .type.
    """
    if pq is None or pa is None:
        # No pyarrow: caller will just read all columns.
        return []

    try:
        pf = pq.ParquetFile(path)
    except Exception:
        # If anything goes wrong, fall back to "no pruning".
        return []

    cols: list[str] = []

    try:
        # Arrow Schema: fields have .name and .type
        schema = pf.schema_arrow
    except Exception:
        # Older pyarrow: give up on pruning
        return []

    for field in schema:
        name = field.name
        if name in required:
            cols.append(name)
            continue

        t = field.type
        # Treat integer / float (including all bit widths) as numeric
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            cols.append(name)

    # Always include required columns even if type detection failed
    return sorted(set(cols) | set(required))


# ---------- parsing ----------

def parse_range_or_list(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return []
    if ".." in spec:
        a, b = spec.split("..", 1)
        a, b = int(a), int(b)
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    return sorted({int(x) for x in spec.split(",") if str(x).strip()})


def nearest_step_hours(lead_h: int, dt_h: int):
    if dt_h <= 0:
        raise ValueError("--dt-hours must be positive")
    steps = int(np.round(lead_h / dt_h))
    steps = max(0, steps)
    return steps * dt_h, steps


# ---------- feature picking ----------

BASE_RESERVED = {
    "time", "lat", "lon",
    "name", "basin", "storm_id", "sid", "pmin", "vmax",
    "lead", "lead_h", "lead_hours", "lead_hour",
    "storm", "near_storm", "pregen", "alert", "alert_final", "event", "target", "label", "y",
    "number", "expver",
    "_grp", "_grp_round",
    "row_id", "id", "ID",
    "storm_hit", "storm_window", "storm_point"  # common downstream target/meta cols
}

def pick_features(df: pd.DataFrame, exclude_cols: set[str]):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num if c not in exclude_cols]
    if not feats:
        raise ValueError("No numeric feature columns found after excluding metadata/labels.")
    return feats


# ---------- time-aware future window labeling ----------

def future_max_label_by_point_timeaware(df: pd.DataFrame, target: str, hours: int,
                                        grp_col: str = "_grp_round") -> np.ndarray:
    """
    For each cell group (grp_col), for each row i at time t_i:
      yL(i) = 1 if any future row for that group within (t_i, t_i+hours] has target==1.
    Robust to missing hours and irregular cadence.
    """
    out = np.zeros(len(df), dtype=np.int8)
    hour_ns = np.int64(hours) * np.int64(3_600_000_000_000)  # 1h in ns

    for _, g in df.groupby(grp_col, sort=False, group_keys=False):
        idx = g.index.to_numpy()
        y   = pd.to_numeric(g[target], errors="coerce").fillna(0).astype(int).to_numpy()

        t_series = pd.to_datetime(g["time"], errors="coerce", utc=True)
        t_ns = t_series.view("int64").to_numpy()

        order = np.argsort(t_ns, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))

        t_sorted = t_ns[order]
        y_sorted = y[order]

        ps = np.zeros(len(y_sorted) + 1, dtype=np.int64)
        ps[1:] = np.cumsum(y_sorted)

        t_end_sorted = t_sorted + hour_ns
        end_pos = np.searchsorted(t_sorted, t_end_sorted, side="right")

        # any label strictly AFTER current time and up to the end window
        any_future_sorted = (ps[end_pos] - ps[np.arange(len(y_sorted)) + 1]) > 0
        any_future = any_future_sorted[inv_order].astype(np.int8)
        out[idx] = any_future
    return out


# ---------- metrics ----------

def _safe_auc(ytrue, p):
    try:
        return roc_auc_score(ytrue, p)
    except Exception:
        return float("nan")


def _safe_ap(ytrue, p):
    try:
        return average_precision_score(ytrue, p)
    except Exception:
        return float("nan")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Per-lead logistic regression trainer with time-aware labels & diagnostics.")
    ap.add_argument("--labelled", required=True, help="Labelled grid CSV/Parquet")
    ap.add_argument("--out", required=True, help="Output pickle path")

    # Mode A: DIRECT per-hour labels
    ap.add_argument("--label-prefix", default=None,
                    help="Prefix for per-hour columns, e.g., 'pregen_h'. If set, use --hours.")
    ap.add_argument("--hours", default=None,
                    help='Hour set for direct labels, e.g., "1..240" or "6,12,18".')

    # Mode B: SHIFTED (time-aware by default)
    ap.add_argument("--label-col", default=None,
                    help="Base label (e.g., 'storm') used to build future windows.")
    ap.add_argument("--leads", default=None,
                    help='Lead hours for shifted mode, e.g., "1..240" or "6,12,18".')
    ap.add_argument("--dt-hours", type=int, default=1,
                    help="Only used if --step-shift is set. Default 1.")
    ap.add_argument("--step-shift", action="store_true",
                    help="Legacy fixed-step shift inside each cell group. Not recommended.")

    # Common knobs
    ap.add_argument("--min-positives", type=int, default=20, help="Min positives required to train a lead model")
    ap.add_argument("--class-weight", default="balanced", help='sklearn class_weight (default "balanced")')
    ap.add_argument("--C", type=float, default=1.0, help="LR inverse regularization strength")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--round-geo", type=int, default=4,
                    help="Decimals to round lat/lon for stable grouping (default 4)")
    ap.add_argument("--eval", action="store_true",
                    help="Do leakage-safe eval (day-grouped split) and write metrics CSV")
    ap.add_argument("--columns-from-schema", action="store_true",
                    help="For parquet: load only numeric columns + required (time,lat,lon,label) to reduce memory.")
    ap.add_argument("--load-numeric-only", action="store_true",
                    help="Alias for --columns-from-schema (kept for clarity).")
    ap.add_argument("--sample-frac", type=float, default=0.0,
                    help="Optional uniform fraction (0<frac<=1) to sample rows for training/eval.")
    ap.add_argument("--neg-frac", type=float, default=0.0,
                    help="In shifted mode: keep all positives and sample this fraction (0<neg-frac<=1) of negatives.")
    ap.add_argument("--chunk-rows", type=int, default=0, help="Alias for orchestrator-injected chunk sizes (ignored).")
    ap.add_argument("--chunksize", type=int, default=0, help="Alias for orchestrator-injected chunk sizes (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=0, help="Alias for orchestrator-injected chunk sizes (ignored).")
    ap.add_argument("--pandas-batch-rows", type=int, default=50_000,
                    help="Max rows per pandas materialization when streaming parquet to limit peak memory.")
    ap.add_argument("--fit-max-rows", type=int, default=0,
                    help="Uniformly subsample to this many rows before fitting to avoid OOM (0=disabled).")

    args = ap.parse_args()

    direct_mode = args.label_prefix is not None and args.hours is not None
    shifted_mode = args.label_col is not None and args.leads is not None

    # Optional column pruning for parquet
    cols = None
    parquet_in = str(args.labelled).lower().endswith((".parquet", ".pq", ".pqt"))
    numeric_only = args.columns_from_schema or args.load_numeric_only or parquet_in
    if numeric_only and parquet_in and pq is not None:
        required = {"time", "lat", "lon"}
        if args.label_col:
            required.add(args.label_col)
        cols = _parquet_numeric_columns(str(args.labelled), required)
        print(f"[load] parquet column prune enabled; reading {len(cols)} columns", flush=True)

    sampled_uniform = False
    sampled_neg = False
    df = None

    batch_rows = max(args.parquet_rows, args.chunk_rows, args.chunksize)
    if parquet_in and batch_rows and pq is not None:
        df, sampled_uniform, sampled_neg = _stream_parquet_with_sampling(
            str(args.labelled),
            cols,
            batch_rows,
            seed=args.seed,
            sample_frac=args.sample_frac,
            neg_frac=(args.neg_frac if shifted_mode else 0.0),
            label_col=(args.label_col if shifted_mode else None),
            pandas_batch_rows=args.pandas_batch_rows,
        )

    if df is None:
        df = pd.read_parquet(args.labelled, columns=cols) if cols else read_any(args.labelled)

    # Optional uniform sampling to bound memory
    if (not sampled_uniform) and args.sample_frac and 0 < args.sample_frac < 1:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"[sample] kept {len(df):,} rows at frac={args.sample_frac}", flush=True)

    # Downcast floats to float32 to reduce footprint
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # Fill NaNs in numeric columns to keep sklearn happy without exploding memory.
    # Preserve t_to_storm_min_h semantics: missing/non-finite => large sentinel (no future storm).
    tts_col = "t_to_storm_min_h" if "t_to_storm_min_h" in df.columns else None
    if tts_col:
        tts = pd.to_numeric(df[tts_col], errors="coerce")
        nonfinite = ~np.isfinite(tts)
        if nonfinite.any():
            sentinel = np.float32(1e6)
            df.loc[nonfinite, tts_col] = sentinel
            print(f"[impute] filled {int(nonfinite.sum()):,} t_to_storm_min_h NaNs with {sentinel}", flush=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if tts_col:
        num_cols = [c for c in num_cols if c != tts_col]
    if num_cols:
        na_counts = df[num_cols].isna().sum().sum()
        if na_counts:
            df[num_cols] = df[num_cols].fillna(0.0)
            print(f"[impute] filled {int(na_counts):,} numeric NaNs with 0.0", flush=True)

    if args.fit_max_rows and args.fit_max_rows > 0 and len(df) > args.fit_max_rows:
        df = df.sample(n=int(args.fit_max_rows), random_state=args.seed).reset_index(drop=True)
        print(f"[fit-cap] trimmed to {len(df):,} rows for fitting (--fit-max-rows)", flush=True)

    # time to datetime, sort for stable operations
    required_cols = {"time", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Labelled file must contain {sorted(required_cols)} (missing {sorted(missing)}).")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["time", "lat", "lon"]).copy()
    df = df.sort_values(["lat", "lon", "time"]).reset_index(drop=True)

    # Determine mode from args
    direct_mode = args.label_prefix is not None and args.hours is not None
    shifted_mode = args.label_col is not None and args.leads is not None
    if direct_mode and shifted_mode:
        raise ValueError("Choose exactly one mode: either (--label-prefix & --hours) OR (--label-col & --leads).")
    if not direct_mode and not shifted_mode:
        raise ValueError("Specify a mode: (--label-prefix & --hours) for direct, or (--label-col & --leads) for shifted.")

    # If shifted mode: coerce base label only
    if shifted_mode:
        if args.label_col not in df.columns:
            raise ValueError(
                f"Column '{args.label_col}' missing from {args.labelled}. "
                f"Columns seen: {list(df.columns)[:20]} ..."
            )
        yraw = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0)
        df[args.label_col] = (yraw > 0).astype(int)

        if args.neg_frac and 0 < args.neg_frac < 1:
            print(
                "[warn] --neg-frac is not applied in time-aware shifted mode "
                "(it would destroy future-window positives). "
                "Use --sample-frac or --fit-max-rows instead.",
                flush=True,
            )

    # Hard memory safety gate: refuse to train on huge tables unless explicitly downscaled
    MAX_SAFE_ROWS = 10_000_000
    if len(df) > MAX_SAFE_ROWS and not (args.sample_frac and 0 < args.sample_frac < 1) \
       and not (args.neg_frac and 0 < args.neg_frac < 1):
        raise RuntimeError(
            f"Labelled dataset has {len(df):,} rows; this is likely to exhaust memory.\n"
            f"Use --sample-frac (uniform) and/or --neg-frac (shifted mode) to downscale, "
            f"or pre-filter the dataset."
        )

    # Stable group ids (after any sampling)
    df["_grp_round"] = pd.factorize(
        list(zip(df["lat"].round(args.round_geo),
                 df["lon"].round(args.round_geo)))
    )[0]
    df["_day"] = pd.to_datetime(df["time"]).dt.floor("D")

    models: Dict[int, Pipeline] = {}
    trained: List[int] = []
    lead_meta: Dict[int, dict] = {}
    metrics_rows: List[dict] = []

    # ===== Mode A: DIRECT per-hour labels =====
    if direct_mode:
        hours = parse_range_or_list(args.hours)
        if not hours:
            raise ValueError("No valid --hours provided.")

        label_cols = {h: f"{args.label_prefix}{h}" for h in hours}
        missing = [c for c in label_cols.values() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing per-hour label columns: {missing[:10]}{'...' if len(missing)>10 else ''}")

        exclude = set(BASE_RESERVED) | set(label_cols.values())
        feats = pick_features(df, exclude)
        diag_header(df, f"direct hours={hours[0]}..{hours[-1]}")

        # Precompute feature matrix once to avoid repeated allocations
        X_full = df[feats].to_numpy(dtype=np.float32, copy=False)

        last_pos = None
        last_y = None

        for H, col in label_cols.items():
            y_arr = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
            pos = int(y_arr.sum())
            neg = int(len(y_arr) - pos)

            same_frac = None
            if last_y is not None and last_y.shape == y_arr.shape:
                same_frac = float((last_y == y_arr).mean())
            diag_per_lead(H, pos, len(y_arr), pos_prev=last_pos, sample_same=same_frac)
            last_pos, last_y = pos, y_arr

            if pos < args.min_positives:
                print(f"[skip] lead={H:>3d}h (direct): positives={pos} < {args.min_positives}")
                continue
            if pos / len(y_arr) < 0.005:
                print(f"[warn] lead={H:>3d}h prevalence is very low ({pos/len(y_arr):.4f}); expect fragile metrics.")

            X = X_full
            y = y_arr.astype(int)

            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True, copy=False)),
                ("lr", LogisticRegression(
                    C=args.C, max_iter=200, class_weight=args.class_weight,
                    solver="lbfgs", penalty="l2", random_state=args.seed
                )),
            ])
            pipe.fit(X, y)

            # Optional eval: day-grouped split to reduce leakage
            if args.eval:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
                tr_idx, te_idx = next(gss.split(X, y, groups=df["_day"]))
                p_tr = pipe.predict_proba(X[tr_idx])[:, 1]
                p_te = pipe.predict_proba(X[te_idx])[:, 1]
                metrics_rows.append(dict(
                    lead_h=H, mode="direct",
                    auc_tr=_safe_auc(y[tr_idx], p_tr), auc_te=_safe_auc(y[te_idx], p_te),
                    ap_tr=_safe_ap(y[tr_idx], p_tr), ap_te=_safe_ap(y[te_idx], p_te),
                    prev=float(y.mean()), n=len(y)
                ))

            models[int(H)] = pipe
            trained.append(int(H))
            lead_meta[int(H)] = {
                "mode": "direct",
                "rounded_lead_h": int(H),
                "steps": 0,
                "dt_hours": int(args.dt_hours),
                "n": int(len(df)),
                "pos": int(pos),
                "neg": int(neg),
                "label_col": col,
            }
            print(f"[fit] lead={H:>3d}h (direct) | n={len(df):>6d} | pos={pos:>6d} | neg={neg:>6d}")

        meta = {
            "type": "per_lead_logit",
            "mode": "direct",
            "feature_names": feats,
            "dt_hours": int(args.dt_hours),
            "min_positives": int(args.min_positives),
            "class_weight": args.class_weight,
            "C": float(args.C),
            "label_prefix": args.label_prefix,
            "hours": hours,
            "lead_mapping": lead_meta,
            "seed": int(args.seed),
            "round_geo": int(args.round_geo),
        }

    # ===== Mode B: SHIFTED (time-aware default) =====
    else:
        # df[args.label_col] has already been coerced to 0/1 above (shifted_mode branch)
        leads = parse_range_or_list(args.leads)
        if not leads:
            raise ValueError("No valid --leads provided.")

        exclude = set(BASE_RESERVED) | {args.label_col}
        feats = pick_features(df, exclude)

        diag_header(
            df,
            f"shifted(base={args.label_col}) leads={leads[0]}..{leads[-1]}  time-aware={'no' if args.step_shift else 'yes'}"
        )

        # Precompute feature matrix once
        X_full = df[feats].to_numpy(dtype=np.float32, copy=False)

        last_pos = None
        last_y = None

        for L in leads:
            if args.step_shift:
                # LEGACY FIXED-STEP SHIFT (compat/debug only)
                rounded_L, k = nearest_step_hours(L, args.dt_hours)
                if k == 0:
                    print(f"[skip] lead={L}h < dt={args.dt_hours}h -> zero-step shift; skipping.")
                    continue

                df_shift = df[["time", "lat", "lon", args.label_col] + feats + ["_grp_round"]].copy()

                def _shift_group(g):
                    g = g.sort_values("time")
                    g["yL"] = g[args.label_col].shift(-k)
                    return g

                df_shift = df_shift.groupby("_grp_round", group_keys=False).apply(_shift_group)
                df_shift = df_shift.dropna(subset=["yL"]).copy()
                df_shift["yL"] = df_shift["yL"].astype(int)

                y_arr = df_shift["yL"].to_numpy(dtype=int)
                X = df_shift[feats].to_numpy(dtype=np.float32, copy=False)
                n_here = len(df_shift)
                pos = int(y_arr.sum())
                neg = int(n_here - pos)

                same_frac = None
                if last_y is not None and len(last_y) == len(y_arr):
                    same_frac = float((last_y == y_arr).mean())
                diag_per_lead(L, pos, n_here, pos_prev=last_pos, sample_same=same_frac)
                last_pos, last_y = pos, y_arr

                if pos < args.min_positives:
                    print(f"[skip] lead={L:>3d}h (-> {rounded_L:>3d}h): positives={pos} < {args.min_positives}")
                    continue

                y_for_fit = y_arr

            else:
                # TIME-AWARE FUTURE WINDOW (default)
                y_arr = future_max_label_by_point_timeaware(df, args.label_col, hours=L, grp_col="_grp_round")
                pos = int(y_arr.sum())
                neg = int(len(y_arr) - pos)

                same_frac = None
                if last_y is not None and len(last_y) == len(y_arr):
                    same_frac = float((last_y == y_arr).mean())
                diag_per_lead(L, pos, len(y_arr), pos_prev=last_pos, sample_same=same_frac)
                last_pos, last_y = pos, y_arr

                if pos < args.min_positives:
                    print(f"[skip] lead={L:>3d}h: positives={pos} < {args.min_positives}")
                    continue
                if pos / len(y_arr) < 0.005:
                    print(f"[warn] lead={L:>3d}h prevalence is very low ({pos/len(y_arr):.4f}); expect fragile metrics.")

                X = X_full
                y_for_fit = y_arr.astype(int)

            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True, copy=False)),
                ("lr", LogisticRegression(
                    C=args.C, max_iter=200, class_weight=args.class_weight,
                    solver="lbfgs", penalty="l2", random_state=args.seed
                )),
            ])
            pipe.fit(X, y_for_fit)

            # Optional eval
            if args.eval:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
                tr_idx, te_idx = next(gss.split(X, y_for_fit, groups=df["_day"]))
                p_tr = pipe.predict_proba(X[tr_idx])[:, 1]
                p_te = pipe.predict_proba(X[te_idx])[:, 1]
                metrics_rows.append(dict(
                    lead_h=L, mode=("shifted_fixedstep" if args.step_shift else "shifted_timeaware"),
                    auc_tr=_safe_auc(y_for_fit[tr_idx], p_tr), auc_te=_safe_auc(y_for_fit[te_idx], p_te),
                    ap_tr=_safe_ap(y_for_fit[tr_idx], p_tr), ap_te=_safe_ap(y_for_fit[te_idx], p_te),
                    prev=float(y_for_fit.mean()), n=len(y_for_fit)
                ))

            models[int(L)] = pipe
            trained.append(int(L))
            lead_meta[int(L)] = {
                "mode": "shifted_timeaware" if not args.step_shift else "shifted_fixedstep",
                "rounded_lead_h": int(L if not args.step_shift else rounded_L),
                "steps": int(0 if not args.step_shift else k),
                "dt_hours": int(args.dt_hours),
                "n": int(len(X)),
                "pos": int(pos),
                "neg": int(neg),
                "label_col": args.label_col,
            }
            if args.step_shift:
                print(f"[fit] lead={L:>3d}h (-> {rounded_L:>3d}h, steps={k}) | n={len(X):>6d} | pos={pos:>6d} | neg={neg:>6d}")
            else:
                print(f"[fit] lead={L:>3d}h (time-aware) | n={len(X):>6d} | pos={pos:>6d} | neg={neg:>6d}")

        meta = {
            "type": "per_lead_logit",
            "mode": "shifted_timeaware" if not args.step_shift else "shifted_fixedstep",
            "feature_names": feats,
            "dt_hours": int(args.dt_hours),
            "min_positives": int(args.min_positives),
            "class_weight": args.class_weight,
            "C": float(args.C),
            "label_col": args.label_col,
            "leads": leads,
            "lead_mapping": lead_meta,
            "seed": int(args.seed),
            "round_geo": int(args.round_geo),
        }

    # ---- SAVE BUNDLE ----
    bundle = {
        "per_lead_models": models,
        "features": meta["feature_names"],
        "meta": meta
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "trained_leads": sorted(trained)}, f, indent=2)

    # Optional metrics CSV
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows).sort_values("lead_h")
        mdf.to_csv(args.out.replace(".pkl", "_metrics.csv"), index=False)
        print(f"[metrics] wrote {args.out.replace('.pkl','_metrics.csv')}")

    # Flattening warning
    if len(trained) >= 3 and "lead_mapping" in meta:
        pos_list = [meta["lead_mapping"][L]["pos"] for L in sorted(trained)]
        if len(set(pos_list)) == 1:
            print("[warn] All trained leads have identical positive counts. "
                  "This suggests label flattening. Prefer time-aware future windows; "
                  "avoid training and evaluating on the same pre-windowed labels.")

    print(f"[write] {args.out} | trained_leads={sorted(trained)} | features={len(meta['feature_names'])}")


if __name__ == "__main__":
    main()
