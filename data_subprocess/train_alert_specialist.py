#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_alert_specialist.py

Agent: train alert-regime specialist on alert-rich subset (positives from
alerts/storm labels + mined hard negatives). Keeps model simple + memory safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import model_versioning

pd.options.mode.copy_on_write = True

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.dataset as ds  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None
    ds = None


# ---------------------------------------------------------------------------#
# IO helpers                                                                 #
# ---------------------------------------------------------------------------#

def _is_parquet(path: str | Path) -> bool:
    low = str(path).lower()
    return low.endswith((".parquet", ".parq", ".pq"))


def _peek_columns(path: str | Path) -> List[str]:
    if _is_parquet(path):
        if ds is None:
            return list(pd.read_parquet(path, nrows=1).columns)
        return ds.dataset(path).schema.names  # type: ignore[arg-type]
    return list(pd.read_csv(path, nrows=2, low_memory=False).columns)


def _stream_frames(path: str, columns: Sequence[str], chunk_rows: int | None, parquet_rows: int | None) -> Iterable[pd.DataFrame]:
    if _is_parquet(path) and ds is not None:
        scanner = ds.dataset(path).scanner(columns=list(columns), batch_size=parquet_rows or chunk_rows or None)
        for batch in scanner.to_batches():
            yield batch.to_pandas()
        return
    if _is_parquet(path):
        yield pd.read_parquet(path, columns=list(columns))
        return
    kwargs = {"usecols": list(columns), "low_memory": False}
    if "time" in columns:
        kwargs["parse_dates"] = ["time"]
    for chunk in pd.read_csv(path, chunksize=chunk_rows or None, **kwargs):
        yield chunk


# ---------------------------------------------------------------------------#
# Dataset construction                                                       #
# ---------------------------------------------------------------------------#

def _load_alert_ids(path: str | None) -> Set[int]:
    if not path:
        return set()
    if not Path(path).exists():
        print(f"[specialist] alert list missing: {path} (continuing without)")
        return set()
    df = pd.read_parquet(path) if _is_parquet(path) else pd.read_csv(path, low_memory=False)
    for candidate in ("row_id", "id", "alert_id"):
        if candidate in df.columns:
            vals = pd.to_numeric(df[candidate], errors="coerce").dropna().astype("UInt64")
            return {int(v) for v in vals.to_list()}
    return set()


def _infer_features(columns: Sequence[str], label: str, time_col: str, prefixes: Sequence[str]) -> List[str]:
    feats: List[str] = []
    pref = tuple(prefixes)
    for c in columns:
        if c in {label, time_col}:
            continue
        if pref and c.startswith(pref):
            feats.append(c)
    return feats or [c for c in columns if c not in {label, time_col}]


def _series_or_default(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Return column if present, otherwise a Series filled with default."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _build_subset(
    df: pd.DataFrame,
    label: str,
    alerts: Set[int],
    G_thr: float,
    SFI_thr: float,
    knee_thr: float,
    max_rows: int | None,
    neg_pos_ratio: float,
    seed: int,
) -> pd.DataFrame:
    label_series = _series_or_default(df, label, 0)
    pos_mask = pd.to_numeric(label_series, errors="coerce").fillna(0) > 0
    if alerts and "row_id" in df.columns:
        row_ids = pd.to_numeric(df["row_id"], errors="coerce").astype("UInt64")
        pos_mask |= row_ids.isin(alerts)

    g_vals = pd.to_numeric(_series_or_default(df, "G", 0.0), errors="coerce")
    sfi_vals = pd.to_numeric(_series_or_default(df, "SFI", 0.0), errors="coerce")
    knee_vals = pd.to_numeric(_series_or_default(df, "gka_knee_ratio", 0.0), errors="coerce")
    hard_mask = (~pos_mask) & (
        (g_vals >= G_thr) | (sfi_vals >= SFI_thr) | (np.abs(knee_vals) >= knee_thr)
    )
    neg_mask = (~pos_mask)

    pos = df.loc[pos_mask]
    hard_neg = df.loc[hard_mask]
    easy_neg = df.loc[neg_mask & (~hard_mask)]

    if neg_pos_ratio > 0 and len(pos) > 0:
        target_neg = int(neg_pos_ratio * len(pos))
        keep_hard = min(len(hard_neg), target_neg)
        negs = hard_neg.sample(n=keep_hard, random_state=seed) if keep_hard > 0 else hard_neg
        remaining = target_neg - len(negs)
        if remaining > 0 and len(easy_neg) > 0:
            take_easy = min(len(easy_neg), remaining)
            negs = pd.concat([negs, easy_neg.sample(n=take_easy, random_state=seed)], axis=0)
    else:
        negs = pd.concat([hard_neg, easy_neg], axis=0)

    subset = pd.concat([pos, negs], axis=0)
    if max_rows and len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=seed)
    return subset.sample(frac=1.0, random_state=seed)


def _collect_data(
    path: str,
    features: Sequence[str],
    label: str,
    time_col: str,
    available_cols: Sequence[str],
    alerts: Set[int],
    G_thr: float,
    SFI_thr: float,
    knee_thr: float,
    chunk_rows: int | None,
    parquet_rows: int | None,
    sample_frac: float,
    max_rows: int | None,
    neg_pos_ratio: float,
    seed: int,
) -> pd.DataFrame:
    cols = [c for c in (set(features) | {label, time_col, "row_id", "G", "SFI", "gka_knee_ratio"}) if c in set(available_cols)]
    parts: List[pd.DataFrame] = []
    seen_non_binary = False
    for chunk in _stream_frames(path, cols, chunk_rows, parquet_rows):
        if chunk.empty:
            continue
        if sample_frac and 0 < sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=seed)
        if label not in chunk.columns:
            # If label column is absent, add zeros so downstream steps do not crash.
            chunk[label] = 0
        vals = pd.to_numeric(chunk[label], errors="coerce").fillna(0)
        if not seen_non_binary:
            bad = ~vals.isin([0, 1])
            if bool(bad.any()):
                bad_vals = pd.unique(vals[bad])[:5]
                sample = ", ".join(str(v) for v in bad_vals)
                print(
                    f"[warn] [train-specialist] label '{label}' has non-binary values (e.g. {sample}); "
                    "binarizing as >0.",
                    file=sys.stderr,
                )
                seen_non_binary = True
        chunk[label] = (vals > 0).astype(int)
        chunk = _build_subset(chunk, label, alerts, G_thr, SFI_thr, knee_thr, max_rows, neg_pos_ratio, seed)
        parts.append(chunk)
        if max_rows and sum(len(p) for p in parts) > max_rows * 2:
            merged = pd.concat(parts, axis=0, ignore_index=True)
            merged = merged.sample(n=max_rows, random_state=seed) if len(merged) > max_rows else merged
            parts = [merged]
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, axis=0, ignore_index=True)
    if label not in df.columns:
        df[label] = 0
    df[label] = (pd.to_numeric(df[label], errors="coerce").fillna(0) > 0).astype(int)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


# ---------------------------------------------------------------------------#
# Model + metrics                                                            #
# ---------------------------------------------------------------------------#

def _prep_xy(df: pd.DataFrame, features: Sequence[str], label: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = (pd.to_numeric(df[label], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    return X, y


def _fit_model(X: np.ndarray, y: np.ndarray, lr: float, depth: int, max_leaf: int, seed: int):
    model = HistGradientBoostingClassifier(
        learning_rate=lr,
        max_depth=depth if depth > 0 else None,
        max_leaf_nodes=max_leaf if max_leaf > 0 else None,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _metrics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> Dict[str, float]:
    res: Dict[str, float] = {}
    y_true = np.asarray(y_true, dtype=int)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    try:
        if pos == 0 or neg == 0:
            res[f"{prefix}_roc_auc"] = float("nan")
            res[f"{prefix}_avg_precision"] = float("nan")
            print(
                f"[warn] [train-specialist] {prefix}: single-class labels (pos={pos} neg={neg}); "
                "AUC/PRAUC set to NaN.",
                file=sys.stderr,
            )
        else:
            res[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, prob))
            res[f"{prefix}_avg_precision"] = float(average_precision_score(y_true, prob))
    except Exception:
        res[f"{prefix}_roc_auc"] = float("nan")
        res[f"{prefix}_avg_precision"] = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, (prob >= 0.5).astype(int), average="binary", zero_division=0
    )
    res[f"{prefix}_precision"] = float(prec)
    res[f"{prefix}_recall"] = float(rec)
    res[f"{prefix}_f1"] = float(f1)
    return res


# ---------------------------------------------------------------------------#
# CLI + main                                                                 #
# ---------------------------------------------------------------------------#

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train alert specialist on alert-rich subset with hard negatives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", required=True, help="Training table (parquet/csv).")
    ap.add_argument("--label", default="storm", help="Label column.")
    ap.add_argument("--alerts", default=None, help="Optional alert list with row_id positives.")
    ap.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Explicit feature list (space-separated). If omitted, prefixes drive selection.",
    )
    ap.add_argument(
        "--include-prefixes",
        default="gka_,sph_,SFI,S3,zeta,div,msl_,dG_,dE_,G_,E_",
        help="Comma-separated prefixes for auto feature selection when --features not set.",
    )
    ap.add_argument("--exclude-cols", default="", help="Comma-separated columns to drop.")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--train-end", default=None, help="Time boundary for train (<= train_end).")
    ap.add_argument("--val-end", default=None, help="Time boundary for validation (>train_end and <= val_end).")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--neg-pos-ratio", type=float, default=5.0, help="Negatives per positive.")
    ap.add_argument("--G-thr", type=float, default=0.8, help="G threshold for hard negative mining.")
    ap.add_argument("--SFI-thr", type=float, default=0.8, help="SFI threshold for hard negative mining.")
    ap.add_argument("--knee-thr", type=float, default=0.6, help="|gka_knee_ratio| threshold for hard negative mining.")
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--calibrate", action="store_true", help="Fit calibrator on validation split.")
    ap.add_argument("--calibration-out", default=None, help="Optional path to save calibrator.")
    ap.add_argument("--model-out", required=True, help="Where to write fitted model (joblib).")
    ap.add_argument("--metrics-out", default=None, help="Optional JSON metrics output.")
    ap.add_argument("--model-dir", default=None, help="Base directory for versioned model outputs.")
    ap.add_argument("--run-name", default=None, help="Run name for versioned model subdir.")
    ap.add_argument("--run-id", default=None, help="Explicit version subdir name under model-dir.")
    ap.add_argument("--write-latest", action="store_true", help="Update model-dir/latest with output copies.")
    ap.add_argument(
        "--allow-existing-version",
        action="store_true",
        help="Allow writing into an existing versioned model folder.",
    )
    ap.add_argument("--provenance-out", default=None, help="Optional provenance JSON output path.")
    ap.add_argument("--config-sha256", default=None, help="Optional config SHA256 for provenance.")
    ap.add_argument("--git-commit", default=None, help="Optional git commit hash for provenance.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", "--chunk-rows", type=int, default=None)
    ap.add_argument("--parquet-rows", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    prefixes = [p.strip() for p in args.include_prefixes.split(",") if p.strip()]
    exclude = {c.strip() for c in args.exclude_cols.split(",") if c.strip()}
    alerts = _load_alert_ids(args.alerts)

    # Agent: optional versioned outputs/provenance; training math unchanged.
    versioned = None
    if args.model_dir:
        versioned = model_versioning.resolve_version_dir(
            Path(args.model_dir),
            run_name=args.run_name,
            run_id=args.run_id,
            allow_existing=args.allow_existing_version,
        )

    cols = _peek_columns(args.train)
    features = args.features
    if not features:
        features = _infer_features(cols, args.label, args.time_col, prefixes)
    features = [f for f in features if f not in exclude]
    if not features:
        raise SystemExit("No feature columns selected.")

    df = _collect_data(
        args.train,
        features,
        args.label,
        args.time_col,
        cols,
        alerts,
        args.G_thr,
        args.SFI_thr,
        args.knee_thr,
        args.chunksize,
        args.parquet_rows,
        args.sample_frac,
        args.max_rows,
        args.neg_pos_ratio,
        args.seed,
    )
    if df.empty:
        raise SystemExit("Training data is empty after filtering/sampling.")

    time_vals = pd.to_datetime(df[args.time_col], errors="coerce")
    if args.train_end:
        train_end = pd.to_datetime(args.train_end)
        val_end = pd.to_datetime(args.val_end) if args.val_end else None
        train_mask = time_vals <= train_end
        val_mask = (time_vals > train_end) & (time_vals <= val_end) if val_end is not None else ~train_mask
    else:
        idx_train, idx_val = train_test_split(df.index, test_size=0.2, random_state=args.seed, shuffle=True)
        train_mask = df.index.isin(idx_train)
        val_mask = df.index.isin(idx_val)

    if not train_mask.any() or not val_mask.any():
        raise SystemExit("Train/validation split is empty; adjust date boundaries.")

    df_train = df.loc[train_mask]
    df_val = df.loc[val_mask]

    X_train, y_train = _prep_xy(df_train, features, args.label)
    X_val, y_val = _prep_xy(df_val, features, args.label)

    model = _fit_model(X_train, y_train, args.learning_rate, args.max_depth, args.max_leaf_nodes, args.seed)
    prob_val = model.predict_proba(X_val)[:, 1]

    metrics: Dict[str, float] = {}
    metrics.update(_metrics(y_val, prob_val, "specialist_base"))
    if "transition_class" in df_val.columns:
        tc = pd.to_numeric(df_val["transition_class"], errors="coerce")
        for cls, g in df_val.groupby(tc):
            if pd.isna(cls):
                continue
            rate = pd.to_numeric(g[args.label], errors="coerce").fillna(0).mean()
            metrics[f"transition_class_{int(cls)}_storm_rate"] = float(rate)

    calibrated = None
    if args.calibrate:
        calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        calibrated.fit(X_val, y_val)
        prob_cal = calibrated.predict_proba(X_val)[:, 1]
        metrics.update(_metrics(y_val, prob_cal, "specialist_calibrated"))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    if calibrated and args.calibration_out:
        joblib.dump(calibrated, args.calibration_out)

    versioned_outputs: List[Path] = []
    if versioned:
        version_dir = versioned.version_dir
        version_model = version_dir / "model.pkl"
        joblib.dump(model, version_model)
        versioned_outputs.append(version_model)
        if calibrated:
            version_cal = version_dir / "calibrator.pkl"
            joblib.dump(calibrated, version_cal)
            versioned_outputs.append(version_cal)

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if versioned:
        version_metrics = versioned.version_dir / "metrics.json"
        version_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        versioned_outputs.append(version_metrics)

        provenance_path = Path(args.provenance_out) if args.provenance_out else (versioned.version_dir / "provenance.json")
        git_commit = args.git_commit or model_versioning.git_commit(Path(__file__).resolve().parents[1])
        outputs = [Path(args.model_out)]
        if calibrated and args.calibration_out:
            outputs.append(Path(args.calibration_out))
        if args.metrics_out:
            outputs.append(Path(args.metrics_out))
        outputs.extend(versioned_outputs)
        source_inputs = [Path(args.train)]
        if args.alerts:
            source_inputs.append(Path(args.alerts))
        model_versioning.write_provenance(
            provenance_path,
            run_id=versioned.version_id,
            run_name=args.run_name,
            config_sha256=args.config_sha256,
            git_commit_hash=git_commit,
            source_inputs=source_inputs,
            outputs=outputs,
            extra={"model_kind": "train-alert-specialist"},
        )
        versioned_outputs.append(provenance_path)
        if args.write_latest:
            model_versioning.update_latest(Path(args.model_dir), versioned.version_dir, versioned_outputs)

    print(
        f"[train-specialist] model -> {args.model_out} rows train/val: {len(df_train):,}/{len(df_val):,} "
        f"alerts_used={len(alerts):,}"
    )
    if args.metrics_out:
        print(f"[train-specialist] metrics -> {args.metrics_out}")
    if versioned:
        print(f"[train-specialist] versioned dir -> {versioned.version_dir}")
        if args.write_latest:
            print(f"[train-specialist] latest pointer -> {Path(args.model_dir) / 'latest'}")


if __name__ == "__main__":
    main()
