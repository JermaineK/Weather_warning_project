#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_viability_model.py

Fit a simple viability model P(y_viable | G, S, E, ...).
Default: logistic regression with balanced class weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Sequence, Optional, Dict
from fnmatch import fnmatch

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import model_versioning


# Agent: baseline viable/not-viable fit; keep maths simple and interpretable.

# Guardrails for leakage-prone columns.
BLOCKLIST_DEFAULT = [
    "storm*",
    "track*",
    "t_to_*",
    "near_storm",
    "row_id",
    "time",
    "time_hr",
    "*_point",
    "*_window",
    "vmax*",
    "dist*track*",
]


def _parse_csv_list(raw: str) -> List[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _blocked_features(features: Sequence[str], patterns: Sequence[str]) -> List[str]:
    blocked: List[str] = []
    for feat in features:
        for pat in patterns:
            if fnmatch(feat, pat):
                blocked.append(feat)
                break
    return sorted(set(blocked))


def _load_feature_manifest(path: str | None) -> List[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Feature manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "features" in data:
        return list(data["features"])
    if isinstance(data, list):
        return data
    raise SystemExit(f"Invalid feature manifest format in {p}")


def _feature_meta_for(feature: str, manifest: List[dict]) -> Optional[dict]:
    exact = [m for m in manifest if str(m.get("name")) == feature]
    if exact:
        return exact[0]
    for meta in manifest:
        name = str(meta.get("name", ""))
        if "*" in name or "?" in name:
            if fnmatch(feature, name):
                return meta
    return None


def _enforce_feature_manifest(
    features: Sequence[str],
    manifest_path: str | None,
    require_metadata: bool,
) -> None:
    manifest = _load_feature_manifest(manifest_path)
    if not manifest and require_metadata:
        raise SystemExit("Feature manifest is required but empty or missing.")
    if not manifest:
        return
    missing = []
    non_causal = []
    for feat in features:
        meta = _feature_meta_for(feat, manifest)
        if meta is None:
            missing.append(feat)
            continue
        if meta.get("past_only") is False:
            non_causal.append(feat)
    if missing and require_metadata:
        raise SystemExit(f"Missing feature metadata for: {missing}")
    if non_causal:
        raise SystemExit(f"Non-causal features are blocked for training: {non_causal}")


def _resolve_key_paths(model_out: str, train_keys_out: str | None, val_keys_out: str | None) -> tuple[Path, Path]:
    base = Path(model_out).with_suffix("")
    train_path = Path(train_keys_out) if train_keys_out else base.with_name(base.name + "_train_keys.parquet")
    val_path = Path(val_keys_out) if val_keys_out else base.with_name(base.name + "_val_keys.parquet")
    return train_path, val_path


def _write_keys(df: pd.DataFrame, key_cols: Sequence[str], out_path: Path) -> None:
    if not key_cols:
        return
    keys = df[list(key_cols)].copy()
    if "time" in keys.columns:
        keys["time"] = pd.to_datetime(keys["time"], errors="coerce")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".parq", ".pq", ".pqt"}:
        keys.to_parquet(out_path, index=False)
    else:
        keys.to_csv(out_path, index=False)


def _count_overlap(train_keys: pd.DataFrame, val_keys: pd.DataFrame, key_cols: Sequence[str]) -> int:
    merged = train_keys.merge(val_keys, on=list(key_cols), how="inner")
    return int(len(merged))


def _append_json_list(path: Path, entry: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, object]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                payload = loaded
        except Exception:
            payload = []
    payload.append(entry)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _hash_feature_list(features: Sequence[str]) -> str:
    raw = "\n".join(features)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _table_fingerprint(
    df: pd.DataFrame,
    cols: Sequence[str],
    key_cols: Sequence[str],
    sample_rows: int = 10_000,
) -> str:
    sample_cols = [c for c in cols if c in df.columns]
    if not sample_cols:
        return ""
    sample = df[sample_cols].copy()
    sort_cols = [c for c in key_cols if c in sample.columns]
    if sort_cols:
        sample = sample.sort_values(sort_cols)
    if len(sample) > sample_rows:
        sample = sample.head(sample_rows)
    h = pd.util.hash_pandas_object(sample, index=True).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def _lead_summary(df: pd.DataFrame, lead_col: str = "lead_h") -> Optional[object]:
    if lead_col not in df.columns:
        return None
    vals = pd.to_numeric(df[lead_col], errors="coerce").dropna().unique().tolist()
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    if len(vals) <= 10:
        return sorted(float(v) for v in vals)
    return "mixed"


def _infer_task_kind(raw: str, label: str, df: pd.DataFrame) -> str:
    if raw and str(raw).lower() != "auto":
        return str(raw).lower()
    label_l = str(label).lower()
    if "coincident" in label_l or "now" in label_l:
        return "coincident"
    if "lead" in label_l or "future" in label_l or "viable" in label_l:
        return "lead"
    if "t_to_storm_min_h" in df.columns or "lead_h" in df.columns:
        return "lead"
    return "coincident"


def _label_sample(df: pd.DataFrame, key_cols: Sequence[str], label: str, n: int = 10) -> List[Dict[str, object]]:
    cols = [c for c in key_cols if c in df.columns]
    if label in df.columns:
        cols.append(label)
    if not cols:
        return []
    sample = df[cols].head(n).copy()
    if "time" in sample.columns:
        sample["time"] = sample["time"].astype(str)
    return sample.to_dict(orient="records")

def _load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".parquet", ".parq", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _add_lead_features(df: pd.DataFrame, horizon: float = 240.0) -> pd.DataFrame:
    """Add lead-derived helper columns if lead is present."""
    if "t_to_storm_min_h" not in df.columns:
        return df
    lead = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
    lead_clip = lead.clip(lower=0.0, upper=horizon)
    df = df.copy()
    df["lead_clip"] = lead_clip
    df["lead_norm"] = 1.0 - (lead_clip / horizon)
    df["lead_inv"] = 1.0 / (1.0 + lead_clip)
    if "G_struct" in df.columns:
        df["G_lead_norm"] = df["G_struct"] * df["lead_norm"]
    # coarse lead band for stratified sampling (optional downstream)
    bins = [0, 24, 72, 120, horizon, np.inf]
    labels = ["0-24", "24-72", "72-120", "120-240", ">240"]
    df["lead_band"] = pd.cut(lead.fillna(horizon + 1), bins=bins, labels=labels, right=True)
    return df


def _filter_by_lead(df: pd.DataFrame, min_lead: float | None, max_lead: float | None) -> pd.DataFrame:
    if "t_to_storm_min_h" not in df.columns or (min_lead is None and max_lead is None):
        return df
    lead = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_lead is not None:
        mask &= lead >= float(min_lead)
    if max_lead is not None:
        mask &= lead <= float(max_lead)
    filtered = df.loc[mask]
    if filtered.empty:
        raise SystemExit(
            f"Lead filter produced empty frame (min_lead={min_lead}, max_lead={max_lead}). "
            "Relax the bounds or check lead column."
        )
    return filtered


def _subsample(
    df: pd.DataFrame,
    target: str,
    neg_pos_ratio: float,
    sample_frac: float,
    seed: int,
    max_train_rows: int | None = None,
) -> pd.DataFrame:
    if target not in df:
        raise SystemExit(f"Target column '{target}' not found in training data.")

    if sample_frac and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    if "lead_band" in df.columns:
        groups = []
        total_rows = 0
        for _, gdf in df.groupby("lead_band"):
            pos = gdf[gdf[target] == 1]
            neg = gdf[gdf[target] == 0]
            if len(pos) == 0 and len(neg) == 0:
                continue
            if len(pos) == 0 or len(neg) == 0:
                groups.append(gdf)
                continue
            if neg_pos_ratio > 0:
                keep_neg = min(len(neg), int(neg_pos_ratio * len(pos)))
                neg = neg.sample(n=keep_neg, random_state=seed)
            g_take = pd.concat([pos, neg], axis=0)
            groups.append(g_take)
            total_rows += len(g_take)
            # keep accumulator bounded
            if max_train_rows and total_rows > max_train_rows * 2:
                merged = pd.concat(groups, axis=0)
                merged = merged.sample(n=max_train_rows, random_state=seed)
                groups = [merged]
                total_rows = len(merged)
        if groups:
            df_fit = pd.concat(groups, axis=0)
            if max_train_rows and len(df_fit) > max_train_rows:
                df_fit = df_fit.sample(n=max_train_rows, random_state=seed)
            return df_fit.sample(frac=1.0, random_state=seed)

    # fallback: original class-balanced sampling
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]
    if neg_pos_ratio > 0 and len(pos) > 0 and len(neg) > 0:
        keep_neg = min(len(neg), int(neg_pos_ratio * len(pos)))
        neg = neg.sample(n=keep_neg, random_state=seed)
    df_fit = pd.concat([pos, neg], axis=0)
    if max_train_rows and len(df_fit) > max_train_rows:
        df_fit = df_fit.sample(n=max_train_rows, random_state=seed)
    return df_fit.sample(frac=1.0, random_state=seed)


def _prepare_xy(df: pd.DataFrame, features: Sequence[str], target: str):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy()
    return X, y


def _fit_model(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=500, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _metrics(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    return {"roc_auc": float(auc), "avg_precision": float(ap)}


def _coeff_table(model: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
    lr = model.named_steps["lr"]
    coefs = lr.coef_.ravel()
    return pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values(
        "abs_coef", ascending=False
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a viability model on GSE panel features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--train", required=True, help="Training table with y_viable.")
    ap.add_argument("--target", default="y_viable", help="Target column.")
    ap.add_argument(
        "--features",
        nargs="+",
        default=["G_struct", "S_shear", "E_energy"],
        help="Feature columns to use (space-separated or comma-separated).",
    )
    ap.add_argument(
        "--blocklist",
        default="",
        help="Comma-separated glob patterns to block from training features (added to defaults).",
    )
    ap.add_argument(
        "--features-manifest",
        default=None,
        help="Optional JSON manifest mapping features to metadata (past_only, source, window).",
    )
    ap.add_argument(
        "--require-feature-metadata",
        action="store_true",
        help="Fail if any training feature lacks manifest metadata.",
    )
    ap.add_argument("--min-lead", type=float, default=None, help="Optional minimum lead (hours) to include (>=).")
    ap.add_argument("--max-lead", type=float, default=None, help="Optional maximum lead (hours) to include (<=).")
    ap.add_argument("--neg-pos-ratio", type=float, default=3.0, help="Max negatives per positive (0 disables).")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="Optional overall subsample fraction (0-1].")
    ap.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength for logistic regression.")
    ap.add_argument("--model-out", required=True, help="Output path for fitted model (.pkl).")
    ap.add_argument("--metrics-json", default=None, help="Where to write metrics JSON.")
    ap.add_argument("--coefs-csv", default=None, help="Where to write coefficient table CSV.")
    ap.add_argument(
        "--task-kind",
        default="auto",
        choices=["auto", "lead", "coincident"],
        help="Task kind metadata for metrics (auto infers from label/data).",
    )
    ap.add_argument(
        "--diagnostics-out-dir",
        default="results/diagnostics",
        help="Directory for diagnostics JSON outputs.",
    )
    ap.add_argument(
        "--diagnostic-shuffle-labels",
        action="store_true",
        help="Train/eval with shuffled labels and fail if skill does not collapse.",
    )
    ap.add_argument(
        "--key-cols",
        default="time,ilat,ilon",
        help="Comma-separated key columns to persist for overlap checks.",
    )
    ap.add_argument("--train-keys-out", default=None, help="Optional path to write train keys table.")
    ap.add_argument("--val-keys-out", default=None, help="Optional path to write val keys table.")
    ap.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow train/val key overlap (not recommended).",
    )
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
    ap.add_argument("--max-train-rows", type=int, default=2_000_000, help="Optional cap on rows for fitting after sampling.")
    ap.add_argument(
        "--chunksize",
        "--chunk-rows",
        "--chunk_rows",
        "--parquet-rows",
        "--parquet_rows",
        type=int,
        default=None,
        help="Optional chunk size for streaming train input (0/None = load whole file).",
    )
    args = ap.parse_args()

    # normalize features: allow comma-separated single arg or space list
    if len(args.features) == 1 and "," in args.features[0]:
        args.features = [f.strip() for f in args.features[0].split(",") if f.strip()]
    key_cols = _parse_csv_list(args.key_cols)
    block_patterns = BLOCKLIST_DEFAULT + _parse_csv_list(args.blocklist)

    # Agent: optional versioned outputs/provenance; training math unchanged.
    versioned = None
    if args.model_dir:
        versioned = model_versioning.resolve_version_dir(
            Path(args.model_dir),
            run_name=args.run_name,
            run_id=args.run_id,
            allow_existing=args.allow_existing_version,
        )

    if args.chunksize and args.chunksize > 0:
        chunk_rows = int(args.chunksize)
        if args.train.lower().endswith((".parquet", ".parq", ".pq")):
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(args.train)
            dfs = [batch.to_pandas() for batch in pf.iter_batches(batch_size=chunk_rows)]
        else:
            dfs = list(pd.read_csv(args.train, low_memory=False, chunksize=chunk_rows))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _load_table(args.train)

    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        raise SystemExit(f"Missing key columns in training data: {missing_keys}")

    # Optional lead filtering to create horizon-specific models
    df = _filter_by_lead(df, args.min_lead, args.max_lead)

    # Add lead-derived helper features for radial/time structure
    df = _add_lead_features(df, horizon=float(args.max_lead) if args.max_lead else 240.0)

    df_fit = _subsample(df, args.target, args.neg_pos_ratio, args.sample_frac, args.seed, args.max_train_rows)
    print(f"[train] using {len(df_fit):,} rows after subsample (features={args.features})")

    features = [f for f in args.features]
    blocked = _blocked_features(features, block_patterns)
    if blocked:
        raise SystemExit(f"Blocked columns in training features: {blocked}")
    features = [f for f in features if f not in key_cols]
    if not features:
        raise SystemExit("No feature columns selected after removing key columns.")
    print(f"[train] feature list -> {features}")
    _enforce_feature_manifest(features, args.features_manifest, args.require_feature_metadata)

    # Ensure we have both classes after subsampling; otherwise fall back to full data
    cls_counts = df_fit[args.target].value_counts(dropna=False)
    if cls_counts.nunique() == 1 or len(cls_counts) < 2:
        print(
            "[warn] Subsample produced a single-class dataset "
            f"({cls_counts.to_dict()}); retrying with full data and no class cap."
        )
        df_fit = _subsample(df, args.target, neg_pos_ratio=0, sample_frac=1.0, seed=args.seed)
        cls_counts = df_fit[args.target].value_counts(dropna=False)
        if cls_counts.nunique() == 1 or len(cls_counts) < 2:
            raise SystemExit(
                "Training data contains only one class even after retry.\n"
                f"Class counts: {cls_counts.to_dict()}\n"
                "This usually means build_viability_targets.py produced y_viable=0 for all rows. "
                "Check lead-window detection and g_min filtering there."
            )
        print(f"[train] retry succeeded; using {len(df_fit):,} rows with class counts {cls_counts.to_dict()}")

    task_kind = _infer_task_kind(args.task_kind, args.target, df_fit)
    if task_kind == "lead" and "coincident" in str(args.target).lower():
        raise SystemExit("Target column looks coincident but task_kind=lead; check training label selection.")
    y_vals = pd.to_numeric(df_fit[args.target], errors="coerce").fillna(0).astype(int)
    y_nunique = int(y_vals.nunique(dropna=True))
    if y_nunique < 2:
        raise SystemExit(f"Target '{args.target}' is constant after sampling; check upstream labels.")
    feature_set_id = _hash_feature_list(features)
    table_fingerprint = _table_fingerprint(df_fit, list(features) + [args.target] + list(key_cols), key_cols)
    lead_summary = _lead_summary(df_fit, "lead_h")
    label_entry = {
        "model_kind": "train-viability",
        "label_col": args.target,
        "task_kind": task_kind,
        "y_mean": float(y_vals.mean()),
        "y_sum": int(y_vals.sum()),
        "y_nunique": y_nunique,
        "n_rows": int(len(df_fit)),
        "feature_count": int(len(features)),
        "features": list(features),
        "key_cols": list(key_cols),
        "lead_h": lead_summary,
        "feature_set_id": feature_set_id,
        "table_fingerprint": table_fingerprint,
        "label_sample": _label_sample(df_fit, key_cols, args.target),
    }
    diag_dir = Path(args.diagnostics_out_dir)
    _append_json_list(diag_dir / "train_label_summary.json", label_entry)
    feat_preview = ", ".join(list(features)[:20])
    if len(features) > 20:
        feat_preview += ", ..."
    print(f"[train] label={args.target} mean={label_entry['y_mean']:.4f} sum={label_entry['y_sum']} "
          f"nunique={label_entry['y_nunique']}")
    print(f"[train] features ({len(features)}): {feat_preview}")

    X, y = _prepare_xy(df_fit, features, args.target)
    idx_train, idx_val = train_test_split(
        df_fit.index, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    df_train = df_fit.loc[idx_train]
    df_val = df_fit.loc[idx_val]

    if key_cols:
        for c in key_cols:
            if c not in df_fit.columns:
                raise SystemExit(f"Key column '{c}' not found in training data.")
        train_keys_out, val_keys_out = _resolve_key_paths(args.model_out, args.train_keys_out, args.val_keys_out)
        train_keys = df_train[list(key_cols)].copy()
        val_keys = df_val[list(key_cols)].copy()
        overlap = _count_overlap(train_keys, val_keys, key_cols)
        _write_keys(train_keys, key_cols, train_keys_out)
        _write_keys(val_keys, key_cols, val_keys_out)
        if overlap > 0 and not args.allow_overlap:
            raise SystemExit(f"Train/val overlap detected: {overlap} rows (keys written).")

    X_train, y_train = _prepare_xy(df_train, features, args.target)
    X_val, y_val = _prepare_xy(df_val, features, args.target)

    model = _fit_model(X_train, y_train, args.C, args.seed)
    train_metrics = _metrics(model, X_train, y_train)
    val_metrics = _metrics(model, X_val, y_val)

    if args.diagnostic_shuffle_labels:
        rng = np.random.default_rng(args.seed)
        y_shuf = rng.permutation(y_train)
        shuffle_model = _fit_model(X_train, y_shuf, args.C, args.seed)
        shuffle_metrics = _metrics(shuffle_model, X_val, y_val)
        base_rate = float(np.mean(y_val)) if len(y_val) else float("nan")
        shuffle_entry = {
            "model_kind": "train-viability",
            "label_col": args.target,
            "task_kind": task_kind,
            "feature_set_id": feature_set_id,
            "table_fingerprint": table_fingerprint,
            "lead_h": lead_summary,
            "base_rate": base_rate,
            "shuffle_roc_auc": shuffle_metrics.get("roc_auc"),
            "shuffle_avg_precision": shuffle_metrics.get("avg_precision"),
        }
        _append_json_list(Path(args.diagnostics_out_dir) / "shuffle_test_metrics.json", shuffle_entry)
        auc = shuffle_metrics.get("roc_auc")
        prauc = shuffle_metrics.get("avg_precision")
        auc_ok = (auc is None) or (not np.isfinite(auc)) or (auc <= 0.6)
        pr_ok = (prauc is None) or (not np.isfinite(prauc)) or (abs(float(prauc) - base_rate) <= 0.05)
        if not (auc_ok and pr_ok):
            raise SystemExit(
                f"Shuffled-label test did not collapse (auc={auc}, prauc={prauc}, base_rate={base_rate})."
            )

    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    print(f"[save] model -> {out_model}")

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "features": list(features),
        "target": args.target,
        "label_col": args.target,
        "task_kind": task_kind,
        "lead_h": lead_summary,
        "feature_set_id": feature_set_id,
        "table_fingerprint": table_fingerprint,
    }
    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))
        print(f"[save] metrics -> {args.metrics_json}")

    versioned_outputs: List[Path] = []
    if versioned:
        version_dir = versioned.version_dir
        version_model = version_dir / "model.pkl"
        joblib.dump(model, version_model)
        versioned_outputs.append(version_model)
        version_metrics = version_dir / "metrics.json"
        version_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        versioned_outputs.append(version_metrics)

    if args.coefs_csv:
        coef_df = _coeff_table(model, features)
        Path(args.coefs_csv).parent.mkdir(parents=True, exist_ok=True)
        coef_path = Path(args.coefs_csv)
        if coef_path.suffix.lower() in {".parquet", ".parq", ".pq", ".pqt"}:
            coef_df.to_parquet(coef_path, index=False)
        else:
            coef_df.to_csv(coef_path, index=False)
        print(f"[save] coefficients -> {coef_path}")

    if versioned:
        provenance_path = Path(args.provenance_out) if args.provenance_out else (versioned.version_dir / "provenance.json")
        git_commit = args.git_commit or model_versioning.git_commit(Path(__file__).resolve().parents[1])
        outputs = [out_model]
        if args.metrics_json:
            outputs.append(Path(args.metrics_json))
        if args.coefs_csv:
            outputs.append(Path(args.coefs_csv))
        outputs.extend(versioned_outputs)
        model_versioning.write_provenance(
            provenance_path,
            run_id=versioned.version_id,
            run_name=args.run_name,
            config_sha256=args.config_sha256,
            git_commit_hash=git_commit,
            source_inputs=[Path(args.train)],
            outputs=outputs,
            extra={"model_kind": "train-viability"},
        )
        versioned_outputs.append(provenance_path)
        if args.write_latest:
            model_versioning.update_latest(Path(args.model_dir), versioned.version_dir, versioned_outputs)
        print(f"[save] versioned dir -> {versioned.version_dir}")
        if args.write_latest:
            print(f"[save] latest pointer -> {Path(args.model_dir) / 'latest'}")


if __name__ == "__main__":
    main()
