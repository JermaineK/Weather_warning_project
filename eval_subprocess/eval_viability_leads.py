#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_viability_leads.py

Lead-aware evaluation for the viability model using t_to_storm_min_h.

Semantics:
  - Coincident target: y_commit (or --target)
  - Lead-L target: 1 if 0 < t_to_storm_min_h <= L (or > lead-lower)

Defaults are aligned to the new pipeline layout:
  panel : data/grid_train_gse_panel_targets.parquet
  model : models/viability_model.pkl
  metrics-json : models/viability_model_metrics.json  (for feature list / target)
  out   : results/metrics/<run_name>_viability_leads.csv (when --run-name set)

Agent: add viability-focused evaluator without touching existing leadtime scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import io_common
from utils import feature_guard

pd.options.mode.copy_on_write = True


def _parse_feature_list(tokens: Iterable[str] | None) -> list[str] | None:
    if not tokens:
        return None
    if len(tokens) == 1 and "," in str(list(tokens)[0]):
        return [t.strip() for t in str(list(tokens)[0]).split(",") if t.strip()]
    return [str(t).strip() for t in tokens if str(t).strip()]


def _load_metrics_features(path: str | None) -> tuple[list[str] | None, str | None]:
    if not path:
        return None, None
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        meta = json.loads(p.read_text())
        feats = meta.get("features")
        tgt = meta.get("target")
        if feats is not None:
            feats = [str(f) for f in feats]
        tgt = str(tgt) if tgt else None
        return feats, tgt
    except Exception:
        return None, None


def _load_panel(path: str, need_cols: Sequence[str]) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=list(dict.fromkeys(need_cols)))
    return pd.read_csv(path, low_memory=False, usecols=lambda c: c in set(need_cols))


def _coerce_binary_series(series: pd.Series, label: str, context: str) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").fillna(0)
    bad = ~vals.isin([0, 1])
    if bool(bad.any()):
        bad_vals = pd.unique(vals[bad])[:5]
        sample = ", ".join(str(v) for v in bad_vals)
        print(
            f"[warn] [{context}] label '{label}' has non-binary values (e.g. {sample}); "
            "binarizing as >0.",
            file=sys.stderr,
        )
    return (vals > 0).astype(int).to_numpy()


def _parse_leads(raw: Iterable[str]) -> list[float]:
    out: list[float] = []
    for tok in raw:
        for part in str(tok).replace(",", " ").split():
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                raise SystemExit(f"Could not parse lead-hours token '{tok}'")
    return out


def _build_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Numeric matrix in the exact feature order; NaNs -> 0."""
    Xdf = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            Xdf[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            Xdf[c] = np.nan
    return Xdf.fillna(0.0).to_numpy(dtype=float, copy=False)


def _shift_label_by_hours(
    df: pd.DataFrame,
    label: str,
    hours: int,
    time_col: str = "time",
    group_cols: Sequence[str] = ("lat", "lon"),
) -> np.ndarray:
    if hours == 0:
        return pd.to_numeric(df[label], errors="coerce").fillna(0).to_numpy()
    if time_col not in df.columns:
        raise SystemExit(f"[audit] time column '{time_col}' missing; cannot run shift audit.")
    work = df[list(group_cols) + [time_col, label]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    work = work.sort_values(list(group_cols) + [time_col])
    shifted = (
        work.groupby(list(group_cols), sort=False)[label]
        .shift(int(hours))
        .fillna(0)
        .to_numpy()
    )
    out = np.zeros(len(df), dtype=int)
    out[work.index.to_numpy()] = (pd.to_numeric(shifted, errors="coerce") > 0).astype(int)
    return out


def _audit_feature_shift(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str,
    shift_h: int,
    max_rows: int,
    max_features: int,
) -> None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        print("[audit] skipped (sklearn missing).")
        return
    if label not in df.columns:
        print(f"[audit] skipped (label '{label}' missing).")
        return
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f"[audit] sampled {len(df):,} rows for shift test.")
    feats = list(features)[: max_features or len(features)]
    y = _coerce_binary_series(df[label], label, "audit")
    try:
        y_plus = _shift_label_by_hours(df, label, abs(shift_h))
        y_minus = _shift_label_by_hours(df, label, -abs(shift_h))
    except SystemExit as exc:
        print(str(exc))
        return
    issues = []
    for f in feats:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce").fillna(0).to_numpy()
        if np.unique(x).size < 2:
            continue
        try:
            auc0 = roc_auc_score(y, x)
            aucp = roc_auc_score(y_plus, x)
            aucm = roc_auc_score(y_minus, x)
        except Exception:
            continue
        if (abs(auc0 - aucp) < 0.01 and abs(auc0 - aucm) < 0.01) or (auc0 > 0.98 and aucp < 0.7 and aucm < 0.7):
            issues.append((f, auc0, aucp, aucm))
    if issues:
        print("[audit] potential shift-invariant or leakage-like features:")
        for f, auc0, aucp, aucm in issues[:20]:
            print(f"  {f}: auc={auc0:.3f} shift+{shift_h}={aucp:.3f} shift-{shift_h}={aucm:.3f}")


def _lead_mask(dt: np.ndarray, lead_h: float, lead_lower: float) -> np.ndarray:
    """Strict future window: (lead_lower, lead_h]."""
    return np.isfinite(dt) & (dt > lead_lower) & (dt <= lead_h)


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    mets: dict[str, float] = {}
    try:
        y = np.asarray(y, dtype=int)
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos == 0 or neg == 0:
            mets["auc"] = float("nan")
            mets["prauc"] = float("nan")
        else:
            mets["auc"] = roc_auc_score(y, p)
            mets["prauc"] = average_precision_score(y, p)
    except Exception:
        mets["auc"] = float("nan")
        mets["prauc"] = float("nan")
    try:
        mets["brier"] = brier_score_loss(y, p)
    except Exception:
        mets["brier"] = float("nan")
    return mets


def _unwrap_estimator(model):
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return steps[-1] if steps else model
    return model


def _sample_df(df: pd.DataFrame, max_rows: int = 200_000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _feature_corrs(df: pd.DataFrame, features: Sequence[str], y: np.ndarray) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    yv = np.asarray(y, dtype=float)
    for f in features:
        x = pd.to_numeric(df.get(f), errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(yv)
        if mask.sum() < 2:
            continue
        corr = np.corrcoef(x[mask], yv[mask])[0, 1]
        if np.isfinite(corr):
            out.append((f, float(abs(corr)), float(corr)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _feature_mutual_info(df: pd.DataFrame, features: Sequence[str], y: np.ndarray) -> list[tuple[str, float]]:
    X = _build_matrix(df, features)
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    except Exception:
        return []
    out = [(f, float(v)) for f, v in zip(features, mi)]
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _model_importance(model, features: Sequence[str]) -> list[tuple[str, float]]:
    est = _unwrap_estimator(model)
    if hasattr(est, "feature_importances_"):
        vals = getattr(est, "feature_importances_")
        out = [(f, float(v)) for f, v in zip(features, vals)]
        return sorted(out, key=lambda t: t[1], reverse=True)
    if hasattr(est, "coef_"):
        coef = getattr(est, "coef_")
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        vals = np.abs(np.asarray(coef))
        out = [(f, float(v)) for f, v in zip(features, vals)]
        return sorted(out, key=lambda t: t[1], reverse=True)
    return []


def _run_perfect_tripwire(
    df: pd.DataFrame,
    features: Sequence[str],
    y: np.ndarray,
    model,
    stage: str,
    path: str,
    target: str | None,
) -> None:
    sample = _sample_df(df)
    if len(sample) != len(df):
        print(f"[tripwire] using sample rows={len(sample):,} from {path}")
        y_series = pd.Series(y, index=df.index)
        y = y_series.loc[sample.index].to_numpy()

    forbidden = feature_guard.forbidden_columns_for_target(features, target)
    print(f"[tripwire] forbidden features in X ({stage}): {forbidden}")

    corrs = _feature_corrs(sample, features, y)
    print("[tripwire] top-30 by |pearson|:")
    for f, absc, corr in corrs[:30]:
        print(f"  {f}: |r|={absc:.4f} r={corr:.4f}")

    mi = _feature_mutual_info(sample, features, y)
    if mi:
        print("[tripwire] top-30 by mutual_info:")
        for f, v in mi[:30]:
            print(f"  {f}: mi={v:.4f}")

    imp = _model_importance(model, features)
    if imp:
        print("[tripwire] top-30 model importance:")
        for f, v in imp[:30]:
            print(f"  {f}: importance={v:.6f}")


def _lead_summary(vals: np.ndarray) -> tuple[int, float, float]:
    finite = np.isfinite(vals)
    count = int(finite.sum())
    if count == 0:
        return 0, float("nan"), float("nan")
    return count, float(np.nanmin(vals)), float(np.nanmax(vals))


def _assert_binary(
    y: np.ndarray,
    label: str,
    *,
    allow_single: bool,
    lead_info: tuple[int, float, float] | None = None,
    panel: str | None = None,
) -> None:
    n = int(len(y))
    if n == 0:
        raise SystemExit(f"[viability-eval] empty target for {label}.")
    pos = int(np.sum(y == 1))
    neg = n - pos
    if pos == 0 or neg == 0:
        msg = f"[viability-eval] single-class target for {label}: pos={pos} neg={neg} n={n}"
        if lead_info is not None:
            non_null, lead_min, lead_max = lead_info
            msg += f" | lead_non_null={non_null} lead_min={lead_min} lead_max={lead_max}"
        if panel:
            msg += f" | panel={panel}"
        if allow_single:
            print(f"[warn] {msg}", file=sys.stderr)
        else:
            raise SystemExit(msg)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Lead-aware viability evaluation using t_to_storm_min_h.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--panel",
        default="data/grid_train_gse_panel_targets.parquet",
        help="Panel with y_commit (or y_viable), lead column, and features.",
    )
    ap.add_argument(
        "--model",
        default="models/viability_model.pkl",
        help="Joblib sklearn estimator or bundle dict with 'model'.",
    )
    ap.add_argument(
        "--model-metrics",
        default="models/viability_model_metrics.json",
        help="JSON with 'features' (and optionally 'target') for the viability model.",
    )
    # Chunking hints (accepted for pipeline compatibility; currently full in-memory)
    ap.add_argument("--chunk-rows", type=int, default=None, help="Optional chunk hint; accepted for compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows.")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Optional row-group hint; accepted for compatibility.")
    ap.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature list (space- or comma-separated). Defaults to metrics JSON if absent.",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Target column for coincident metrics. Defaults to metrics JSON target or y_commit.",
    )
    ap.add_argument(
        "--lead-col",
        default="t_to_storm_min_h",
        help="Column holding minutes-to-storm (hours).",
    )
    ap.add_argument(
        "--lead-hours",
        nargs="+",
        type=str,
        default=["24", "48", "72", "120"],
        help="Lead horizons (hours) to evaluate.",
    )
    ap.add_argument(
        "--lead-target-template",
        default=None,
        help="Optional label template for per-lead targets (e.g., 'y_knee_cross_{lead}h'). "
             "If set, per-lead labels are read directly from columns instead of lead_col windows.",
    )
    ap.add_argument(
        "--lead-lower",
        type=float,
        default=0.0,
        help="Strict lower bound for lead window (exclude current/negative).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to stamp outputs (results/metrics/<run>_viability_leads.csv).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV for metrics (defaults to run-stamped path when run-name is set).",
    )
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip work if output already exists.",
    )
    ap.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Allow single-class targets (will emit warnings instead of failing).",
    )
    ap.add_argument(
        "--allow-perfect",
        action="store_true",
        help="Allow near-perfect coincident scores without failing the run.",
    )
    ap.add_argument(
        "--regime-col",
        default="mud_high",
        help="Optional regime column for conditioned metrics (e.g., mud_high).",
    )
    ap.add_argument(
        "--regime-out",
        default=None,
        help="Optional CSV for regime-conditioned metrics.",
    )
    ap.add_argument(
        "--audit-shift-hours",
        type=int,
        default=6,
        help="Run a quick shift-causality audit using +/- this many hours (0 disables).",
    )
    ap.add_argument(
        "--audit-max-rows",
        type=int,
        default=200_000,
        help="Max rows sampled for shift audit.",
    )
    ap.add_argument(
        "--audit-max-features",
        type=int,
        default=50,
        help="Max features to test in shift audit.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    feats_from_args = _parse_feature_list(args.features)
    feats_from_metrics, tgt_from_metrics = _load_metrics_features(args.model_metrics)

    # Load estimator (allow dict bundle with 'model')
    bundle = joblib.load(args.model)
    model = bundle.get("model") if isinstance(bundle, dict) else bundle
    if model is None:
        raise SystemExit("Model bundle missing 'model' key.")

    features = feats_from_args or feats_from_metrics
    if not features and isinstance(bundle, dict):
        maybe_feats = bundle.get("features")
        if maybe_feats:
            features = list(maybe_feats)
    if not features:
        raise SystemExit("No feature list found. Provide --features, metrics JSON, or a bundle with 'features'.")
    feature_guard.assert_no_forbidden_features(
        features,
        stage="eval.viability-leads",
        path=str(args.panel),
        target=args.target or tgt_from_metrics,
    )

    target = args.target or tgt_from_metrics or "y_commit"
    lead_col = args.lead_col
    lead_hours = _parse_leads(args.lead_hours)
    lead_template = args.lead_target_template

    # Resolve default output path
    out_path = args.out
    if not out_path and args.run_name:
        out_path = f"results/metrics/{args.run_name}_viability_leads.csv"
    if not out_path:
        out_path = "results/metrics/viability_leads.csv"
    if args.skip_if_exists and Path(out_path).exists():
        print(f"[skip] output already exists: {out_path}")
        return

    need_cols = set(features)
    if lead_template:
        for h in lead_hours:
            col = lead_template.format(lead=int(float(h)))
            need_cols.add(col)
        # only require coincident target if explicitly provided
        if args.target:
            need_cols.add(target)
    else:
        need_cols |= {target, lead_col}
    df = _load_panel(args.panel, need_cols)

    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Panel missing columns: {missing}")

    # Convert lead/target to numeric
    lead_vals = None
    lead_info = (0, float("nan"), float("nan"))
    if not lead_template:
        lead_vals = pd.to_numeric(df[lead_col], errors="coerce").to_numpy(dtype=float)
        lead_info = _lead_summary(lead_vals)
        if lead_info[0] == 0:
            raise SystemExit(
                f"[viability-eval] lead column '{lead_col}' has no finite values after coercion "
                f"(panel={args.panel})."
            )

    X = _build_matrix(df, features)
    probs = model.predict_proba(X)[:, 1]

    if args.audit_shift_hours and args.audit_shift_hours != 0:
        _audit_feature_shift(
            df,
            features,
            target,
            shift_h=int(args.audit_shift_hours),
            max_rows=int(args.audit_max_rows),
            max_features=int(args.audit_max_features),
        )

    rows = []

    # Coincident metrics (info only, lead_h=0 marker)
    if lead_template is None or args.target:
        y_coincident = _coerce_binary_series(df[target], target, "viability-eval")
        _assert_binary(
            y_coincident,
            f"coincident target '{target}'",
            allow_single=args.allow_single_class,
            lead_info=lead_info,
            panel=args.panel,
        )
        coinc = _safe_metrics(y_coincident, probs)
        if (not args.allow_perfect) and (
            (coinc.get("auc", 0.0) > 0.995) or (coinc.get("prauc", 0.0) > 0.995)
        ):
            _run_perfect_tripwire(
                df,
                features,
                y_coincident,
                model,
                stage="eval.viability-leads",
                path=str(args.panel),
                target=target,
            )
            raise SystemExit(
                "[viability-eval] coincident score is near-perfect; "
                "run aborted (use --allow-perfect to override)."
            )
        pos_rate = float(y_coincident.mean() if len(y_coincident) else np.nan)
        prauc = coinc["prauc"]
        lift = float(prauc / pos_rate) if pos_rate and np.isfinite(prauc) else np.nan
        rows.append(
            {
                "kind": "coincident",
                "lead_h": 0.0,
                "pos": int(y_coincident.sum()),
                "samples": int(len(y_coincident)),
                "pos_rate": pos_rate,
                "prauc_chance": pos_rate,
                "lift": lift,
                "auc": coinc["auc"],
                "prauc": prauc,
                "brier": coinc["brier"],
                "lead_lower": args.lead_lower,
                "lead_col": lead_col,
                "target": target,
                "run_name": args.run_name or "",
            }
        )

    # Per-lead metrics
    for h in lead_hours:
        if lead_template:
            col = lead_template.format(lead=int(float(h)))
            if col not in df.columns:
                raise SystemExit(f"Lead target column not found: {col}")
            y_lead = _coerce_binary_series(df[col], col, "viability-eval")
            label_desc = f"lead_h={float(h)}h (target={col})"
        else:
            mask = _lead_mask(lead_vals, lead_h=float(h), lead_lower=float(args.lead_lower))
            y_lead = mask.astype(int)
            label_desc = f"lead_h={float(h)}h (lead_col={lead_col})"
        _assert_binary(
            y_lead,
            label_desc,
            allow_single=args.allow_single_class,
            lead_info=lead_info,
            panel=args.panel,
        )
        mets = _safe_metrics(y_lead, probs)
        pos_rate = float(y_lead.mean() if len(y_lead) else np.nan)
        prauc = mets["prauc"]
        lift = float(prauc / pos_rate) if pos_rate and np.isfinite(prauc) else np.nan
        rows.append(
            {
                "kind": "lead",
                "lead_h": float(h),
                "pos": int(y_lead.sum()),
                "samples": int(len(y_lead)),
                "pos_rate": pos_rate,
                "prauc_chance": pos_rate,
                "lift": lift,
                "auc": mets["auc"],
                "prauc": prauc,
                "brier": mets["brier"],
                "lead_lower": args.lead_lower,
                "lead_col": lead_col if not lead_template else "",
                "target": (target if not lead_template else col),
                "run_name": args.run_name or "",
            }
        )

    out_df = pd.DataFrame(rows)
    io_common.write_any(out_path, out_df)

    # Regime-conditioned metrics (e.g., mud_high vs mud_low)
    if args.regime_col and args.regime_col in df.columns:
        reg_out = args.regime_out
        if not reg_out and args.run_name:
            reg_out = f"results/metrics/{args.run_name}_viability_leads_regimes.csv"
        if not reg_out:
            reg_out = "results/metrics/viability_leads_regimes.csv"
        reg_rows = []
        reg_vals = pd.to_numeric(df[args.regime_col], errors="coerce").fillna(0.0).to_numpy()
        for reg_name, reg_mask in [("regime_high", reg_vals > 0.5), ("regime_low", reg_vals <= 0.5)]:
            if not reg_mask.any():
                continue
            for h in lead_hours:
                if lead_template:
                    col = lead_template.format(lead=int(float(h)))
                    y_lead = _coerce_binary_series(df[col], col, "viability-eval")[reg_mask]
                else:
                    mask = _lead_mask(lead_vals, lead_h=float(h), lead_lower=float(args.lead_lower))
                    y_lead = mask.astype(int)[reg_mask]
                p_lead = probs[reg_mask]
                if len(y_lead) == 0 or np.unique(y_lead).size < 2:
                    continue
                mets = _safe_metrics(y_lead, p_lead)
                pos_rate = float(np.mean(y_lead))
                prauc = mets["prauc"]
                lift = float(prauc / pos_rate) if pos_rate and np.isfinite(prauc) else np.nan
                reg_rows.append(
                    {
                        "regime": reg_name,
                        "lead_h": float(h),
                        "pos": int(np.sum(y_lead)),
                        "samples": int(len(y_lead)),
                        "pos_rate": pos_rate,
                        "prauc_chance": pos_rate,
                        "lift": lift,
                        "auc": mets["auc"],
                        "prauc": prauc,
                        "brier": mets["brier"],
                        "regime_col": args.regime_col,
                        "target": (target if not lead_template else col),
                        "run_name": args.run_name or "",
                    }
                )
        if reg_rows:
            io_common.write_any(reg_out, pd.DataFrame(reg_rows))
            print(f"[viability-eval] regime metrics -> {reg_out}")

    print(
        f"[viability-eval] rows={len(df):,} coincident_pos={y_coincident.sum():,} "
        f"features={len(features)} out={out_path}"
    )
    for r in out_df.itertuples(index=False):
        print(
            f"  lead={r.lead_h:>6.1f}h | kind={r.kind:<10} | pos={r.pos:>8,} "
            f"| AUC={r.auc:.3f} PRAUC={r.prauc:.3f} Brier={r.brier:.3f}"
        )


if __name__ == "__main__":
    main()
