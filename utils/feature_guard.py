from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Dict


# Agent: central guardrails for leakage-prone feature names (shared by training + eval).

FORBIDDEN_SUBSTRINGS = [
    "near_storm",
    "t_to_storm",
    "storm_window",
    "storm_point",
    "storm_id",
    "storm",
    "row_id",
    "time_hr",
    "lead_h",
    "label",
    "target",
]

FORBIDDEN_REGEX = [
    r".*t_to_.*",
    r".*storm.*",
    r".*future.*|.*ahead.*|.*lead.*",
]

META_COLS = [
    "time",
    "lat",
    "lon",
    "ilat",
    "ilon",
    "row_id",
    "cell_id",
    "lead_h",
    "time_hr",
]

_FORBIDDEN_RX = [re.compile(pat, flags=re.IGNORECASE) for pat in FORBIDDEN_REGEX]
_META_SET = {c.lower() for c in META_COLS}
_SUBS = [s.lower() for s in FORBIDDEN_SUBSTRINGS]
_CAUSAL_BLOCK_TOKENS = ("lock", "future", "post", "tplus")
_CAUSAL_ALLOW_TOKENS = ("past", "lag")
_TARGET_POLICY_CACHE: Dict[str, Dict[str, List[str]]] | None = None


def _norm(col: str) -> str:
    return str(col).strip().lower()


def _load_target_policy() -> Dict[str, Dict[str, List[str]]]:
    global _TARGET_POLICY_CACHE
    if _TARGET_POLICY_CACHE is not None:
        return _TARGET_POLICY_CACHE
    cfg_path = Path("config/forbidden_features.yaml")
    if not cfg_path.exists():
        _TARGET_POLICY_CACHE = {}
        return _TARGET_POLICY_CACHE
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}
    # Normalize to list-of-strings
    out: Dict[str, Dict[str, List[str]]] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        forbid = [str(x) for x in (val.get("forbid") or [])]
        allow = [str(x) for x in (val.get("allow_lagged_variants") or [])]
        out[str(key)] = {"forbid": forbid, "allow_lagged_variants": allow}
    _TARGET_POLICY_CACHE = out
    return out


def _target_family(target: str | None) -> str | None:
    if not target:
        return None
    t = str(target).strip().lower()
    if not t:
        return None
    policies = _load_target_policy()
    for fam in policies.keys():
        base = fam.replace("_targets", "").lower()
        if base and base in t:
            return fam
    return None


def _policy_for_target(target: str | None) -> Tuple[List[str], List[str]]:
    fam = _target_family(target)
    if not fam:
        return [], []
    policy = _load_target_policy().get(fam, {})
    return list(policy.get("forbid") or []), list(policy.get("allow_lagged_variants") or [])


def forbidden_columns(
    columns: Iterable[str],
    allow: Iterable[str] | None = None,
    extra_forbidden: Iterable[str] | None = None,
) -> List[str]:
    allow_set = {str(a).strip().lower() for a in (allow or [])}
    extra_set = {str(a).strip().lower() for a in (extra_forbidden or [])}
    bad: List[str] = []
    for c in columns:
        name = _norm(c)
        if any(tok in name for tok in _CAUSAL_BLOCK_TOKENS) and not any(
            allow in name for allow in _CAUSAL_ALLOW_TOKENS
        ):
            bad.append(str(c))
            continue
        if name in allow_set:
            continue
        if name in extra_set:
            bad.append(str(c))
            continue
        if name in _META_SET:
            bad.append(str(c))
            continue
        if any(sub in name for sub in _SUBS):
            bad.append(str(c))
            continue
        if any(rx.search(name) for rx in _FORBIDDEN_RX):
            bad.append(str(c))
            continue
    return sorted(dict.fromkeys(bad))


def candidate_leak_columns(columns: Iterable[str]) -> List[str]:
    cols = list(columns)
    bad = set(forbidden_columns(cols))
    for c in cols:
        if _norm(c) in _META_SET:
            bad.add(str(c))
    return sorted(bad)


def forbidden_columns_for_target(columns: Iterable[str], target: str | None) -> List[str]:
    forbid, allow = _policy_for_target(target)
    return forbidden_columns(columns, allow=allow, extra_forbidden=forbid)


def assert_no_forbidden_features(
    features: Sequence[str],
    *,
    stage: str,
    path: str,
    allow: Iterable[str] | None = None,
    target: str | None = None,
) -> None:
    forbid_extra: Iterable[str] | None = None
    allow_extra: Iterable[str] | None = None
    if target:
        forbid_extra, allow_extra = _policy_for_target(target)
    allow_set = list(allow or [])
    if allow_extra:
        allow_set.extend(allow_extra)
    bad = forbidden_columns(features, allow=allow_set, extra_forbidden=forbid_extra)
    if bad:
        raise SystemExit(
            f"[feature-guard] {stage}: forbidden columns in features: {bad} | path={path}"
        )


def scan_leakage_auc(
    df,
    label: str,
    *,
    stage: str,
    path: str,
    max_rows: int = 200_000,
    auc_threshold: float = 0.999,
    feature_cols: Sequence[str] | None = None,
) -> List[dict]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
    except Exception:
        print("[feature-guard] leakage scan skipped (missing numpy/pandas/sklearn).")
        return []

    if label not in df.columns:
        print(f"[feature-guard] leak scan skipped: label '{label}' missing.")
        return []

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
        print(f"[feature-guard] leak scan sampled {len(df):,} rows from {path}.")

    y = pd.to_numeric(df[label], errors="coerce").fillna(0).to_numpy()
    y_bin = (y > 0).astype(int)
    if np.unique(y_bin).size < 2:
        print(f"[feature-guard] leak scan skipped: label '{label}' is single-class.")
        return []

    findings: List[dict] = []
    if feature_cols:
        cols = [c for c in feature_cols if c in df.columns and c != label]
        candidates = candidate_leak_columns(cols)
    else:
        candidates = candidate_leak_columns(df.columns)
    if not candidates:
        return findings

    for col in candidates:
        if col not in df.columns or col == label:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().all():
            continue
        xf = x.fillna(0.0).to_numpy()
        try:
            auc_pos = roc_auc_score(y_bin, xf)
            auc_neg = roc_auc_score(y_bin, -xf)
            best = max(auc_pos, auc_neg)
        except Exception:
            continue
        if best <= auc_threshold:
            continue

        def _stats(mask):
            vals = x[mask]
            return {
                "mean": float(np.nanmean(vals)),
                "median": float(np.nanmedian(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
            }

        stats0 = _stats(y_bin == 0)
        stats1 = _stats(y_bin == 1)
        example_cols = [label, col]
        for extra in ("time", "lat", "lon", "row_id"):
            if extra in df.columns and extra not in example_cols:
                example_cols.append(extra)
        ex = df[example_cols].copy()
        ex["__abs"] = np.abs(x.to_numpy())
        ex = ex.sort_values("__abs", ascending=False).head(10).drop(columns="__abs")

        print(
            f"[feature-guard] LEAK-CHECK {stage}: column '{col}' "
            f"auc={best:.6f} (pos={auc_pos:.6f}, neg={auc_neg:.6f}) | path={path}"
        )
        print(f"[feature-guard] stats y=0: {stats0}")
        print(f"[feature-guard] stats y=1: {stats1}")
        print("[feature-guard] example rows:")
        print(ex.to_string(index=False))

        findings.append(
            {
                "column": col,
                "auc_pos": float(auc_pos),
                "auc_neg": float(auc_neg),
                "auc_best": float(best),
            }
        )

    return findings
