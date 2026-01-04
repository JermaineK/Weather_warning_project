#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_and_maps_manager.py

One-stop orchestrator to bundle maps + summary + sanity checks
into a per-run folder like:

  results/reports/20251122_run001/

It wires together:
  - report_make_maps.py
  - plot_seed_map_cartopy.py
  - plot_seeds_with_ibtracs.py
  - plot_seed_track_map_cartopy.py
  - report_generate_summary.py
  - report_sanity_checks.py

Typical usage
-------------
python reports_and_maps_manager.py \
  --run-name coral_sea_demo \
  --union-csv   results/seedmaps/coral_sea_demo_union_byhour.csv \
  --patches-csv results/seedmaps/coral_sea_demo_seed_patches.csv \
  --matches-csv results/seedmaps/coral_sea_demo_seed_track_matches.csv \
  --seed-summary  results/seedmaps/coral_sea_demo_seed_summary.txt \
  --seed-analysis results/seedmaps/coral_sea_demo_seed_analysis.txt \
  --alerts-dir    results/alerts \
  --conversion-csv results/seedmaps/coral_sea_demo_conversion_rates.csv \
  --viability-thresholds results/sweeps/viability_best_thresholds.csv \
  --ibtracs data/tracks/tracks_subset.csv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.run_naming import make_run_dir
from utils import config_normalize
from pipeline_contracts import order_steps, preflight_step, postflight_step

HERE = HERE.parent


def _ensure_venv() -> None:
    venv_bin = "Scripts" if os.name == "nt" else "bin"
    venv_exe = "python.exe" if os.name == "nt" else "python"
    venv_py = (REPO_ROOT / ".venv" / venv_bin / venv_exe)
    if not venv_py.exists():
        return
    if os.environ.get("PIPELINE_ALLOW_SYSTEM_PYTHON", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        if venv_py.resolve() != Path(sys.executable).resolve():
            raise SystemExit(
                "[python] refusing to run reports with system interpreter while .venv exists "
                "(set PIPELINE_ALLOW_SYSTEM_PYTHON=1 to override)."
            )
    except Exception:
        pass


def _rewrite_flag_values(argv: List[str], flags: set[str]) -> List[str]:
    """
    Rewrite `--flag value` to `--flag=value` so values like '-180..180' are
    not parsed as new options by argparse.
    """
    out: List[str] = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok in flags and i + 1 < len(argv):
            nxt = str(argv[i + 1])
            if not nxt.startswith("--"):
                out.append(f"{tok}={nxt.strip()}")
                skip = True
                continue
        if any(tok.startswith(f"{f}=") for f in flags):
            lhs, rhs = tok.split("=", 1)
            out.append(f"{lhs}={rhs.strip()}")
            continue
        out.append(tok)
    return out


REWRITE_FLAGS = {
    "--ibtracs-normalize-lon",
    "--slowtick-normalize-lon",
    "--normalize-lon",
    "--ibtracs-area",
    "--slowtick-area",
    "--area",
}


def build_cmd(script: Path, args: List[str]) -> List[str]:
    return [sys.executable, str(script), *args]


def run_step(tag: str, script: Path, args: List[str]) -> Tuple[bool, int]:
    """
    Run a single subprocess step; return (ok, returncode).
    """
    args = _rewrite_flag_values(args, REWRITE_FLAGS)
    cmd = build_cmd(script, args)
    print(f"\n[manager] STEP {tag}:")
    print("  $ " + " ".join(str(x) for x in cmd))
    res = subprocess.run(cmd)
    ok = (res.returncode == 0)
    if not ok:
        print(f"[manager] STEP {tag} FAILED with code {res.returncode}")
    return ok, res.returncode


def _parse_leads(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()] if spec else []
    leads = []
    for p in parts:
        try:
            leads.append(int(p))
        except Exception:
            continue
    return leads


def _read_yaml_or_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(raw) or {}
    except Exception:
        try:
            return json.loads(raw) if raw.strip() else {}
        except Exception:
            return {}


def _count_csv_rows(path: Path) -> int:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    try:
        row_idx = -1
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for row_idx, _ in enumerate(fh):
                pass
        return max(0, row_idx)
    except Exception:
        return 0


def _parquet_row_counts(path: Path) -> tuple[int | None, int | None]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None, None
    try:
        pf = pq.ParquetFile(path)
    except Exception:
        return None, None
    meta = pf.metadata
    meta_rows = int(meta.num_rows) if meta is not None else None
    counted = None
    if meta is not None:
        try:
            counted = int(sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups)))
        except Exception:
            counted = None
    if not meta_rows:
        try:
            counted = sum(len(b) for b in pf.iter_batches(batch_size=200_000, columns=[]))
        except Exception:
            counted = counted if counted is not None else 0
    if counted is None:
        counted = meta_rows if meta_rows is not None else 0
    return meta_rows, counted


def _row_counts(path: Path) -> tuple[int | None, int | None]:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()
    if ext == ".parquet":
        return _parquet_row_counts(path)
    if ext in {".csv", ".csv.gz"}:
        return None, _count_csv_rows(path)
    try:
        return None, 1 if path.exists() and path.stat().st_size > 0 else 0
    except Exception:
        return None, 0


def _row_count(path: Path) -> int:
    _, counted = _row_counts(path)
    return int(counted or 0)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _file_signature(path: Path) -> Dict[str, Any]:
    sig: Dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
    if not path.exists():
        return sig
    try:
        sig["size_mb"] = round(path.stat().st_size / 1e6, 2)
        sig["mtime"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        pass
    rows_meta, rows_counted = _row_counts(path)
    if rows_meta is not None:
        sig["rows_metadata"] = int(rows_meta)
    if rows_counted is not None:
        sig["rows_counted"] = int(rows_counted)
    sig["sha256"] = _sha256_file(path)
    return sig


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip() or None
    except Exception:
        return None
    return None


def _config_sha256(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_provenance(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _provenance_matches(run_dir: Path, config_sha: Optional[str], out_path: Optional[Path]) -> bool:
    if not config_sha or not out_path:
        if not config_sha:
            return False
        prov = _load_provenance(run_dir / "provenance.json")
        return prov.get("config_sha256") == config_sha
    prov = _load_provenance(run_dir / "provenance.json")
    if prov.get("config_sha256") != config_sha:
        return False
    for art in prov.get("artifacts", []):
        if art.get("path") == str(out_path):
            return True
    return False


def _collect_files(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        if not p:
            continue
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            out.append(p)
    return out


def _collect_alert_files(alerts_dir: Path, run_name: str) -> List[Path]:
    if not alerts_dir.exists():
        return []
    patterns = [
        f"alerts_{run_name}_*.csv",
        f"alerts_{run_name}_*.csv.gz",
        f"alerts_{run_name}_*.parquet",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(alerts_dir.glob(pat))
    return [p for p in files if p.is_file()]


def _collect_metrics_files(run_name: str) -> List[Path]:
    files: List[Path] = []
    for base in [Path("results/metrics"), Path("results/eval"), Path("results/per_hour")]:
        if not base.exists():
            continue
        files.extend([p for p in base.glob(f"*{run_name}*") if p.is_file()])
    return files


def _write_provenance(
    run_dir: Path,
    run_name: str,
    config_sha: Optional[str],
    source_inputs: List[Dict[str, Any]],
    artifacts: List[Path],
    git_commit: Optional[str],
) -> Path:
    rows = []
    for p in _collect_files(artifacts):
        rows.append(
            {
                **_file_signature(p),
                "run_id": run_name,
                "git_commit": git_commit,
                "config_sha256": config_sha,
                "source_inputs": source_inputs,
            }
        )
    payload = {
        "run_id": run_name,
        "git_commit": git_commit,
        "config_sha256": config_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_inputs": source_inputs,
        "artifacts": rows,
    }
    out_path = run_dir / "provenance.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


def _flatten_kv(prefix: str, obj: Any) -> List[str]:
    """
    Turn nested dicts into CLI flags. Mirrors run_pipeline for parity.
    """
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v is None:
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_kv(key, v))
        return out
    if isinstance(obj, bool):
        if obj:
            out.append(f"--{prefix.replace('_','-')}")
        return out
    if isinstance(obj, (list, tuple)):
        if not obj:
            return out
        joined = ",".join(map(str, obj))
        out += [f"--{prefix.replace('_','-')}", joined]
        return out
    key_leaf = prefix.split(".")[-1]
    if isinstance(obj, str):
        val = obj.strip()
        if val == "":
            return out
        if key_leaf in {"normalize_lon", "normalize-lon", "area"}:
            flag = f"--{prefix.replace('_','-')}"
            out.append(f"{flag}={val}")
            return out
        out += [f"--{prefix.replace('_','-')}", val]
        return out
    out += [f"--{prefix.replace('_','-')}", str(obj)]
    return out


def _write_blocked(path: Path, blocked: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"blocked": blocked}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _step_label(section: str, mode: str) -> str:
    return f"{section}.{mode}"


def _preflight_inputs(
    section: str,
    steps: List[Dict[str, Any]],
    enabled_ids: set[str],
    min_rows: int,
) -> List[Dict[str, str]]:
    blocked: List[Dict[str, str]] = []
    for step in steps:
        mode = str(step.get("mode", "")).strip() or "unknown"
        label = _step_label(section, mode)
        pref = preflight_step(section, step)
        for issue in pref.input_issues:
            produced = list(issue.spec.produced_by or ())
            upstream_enabled = any(p in enabled_ids for p in produced)
            if upstream_enabled and issue.reason in {"missing_file", "missing_glob", "empty_file", "empty_glob"}:
                continue
            if upstream_enabled and issue.reason == "missing_columns":
                continue
            blocked.append(
                {
                    "step": label,
                    "reason": issue.reason,
                    "detail": issue.detail,
                    "path": str(issue.path) if issue.path else "",
                    "produced_by": ",".join(produced),
                }
            )

        # Row-count guard for required file inputs that are not produced by enabled upstream steps.
        for spec in pref.contract.inputs:
            if spec.kind != "file" or not spec.required:
                continue
            path, _ = spec.resolve(step)
            if not path or not path.exists():
                continue
            if spec.produced_by and any(p in enabled_ids for p in spec.produced_by):
                continue
            rows = _row_count(path)
            if rows < max(1, int(min_rows)):
                blocked.append(
                    {
                        "step": label,
                        "reason": "min_rows",
                        "detail": f"rows={rows} < min_rows={min_rows}",
                        "path": str(path),
                        "produced_by": ",".join(spec.produced_by or ()),
                    }
                )
    return blocked


def _check_required_file(path: str | Path, label: str, min_rows: int) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return [
            {
                "step": label,
                "reason": "missing_file",
                "detail": "required input missing",
                "path": str(p),
                "produced_by": "",
            }
        ]
    rows = _row_count(p)
    if rows < max(1, int(min_rows)):
        return [
            {
                "step": label,
                "reason": "min_rows",
                "detail": f"rows={rows} < min_rows={min_rows}",
                "path": str(p),
                "produced_by": "",
            }
        ]
    return []


def _run_eval_from_config(
    cfg: Dict[str, Any],
    eval_mgr: Path,
    run_name: str,
    min_rows: int,
    blocked: List[Dict[str, str]],
    run_dir: Path,
    config_sha: Optional[str],
) -> None:
    sec = cfg.get("eval", {}) if isinstance(cfg.get("eval"), dict) else {}
    if not sec.get("enabled"):
        return
    raw_steps = [s for s in sec.get("steps", []) if isinstance(s, dict)]
    if not raw_steps:
        return

    # Normalize + order
    raw_steps, _ = config_normalize.normalize_config(raw_steps)
    try:
        steps = order_steps("eval", raw_steps)
    except SystemExit as exc:
        blocked.append(
            {
                "step": "eval",
                "reason": "dependency",
                "detail": str(exc),
                "path": "",
                "produced_by": "",
            }
        )
        steps = raw_steps

    for step in steps:
        if step.get("enabled") is False:
            continue
        mode = str(step.get("mode", "hourly-rollup"))
        label = _step_label("eval", mode)

        pref = preflight_step("eval", step)
        if pref.errors:
            for err in pref.errors:
                blocked.append(
                    {
                        "step": label,
                        "reason": "preflight",
                        "detail": err,
                        "path": "",
                        "produced_by": "",
                    }
                )
            continue

        # Skip if output already exists and passes validation (only when provenance matches).
        if _provenance_matches(run_dir, config_sha, None):
            try:
                postflight_step("eval", step, pref.expected_output_columns)
                print(f"[eval] skip (exists, schema ok): {mode}")
                continue
            except SystemExit:
                pass

        # Run eval step
        args = _flatten_kv(
            "",
            {k: v for k, v in step.items() if k not in ("mode", "enabled", "skip_if_exists")},
        )
        if run_name and "--run-name" not in args and "--run_name" not in args:
            args = ["--run-name", run_name, *args]
        ok, code = run_step(f"eval.{mode}", eval_mgr, [mode, *args])
        if not ok:
            blocked.append(
                {
                    "step": label,
                    "reason": "exec_failed",
                    "detail": f"eval step failed with code {code}",
                    "path": "",
                    "produced_by": "",
                }
            )
            continue
        try:
            postflight_step("eval", step, pref.expected_output_columns)
        except SystemExit as exc:
            blocked.append(
                {
                    "step": label,
                    "reason": "postflight",
                    "detail": str(exc),
                    "path": "",
                    "produced_by": "",
                }
            )


# --------------------- main orchestration ---------------------


def main() -> int:
    _ensure_venv()
    # Support mode-first calls (e.g., "summary") as used by run_pipeline.
    argv = sys.argv[1:]
    mode_first = None
    if argv and not argv[0].startswith("--"):
        mode_first = argv[0]
        argv = argv[1:]
    argv = _rewrite_flag_values(argv, REWRITE_FLAGS)

    # Convenience dispatch for summary-only calls so pipeline can do:
    #   reports_and_maps_manager.py summary --out-md ...
    if mode_first == "summary":
        ignore_flags = {"--chunk-rows", "--chunksize", "--parquet-rows"}
        args_iter = iter(argv)
        passthrough: List[str] = []
        run_name = "auto"
        out_dir = Path("results/reports")
        out_md = None
        for tok in args_iter:
            if tok in ignore_flags:
                # skip value if present
                nxt = next(args_iter, None)
                continue
            if tok == "--run-name":
                run_name = next(args_iter, run_name)
                continue
            if tok == "--out-dir":
                out_dir = Path(next(args_iter, str(out_dir)))
                continue
            if tok == "--out-md":
                out_md = Path(next(args_iter, "results/reports/summary_auto.md"))
                continue
            passthrough.append(tok)
        if "--run-name" not in passthrough:
            passthrough = ["--run-name", run_name] + passthrough
        if "--out-dir" not in passthrough:
            passthrough = ["--out-dir", str(out_dir)] + passthrough
        script = HERE / "report_generate_summary.py"
        ok, code = run_step("summary", script, passthrough)
        if ok and out_md:
            safe = Path(out_dir) / f"{run_name.strip().replace(' ', '_')}_report.txt"
            try:
                out_md.parent.mkdir(parents=True, exist_ok=True)
                out_md.write_text(safe.read_text())
                print(f"[manager] copied summary to {out_md}")
            except Exception as e:
                print(f"[manager] warning: could not copy summary to {out_md}: {e}")
        return code
    if mode_first in {"objects-by-hour", "object-matches", "object-maps", "report-pack", "reporting-v2"}:
        mode_map = {
            "objects-by-hour": HERE / "objects_by_hour.py",
            "object-matches": HERE / "match_objects_to_tracks.py",
            "object-maps": HERE / "plot_object_matches.py",
            "report-pack": HERE / "report_pack.py",
            "reporting-v2": HERE / "reporting_v2.py",
        }
        script = mode_map[mode_first]
        ok, code = run_step(mode_first, script, argv)
        return code

    ap = argparse.ArgumentParser(
        description="Bundle per-run maps + summary + sanity checks into a dated run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core run identity
    ap.add_argument(
        "--run-name",
        default="auto",
        help="Human-readable name/tag for this run (used in summary filename).",
    )
    ap.add_argument(
        "--run-root",
        default="results/reports",
        help="Root folder under which per-run folders YYYYMMDD_runNNN are created.",
    )
    ap.add_argument(
        "--run-date",
        default=None,
        help="Date prefix for run folder (YYYYMMDD). Default: today (UTC).",
    )

    # Key inputs shared across reports
    ap.add_argument(
    "--union-csv",
        default=None,
        help="Seed union-by-hour CSV for maps (default: results/seedmaps/<run>_union_byhour.csv).",
    )
    ap.add_argument(
        "--patches-csv",
        default=None,
        help="Seed patches CSV (centroids) for quick QA maps (default: results/seedmaps/<run>_seed_patches.csv).",
    )
    ap.add_argument(
        "--matches-csv",
        default=None,
        help="Seed-track matches CSV for overlays (default: results/seedmaps/<run>_seed_track_matches.csv).",
    )
    ap.add_argument(
        "--seed-summary",
        default=None,
        help="Seed-level summary text produced earlier in the pipeline (default: results/seedmaps/<run>_seed_summary.txt).",
    )
    ap.add_argument(
        "--seed-analysis",
        default=None,
        help="Optional seed analysis text report (e.g. from analyze_seeds.py).",
    )
    ap.add_argument(
        "--alerts-dir",
        default="results/alerts",
        help="Alerts directory for snapshot + sanity checks.",
    )
    ap.add_argument(
        "--conversion-csv",
        default=None,
        help="Optional conversion/proto-outcomes CSV to embed in summary (default: results/seedmaps/<run>_conversion_rates.csv if present).",
    )
    ap.add_argument(
        "--viability-thresholds",
        default=None,
        help="Optional viability thresholds CSV (default: results/sweeps/viability_best_thresholds.csv).",
    )
    ap.add_argument(
        "--storm-timeseries",
        default="data/storm_timeseries_panel.parquet",
        help="Optional storm-centric time-series panel to surface in the summary.",
    )

    # IBTrACS / storm overlays
    ap.add_argument(
        "--ibtracs",
        default=None,
        help="IBTrACS CSV for overlays (if omitted, IBTrACS maps are skipped).",
    )
    ap.add_argument(
        "--ibtracs-area",
        default=None,
        help='Optional AOI for IBTrACS overlays: "latN,lonW,latS,lonE".',
    )
    ap.add_argument(
        "--ibtracs-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Longitude frame to use for IBTrACS overlays.",
    )

    # Behaviour toggles
    ap.add_argument(
        "--skip-quick-maps",
        action="store_true",
        help="Skip report_make_maps quick QA maps.",
    )
    ap.add_argument(
        "--skip-cartopy-seed-map",
        action="store_true",
        help="Skip plot_seed_map_cartopy (cartopy seed map).",
    )
    ap.add_argument(
        "--skip-ibtracs-maps",
        action="store_true",
        help="Skip IBTrACS-related plots (even if --ibtracs is given).",
    )
    ap.add_argument(
        "--union-seeds-per-hour",
        action="store_true",
        help="Emit per-hour frames for the union seed map.",
    )
    ap.add_argument(
        "--seeds-ibtracs-per-hour",
        action="store_true",
        help="Emit per-hour frames for seeds-with-IBTrACS maps.",
    )
    ap.add_argument(
        "--seed-track-per-hour",
        action="store_true",
        help="Emit per-hour frames for seed-track match maps.",
    )
    ap.add_argument(
        "--seed-track-per-storm-gifs",
        action="store_true",
        help="Emit per-storm hourly GIFs for seed-track match maps.",
    )
    ap.add_argument("--seed-track-storm-id-col", default=None, help="Storm id column override for seed-track maps.")
    ap.add_argument("--seed-track-direction-col", default=None, help="Bearing column for direction arrows.")
    ap.add_argument("--seed-track-direction-scale", type=float, default=0.6, help="Arrow length in degrees.")
    ap.add_argument("--seed-track-direction-color", default="tab:green", help="Arrow color for direction overlay.")
    ap.add_argument("--seed-track-max-direction-arrows", type=int, default=0, help="Cap direction arrows per frame.")
    ap.add_argument(
        "--per-hour-step",
        type=int,
        default=2,
        help="Step between per-hour frames (2 = every 2nd hour).",
    )
    ap.add_argument(
        "--per-hour-max-frames",
        type=int,
        default=150,
        help="Limit per-hour frame count (0 disables).",
    )
    ap.add_argument(
        "--animate",
        action="store_true",
        help="Stitch per-hour frames into an animation.",
    )
    ap.add_argument(
        "--animate-format",
        choices=["gif", "mp4"],
        default="gif",
        help="Animation output format.",
    )
    ap.add_argument(
        "--animate-fps",
        type=float,
        default=6.0,
        help="Frames per second for animations.",
    )
    ap.add_argument(
        "--animate-loop",
        type=int,
        default=0,
        help="GIF loop count (0 = infinite).",
    )
    ap.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip final sanity checks step.",
    )
    ap.add_argument(
        "--skip-objects",
        action="store_true",
        help="Skip object extraction (objects_by_hour).",
    )
    ap.add_argument(
        "--skip-object-matches",
        action="store_true",
        help="Skip object-to-track matching.",
    )
    ap.add_argument(
        "--skip-object-maps",
        action="store_true",
        help="Skip object-based per-storm maps.",
    )
    ap.add_argument(
        "--skip-report-pack",
        action="store_true",
        help="Skip report-pack tables (health/feature stats/associations).",
    )
    ap.add_argument(
        "--skip-reporting-v2",
        action="store_true",
        help="Skip reporting_v2 (Markdown/JSON report + storm pages).",
    )

    # Slow-tick diagnostics (optional)
    ap.add_argument(
        "--run-slowtick",
        action="store_true",
        default=True,
        help="Run slowtick_diagnostics.py on alerts (diagnostic only; default on).",
    )
    ap.add_argument(
        "--skip-slowtick",
        action="store_true",
        help="Skip slowtick diagnostics (overrides --run-slowtick).",
    )
    ap.add_argument(
        "--slowtick-alerts-dir",
        default=None,
        help="Alerts directory for slow-tick diagnostics (defaults to --alerts-dir).",
    )
    ap.add_argument(
        "--slowtick-run-name",
        default=None,
        help="Run name used in alerts filenames (defaults to --run-name).",
    )
    ap.add_argument(
        "--slowtick-leads",
        default="24,48,72,120,240",
        help="Comma-separated lead hours to include in diagnostics.",
    )
    ap.add_argument(
        "--slowtick-flag-col",
        default="alert_final",
        help="Flag column to use (tries fallbacks if missing).",
    )
    ap.add_argument(
        "--slowtick-thresholds",
        default=None,
        help="Optional thresholds table for per-lead slowtick flags.",
    )
    ap.add_argument(
        "--slowtick-threshold-col",
        default="thr_Fbeta",
        help="Threshold column to use in --slowtick-thresholds.",
    )
    ap.add_argument(
        "--slowtick-prob-col",
        default="prob_viable",
        help="Probability column for derived per-lead flags.",
    )
    ap.add_argument(
        "--slowtick-out-subdir",
        default="slowtick",
        help="Subdirectory under the run folder for slow-tick outputs.",
    )
    ap.add_argument(
        "--slowtick-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Longitude normalization for diagnostics input.",
    )
    ap.add_argument(
        "--slowtick-area",
        default=None,
        help='Optional AOI "latN,lonW,latS,lonE" for diagnostics.',
    )
    ap.add_argument(
        "--slowtick-time-format",
        default=None,
        help="Optional strptime for parsing alert times.",
    )
    ap.add_argument(
        "--slowtick-prefer",
        choices=["throttled", "denoised", "base"],
        default="throttled",
        help="Preferred alerts stage if multiple files exist.",
    )
    ap.add_argument(
        "--slowtick-save-timeseries",
        action="store_true",
        help="Save hourly global coverage time series.",
    )
    ap.add_argument(
        "--slowtick-save-hemi-timeseries",
        action="store_true",
        help="Save hourly N/S coverage time series.",
    )
    ap.add_argument(
        "--slowtick-bootstrap-B",
        type=int,
        default=500,
        help="Bootstrap reps for CIs (knee/parity).",
    )
    ap.add_argument(
        "--slowtick-min-hours-per-lead",
        type=int,
        default=8,
        help="Minimum hourly points required per lead.",
    )
    ap.add_argument(
        "--slowtick-fft-gap-fill",
        type=int,
        default=2,
        help="Fill NaN gaps up to this length before FFT (hours).",
    )
    ap.add_argument(
        "--slowtick-cache-fallback",
        action="store_true",
        help="Cache fallback alerts in memory for slowtick reuse.",
    )
    ap.add_argument(
        "--slowtick-debug",
        action="store_true",
        help="Verbose file/range debug for diagnostics.",
    )
    # Post-run diagnostics (read-only)
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run run_diagnostics.py on the completed run folder.",
    )
    ap.add_argument("--diagnostics-out", default=None, help="Override diagnostics output directory.")
    ap.add_argument("--diagnostics-train-table", default=None, help="Optional training table for diagnostics.")
    ap.add_argument("--diagnostics-train-keys", default=None, help="Optional train keys table for overlap checks.")
    ap.add_argument("--diagnostics-val-keys", default=None, help="Optional val keys table for overlap checks.")
    ap.add_argument("--diagnostics-metrics-json", default=None, help="Optional metrics JSON with feature list.")
    ap.add_argument("--diagnostics-feature-meta", default=None, help="Optional feature metadata JSON for causality checks.")
    ap.add_argument(
        "--diagnostics-fail-on-checks",
        action="store_true",
        help="Exit non-zero if diagnostics finds any failures.",
    )
    # Object-based reporting inputs/outputs
    ap.add_argument("--objects-in", default=None, help="Alerts/predictions table for object extraction.")
    ap.add_argument("--objects-out", default="results/objects/objects_by_hour.parquet", help="Object-by-hour output table.")
    ap.add_argument("--cells-out", default=None, help="Optional per-cell table with object_id.")
    ap.add_argument("--objects-mask-col", default="alert_final", help="Flag column for object candidates.")
    ap.add_argument("--objects-score-col", default="prob_viable", help="Score column for object candidates.")
    ap.add_argument("--objects-threshold", type=float, default=None, help="Score threshold for object candidates.")
    ap.add_argument("--objects-top-k", type=int, default=20, help="Top-K candidates per storm-hour (after de-dup).")
    ap.add_argument("--objects-min-sep-km", type=float, default=75.0, help="Minimum separation for candidate de-dup.")
    ap.add_argument("--objects-adaptive-quantile", type=float, default=0.995, help="Per-hour score quantile for gating.")
    ap.add_argument("--objects-adaptive-base-threshold", type=float, default=None, help="Base score threshold for gating.")
    ap.add_argument("--objects-min-area-cells", type=int, default=5, help="Minimum object area (cells) before filtering.")
    ap.add_argument("--objects-persist-hours-small", type=int, default=3, help="Keep small objects only if they persist this many hours.")
    ap.add_argument("--objects-persist-link-km", type=float, default=75.0, help="Link radius for persistence tracking (km).")
    ap.add_argument(
        "--objects-morphology",
        choices=["none", "majority"],
        default="none",
        help="Optional morphology smoothing on candidate mask.",
    )
    ap.add_argument("--objects-morph-k", type=int, default=3, help="Neighbor threshold for majority smoothing.")
    ap.add_argument("--objects-rejects-out", default=None, help="Optional per-hour rejected objects summary output.")
    ap.add_argument("--objects-match-top-n", type=int, default=1, help="Top-N matches to highlight per storm-hour.")
    ap.add_argument("--object-matches-out", default="results/matches/storm_object_matches.parquet", help="Output matches table.")
    ap.add_argument("--objects-with-motion-out", default=None, help="Optional objects table with motion columns.")
    ap.add_argument("--tracks-with-motion-out", default=None, help="Optional tracks table with motion columns.")
    ap.add_argument("--object-maps-dir", default=None, help="Output dir for per-storm object maps.")
    ap.add_argument("--object-maps-per-hour", action="store_true", help="Emit one map per hour in the window.")
    ap.add_argument("--object-maps-arrows", action="store_true", help="Overlay motion direction arrows.")
    ap.add_argument("--object-maps-flow-arrows", action="store_true", help="Overlay flow-direction arrows.")
    ap.add_argument("--object-hours-before", type=float, default=72.0, help="Hours before genesis for maps.")
    ap.add_argument("--object-hours-after", type=float, default=24.0, help="Hours after genesis for maps.")
    ap.add_argument("--report-pack-config", default=None, help="Optional pipeline YAML for run_health checks.")
    ap.add_argument("--report-pack-out-dir", default=None, help="Output dir for report-pack tables.")
    ap.add_argument("--pipeline-config", default=None, help="Optional pipeline YAML for preflight/eval steps.")
    ap.add_argument("--skip-preflight", action="store_true", help="Skip report preflight checks.")
    ap.add_argument("--preflight-min-rows", type=int, default=1, help="Minimum rows required for report inputs.")
    ap.add_argument("--skip-eval", action="store_true", help="Skip eval steps from the pipeline config.")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, stop on first failing step and exit with its code.",
    )
    # Pipeline compatibility: accept chunking hints even though manager doesn't use them.
    ap.add_argument("--chunk-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")

    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    run_dir = make_run_dir(run_root, args.run_date)
    maps_dir = run_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = maps_dir / "frames"

    print(f"[manager] Run folder: {run_dir}")

    def require_file(path: str | Path, label: str) -> Path:
        p = Path(path)
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            raise SystemExit(f"[manager] missing required {label}: {p}")
        return p

    def or_default(path_arg: str | None, pattern: str | None, fallback: str | None = None) -> str:
        if path_arg:
            return path_arg
        if pattern and args.run_name:
            return pattern.format(run=args.run_name)
        return fallback or ""

    def run_animation(tag: str, frame_glob: str, out_path: Path) -> Tuple[bool, int]:
        # Agent: centralize animation stitching for per-hour maps.
        script = HERE / "animate_frames.py"
        step_args = [
            "--frames", frame_glob,
            "--out", str(out_path),
            "--fps", str(args.animate_fps),
        ]
        if out_path.suffix.lower() == ".gif":
            step_args += ["--loop", str(args.animate_loop)]
        if args.per_hour_max_frames and args.per_hour_max_frames > 0:
            step_args += ["--max-frames", str(args.per_hour_max_frames)]
        return run_step(tag, script, step_args)

    def _append_seed_track_options(step_args: List[str]) -> None:
        if args.seed_track_storm_id_col:
            step_args += ["--storm-id-col", args.seed_track_storm_id_col]
        if args.seed_track_direction_col:
            step_args += ["--direction-col", args.seed_track_direction_col]
        if args.seed_track_direction_scale:
            step_args += ["--direction-scale", str(args.seed_track_direction_scale)]
        if args.seed_track_direction_color:
            step_args += ["--direction-color", args.seed_track_direction_color]
        if args.seed_track_max_direction_arrows:
            step_args += ["--max-direction-arrows", str(args.seed_track_max_direction_arrows)]

    union_csv = or_default(args.union_csv, "results/seedmaps/{run}_union_byhour.csv")
    patches_csv = or_default(args.patches_csv, "results/seedmaps/{run}_seed_patches.csv")
    matches_csv = or_default(args.matches_csv, "results/seedmaps/{run}_seed_track_matches.csv")
    seed_summary = or_default(args.seed_summary, "results/seedmaps/{run}_seed_summary.txt")
    seed_analysis = or_default(args.seed_analysis, "results/seedmaps/{run}_seed_analysis.txt", "")
    conversion_csv = or_default(args.conversion_csv, "results/seedmaps/{run}_conversion_rates.csv", "")
    viability_thr = args.viability_thresholds or "results/sweeps/viability_best_thresholds.csv"
    ibtracs_default = "data/tracks/tracks_subset.csv"
    ibtracs_path = args.ibtracs or (ibtracs_default if Path(ibtracs_default).exists() else None)

    objects_in = args.objects_in
    if not objects_in:
        cand = [
            f"results/alerts/alerts_{args.run_name}_final.parquet",
            f"results/alerts/alerts_{args.run_name}_final.csv.gz",
            "results/predictions_base_specialist.parquet",
        ]
        for c in cand:
            if Path(c).exists():
                objects_in = c
                break
        if not objects_in:
            objects_in = cand[0]
    object_maps_dir = Path(args.object_maps_dir) if args.object_maps_dir else (maps_dir / "objects")
    objects_rejects_out = args.objects_rejects_out
    if objects_rejects_out is None:
        objects_rejects_out = str(Path(args.objects_out).with_name("objects_rejects_by_hour.parquet"))
    safe_run = args.run_name.strip().replace(" ", "_")
    report_pack_out = Path(args.report_pack_out_dir) if args.report_pack_out_dir else (run_dir / f"{safe_run}_tables")
    report_pack_config = args.report_pack_config
    if report_pack_config is None:
        cfg_candidate = Path("config/pipeline.yaml")
        if cfg_candidate.exists():
            report_pack_config = str(cfg_candidate)

    pipeline_cfg_path = Path(args.pipeline_config) if args.pipeline_config else None
    if pipeline_cfg_path is None and report_pack_config:
        pipeline_cfg_path = Path(report_pack_config)
    cfg_obj = _read_yaml_or_json(pipeline_cfg_path) if pipeline_cfg_path else {}
    if cfg_obj:
        cfg_obj, _ = config_normalize.normalize_config(cfg_obj)
    config_sha = _config_sha256(pipeline_cfg_path)

    blocked: List[Dict[str, str]] = []
    if cfg_obj and not args.skip_eval:
        eval_mgr = REPO_ROOT / "eval_subprocess" / "eval_manager.py"
        if eval_mgr.exists():
            _run_eval_from_config(
                cfg_obj,
                eval_mgr,
                args.run_name,
                args.preflight_min_rows,
                blocked,
                run_dir,
                config_sha,
            )

    # Agent: preflight report inputs so missing dependencies surface as BLOCKED.
    if not args.skip_preflight:
        report_steps: List[Dict[str, Any]] = []
        if not args.skip_objects:
            report_steps.append({"mode": "objects-by-hour", "infile": objects_in})
        if not args.skip_object_matches:
            report_steps.append({"mode": "object-matches", "objects": args.objects_out, "tracks": ibtracs_path})
        if not args.skip_object_maps:
            report_steps.append({"mode": "object-maps", "matches": args.object_matches_out, "tracks": ibtracs_path, "out_dir": str(object_maps_dir)})
        if not args.skip_report_pack:
            report_steps.append({"mode": "report-pack", "config": report_pack_config, "objects": args.objects_out, "matches": args.object_matches_out, "out_dir": str(report_pack_out)})
        if not args.skip_reporting_v2:
            report_steps.append({"mode": "reporting-v2", "tables_dir": str(report_pack_out), "matches": args.object_matches_out, "tracks": ibtracs_path, "out_dir": str(run_dir)})

        ordered_steps = report_steps
        try:
            ordered_steps = order_steps("report", report_steps)
        except SystemExit as exc:
            blocked.append(
                {
                    "step": "report",
                    "reason": "dependency",
                    "detail": str(exc),
                    "path": "",
                    "produced_by": "",
                }
            )
        enabled_ids = {f"report.{s.get('mode')}" for s in ordered_steps if s.get("mode")}
        blocked.extend(_preflight_inputs("report", ordered_steps, enabled_ids, args.preflight_min_rows))

        if not args.skip_quick_maps:
            blocked.extend(_check_required_file(union_csv, "report.quick-maps (union-csv)", args.preflight_min_rows))
            blocked.extend(_check_required_file(patches_csv, "report.quick-maps (patches-csv)", args.preflight_min_rows))
        if not args.skip_cartopy_seed_map:
            blocked.extend(_check_required_file(union_csv, "report.seed-map (union-csv)", args.preflight_min_rows))
        if (ibtracs_path is not None) and (not args.skip_ibtracs_maps):
            blocked.extend(_check_required_file(ibtracs_path, "report.ibtracs (tracks)", args.preflight_min_rows))
        if not args.skip_sanity:
            blocked.extend(_check_required_file(seed_summary, "report.summary (seed-summary)", args.preflight_min_rows))

        if blocked:
            for item in blocked:
                print(f"[manager] BLOCKED {item.get('step')}: {item.get('reason')} {item.get('detail')} ({item.get('path')})")
            blocked_path = report_pack_out / "blocked.json"
            _write_blocked(blocked_path, blocked)
            if not args.skip_reporting_v2:
                script = HERE / "reporting_v2.py"
                step_args = [
                    "--run-name", args.run_name,
                    "--out-dir", str(run_dir),
                    "--tables-dir", str(report_pack_out),
                    "--objects", args.objects_out,
                    "--matches", args.object_matches_out,
                    "--blocked", str(blocked_path),
                ]
                if ibtracs_path:
                    step_args += ["--tracks", ibtracs_path]
                if report_pack_config:
                    step_args += ["--config", report_pack_config]
                run_step("reporting-v2", script, step_args)
            return 2

    # Agent: object-based reporting to avoid seed-map blobs.
    # --- STEP 0: object extraction (per-hour components) ---
    if not args.skip_objects:
        script = HERE / "objects_by_hour.py"
        require_file(objects_in, "objects input")
        step_args = [
            "--infile", objects_in,
            "--objects-out", args.objects_out,
            "--mask-col", args.objects_mask_col,
            "--score-col", args.objects_score_col,
            "--connectivity", "8",
            "--min-neighbors", "1",
        ]
        step_args += [
            "--min-area-cells", str(args.objects_min_area_cells),
            "--persist-hours-small", str(args.objects_persist_hours_small),
            "--persist-link-km", str(args.objects_persist_link_km),
            "--morphology", str(args.objects_morphology),
            "--morph-k", str(args.objects_morph_k),
        ]
        if args.objects_threshold is not None:
            step_args += ["--threshold", str(args.objects_threshold)]
        if args.cells_out:
            step_args += ["--cells-out", args.cells_out]
        if objects_rejects_out:
            step_args += ["--rejects-out", objects_rejects_out]
        ok, code = run_step("objects-by-hour", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 0b: object-to-track matching ---
    if not args.skip_object_matches:
        if ibtracs_path is None:
            raise SystemExit("[manager] object matching requires --ibtracs.")
        require_file(args.objects_out, "objects-out")
        require_file(ibtracs_path, "ibtracs")
        script = HERE / "match_objects_to_tracks.py"
        step_args = [
            "--objects", args.objects_out,
            "--tracks", ibtracs_path,
            "--out", args.object_matches_out,
            "--normalize-lon", args.ibtracs_normalize_lon,
            "--top-n", str(args.objects_match_top_n),
            "--top-k", str(args.objects_top_k),
            "--min-sep-km", str(args.objects_min_sep_km),
        ]
        if args.objects_adaptive_quantile is not None:
            step_args += ["--adaptive-quantile", str(args.objects_adaptive_quantile)]
        if args.objects_adaptive_base_threshold is not None:
            step_args += ["--adaptive-base-threshold", str(args.objects_adaptive_base_threshold)]
        if args.objects_with_motion_out:
            step_args += ["--objects-out", args.objects_with_motion_out]
        if args.tracks_with_motion_out:
            step_args += ["--tracks-out", args.tracks_with_motion_out]
        ok, code = run_step("object-matches", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 0c: object-based per-storm maps ---
    if not args.skip_object_maps:
        if ibtracs_path is None:
            raise SystemExit("[manager] object maps require --ibtracs.")
        require_file(args.object_matches_out, "object matches")
        require_file(ibtracs_path, "ibtracs")
        script = HERE / "plot_object_matches.py"
        step_args = [
            "--matches", args.object_matches_out,
            "--tracks", ibtracs_path,
            "--out-dir", str(object_maps_dir),
            "--normalize-lon", args.ibtracs_normalize_lon,
            "--hours-before", str(args.object_hours_before),
            "--hours-after", str(args.object_hours_after),
            "--objects", args.objects_out,
            "--match-top-n", str(args.objects_match_top_n),
        ]
        if args.object_maps_per_hour:
            step_args.append("--per-hour")
        if args.object_maps_arrows:
            step_args.append("--show-arrows")
        if args.object_maps_flow_arrows:
            step_args.append("--show-flow-arrows")
        ok, code = run_step("object-maps", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 1: quick QA maps (simple scatter maps) ---
    if not args.skip_quick_maps:
        require_file(union_csv, "union-csv")
        require_file(patches_csv, "patches-csv")
        script = HERE / "report_make_maps.py"
        step_args = [
            "--run-name", args.run_name,
            "--union-csv", union_csv,
            "--patches-csv", patches_csv,
            "--out-dir", str(maps_dir / "quick"),
            "--union-value-col", "prob_max",
            "--min-prob", "0.9",
            "--color-by-time-band",
            "--top-quantile", "0.9",
        ]
        step_args += ["--max-points-per-hour", "2000"]
        if objects_in:
            step_args += ["--prob-path", objects_in]
        if args.objects_out:
            step_args += ["--objects-path", args.objects_out]
        ok, code = run_step("quick-maps", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 2: cartopy seed map from union CSV ---
    if not args.skip_cartopy_seed_map:
        require_file(union_csv, "union-csv")
        script = HERE / "plot_seed_map_cartopy.py"
        out_png = maps_dir / "seeds_union_cartopy.png"
        step_args = [
            "--seeds", union_csv,
            "--out-png", str(out_png),
            "--value-col", "prob_max",
            "--min-prob", "0.9",
            "--top-quantile", "0.9",
            "--max-points-per-hour", "2000",
            "--title", f"Seeds (union by hour) - {args.run_name}",
        ]
        if ibtracs_path:
            step_args += [
                "--tracks", ibtracs_path,
                "--storm-window-before-h", "240",
                "--storm-window-after-h", "72",
                "--storm-radius-deg", "5.0",
                "--per-storm",
            ]
        ok, code = run_step("seed-map-cartopy", script, step_args)
        if not ok and args.strict:
            return code
        if args.union_seeds_per_hour:
            frames_dir.mkdir(parents=True, exist_ok=True)
            out_png = frames_dir / "seeds_union_hourly.png"
            step_args = [
                "--seeds", union_csv,
                "--out-png", str(out_png),
                "--value-col", "prob_max",
                "--min-prob", "0.9",
                "--top-quantile", "0.9",
                "--max-points-per-hour", "2000",
                "--title", f"Seeds (union by hour) - {args.run_name}",
                "--per-hour",
                "--time-col", "time",
                "--hour-step", str(args.per_hour_step),
                "--max-frames", str(args.per_hour_max_frames),
            ]
            ok, code = run_step("seed-map-cartopy-hourly", script, step_args)
            if not ok and args.strict:
                return code
            if ok and args.animate:
                anim_out = maps_dir / f"seeds_union_hourly.{args.animate_format}"
                frame_glob = str(frames_dir / "seeds_union_hourly_*.png")
                ok, code = run_animation("seed-map-cartopy-anim", frame_glob, anim_out)
                if not ok and args.strict:
                    return code

    # --- STEP 3: IBTrACS + seeds overlay ---
    if (ibtracs_path is not None) and (not args.skip_ibtracs_maps):
        script = HERE / "plot_seeds_with_ibtracs.py"
        out_png = maps_dir / "seeds_with_ibtracs.png"
        step_args = [
            "--ibtracs", ibtracs_path,
            "--out-png", str(out_png),
            "--max-points-per-hour", "2000",
            "--max-points-total", "20000",
        ]
        # prefer matches if present; otherwise union seeds
        matches_path = Path(matches_csv)
        if matches_path.exists():
            step_args += ["--matches", str(matches_path)]
        else:
            require_file(union_csv, "union-csv")
            step_args += ["--seeds", union_csv]

        if args.ibtracs_area:
            step_args += ["--area", args.ibtracs_area]
        if args.ibtracs_normalize_lon:
            step_args += ["--normalize-lon", args.ibtracs_normalize_lon]

        ok, code = run_step("seeds-with-ibtracs", script, step_args)
        if not ok and args.strict:
            return code

        # time-colored variant for temporal progression
        out_png = maps_dir / "seeds_with_ibtracs_time.png"
        step_args = [
            "--ibtracs", ibtracs_path,
            "--out-png", str(out_png),
            "--max-points-per-hour", "2000",
            "--max-points-total", "20000",
            "--color-by-time",
        ]
        # prefer matches if present; otherwise union seeds
        if matches_path.exists():
            step_args += ["--matches", str(matches_path)]
        else:
            step_args += ["--seeds", union_csv]
        if args.ibtracs_area:
            step_args += ["--area", args.ibtracs_area]
        if args.ibtracs_normalize_lon:
            step_args += ["--normalize-lon", args.ibtracs_normalize_lon]
        ok, code = run_step("seeds-with-ibtracs-time", script, step_args)
        if not ok and args.strict:
            return code

        if args.seeds_ibtracs_per_hour:
            frames_dir.mkdir(parents=True, exist_ok=True)
            out_png = frames_dir / "seeds_with_ibtracs_hourly.png"
            step_args = [
                "--ibtracs", ibtracs_path,
                "--out-png", str(out_png),
                "--max-points-per-hour", "2000",
                "--max-points-total", "20000",
                "--per-hour",
                "--hour-step", str(args.per_hour_step),
                "--max-frames", str(args.per_hour_max_frames),
            ]
            if matches_path.exists():
                step_args += ["--matches", str(matches_path)]
            else:
                step_args += ["--seeds", union_csv]
            if args.ibtracs_area:
                step_args += ["--area", args.ibtracs_area]
            if args.ibtracs_normalize_lon:
                step_args += ["--normalize-lon", args.ibtracs_normalize_lon]
            ok, code = run_step("seeds-with-ibtracs-hourly", script, step_args)
            if not ok and args.strict:
                return code
            if ok and args.animate:
                anim_out = maps_dir / f"seeds_with_ibtracs_hourly.{args.animate_format}"
                frame_glob = str(frames_dir / "seeds_with_ibtracs_hourly_*.png")
                ok, code = run_animation("seeds-with-ibtracs-anim", frame_glob, anim_out)
                if not ok and args.strict:
                    return code

    # --- STEP 4: seed-track match map (cartopy) ---
    if (not args.skip_ibtracs_maps) and Path(matches_csv).exists():
        require_file(matches_csv, "matches-csv")
        script = HERE / "plot_seed_track_map_cartopy.py"
        out_png = maps_dir / "seed_track_map.png"
        step_args = [
            "--matches", matches_csv,
            "--out", str(out_png),
            "--overlay-prob", "prob_max",
            "--min-prob", "0.5",
            "--max-points-per-hour", "2000",
            "--max-points-total", "20000",
        ]
        _append_seed_track_options(step_args)
        ok, code = run_step("seed-track-map", script, step_args)
        if not ok and args.strict:
            return code

        out_png = maps_dir / "seed_track_map_time.png"
        step_args = [
            "--matches", matches_csv,
            "--out", str(out_png),
            "--overlay-prob", "prob_max",
            "--min-prob", "0.5",
            "--max-points-per-hour", "2000",
            "--max-points-total", "20000",
            "--color-by-time",
        ]
        _append_seed_track_options(step_args)
        ok, code = run_step("seed-track-map-time", script, step_args)
        if not ok and args.strict:
            return code

        if args.seed_track_per_hour:
            frames_dir.mkdir(parents=True, exist_ok=True)
            out_png = frames_dir / "seed_track_map_hourly.png"
            step_args = [
                "--matches", matches_csv,
                "--out", str(out_png),
                "--overlay-prob", "prob_max",
                "--min-prob", "0.5",
                "--max-points-per-hour", "2000",
                "--max-points-total", "20000",
                "--per-hour",
                "--hour-step", str(args.per_hour_step),
                "--max-frames", str(args.per_hour_max_frames),
            ]
            _append_seed_track_options(step_args)
            ok, code = run_step("seed-track-map-hourly", script, step_args)
            if not ok and args.strict:
                return code
            if ok and args.animate:
                anim_out = maps_dir / f"seed_track_map_hourly.{args.animate_format}"
                frame_glob = str(frames_dir / "seed_track_map_hourly_*.png")
                ok, code = run_animation("seed-track-map-anim", frame_glob, anim_out)
                if not ok and args.strict:
                    return code

        if args.seed_track_per_storm_gifs:
            frames_dir.mkdir(parents=True, exist_ok=True)
            out_png = frames_dir / "seed_track_map_hourly_storm.png"
            step_args = [
                "--matches", matches_csv,
                "--out", str(out_png),
                "--overlay-prob", "prob_max",
                "--min-prob", "0.5",
                "--max-points-per-hour", "2000",
                "--max-points-total", "20000",
                "--per-hour",
                "--per-storm",
                "--hour-step", str(args.per_hour_step),
                "--max-frames", str(args.per_hour_max_frames),
            ]
            _append_seed_track_options(step_args)
            ok, code = run_step("seed-track-map-hourly-storm", script, step_args)
            if not ok and args.strict:
                return code
            if ok:
                storm_dir = maps_dir / "storm_gifs"
                storm_dir.mkdir(parents=True, exist_ok=True)
                frame_files = list(frames_dir.glob("seed_track_map_hourly_storm_*_*.png"))
                storm_ids = set()
                pattern = re.compile(r"seed_track_map_hourly_storm_(.+)_[0-9]{10}\\.png$")
                for fp in frame_files:
                    m = pattern.search(fp.name)
                    if m:
                        storm_ids.add(m.group(1))
                if not storm_ids:
                    print("[manager] seed-track per-storm GIFs: no frames found.")
                for sid in sorted(storm_ids):
                    anim_out = storm_dir / f"seed_track_map_hourly_storm_{sid}.{args.animate_format}"
                    frame_glob = str(frames_dir / f"seed_track_map_hourly_storm_{sid}_*.png")
                    ok, code = run_animation(f"seed-track-storm-{sid}", frame_glob, anim_out)
                    if not ok and args.strict:
                        return code

    # --- STEP 4b: per-storm hourly counts (heatmap) ---
    if Path(matches_csv).exists():
        script = HERE / "storm_hourly_counts.py"
        out_csv = report_pack_out / "seed_storm_hourly_counts.csv"
        out_png = maps_dir / "seed_storm_hourly_counts.png"
        step_args = [
            "--matches", matches_csv,
            "--out-csv", str(out_csv),
            "--out-png", str(out_png),
        ]
        ok, code = run_step("storm-hourly-counts", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 5: summary text report ---
    script = HERE / "report_generate_summary.py"
    step_args = [
        "--run-name", args.run_name,
        "--out-dir", str(run_dir),
        "--seed-summary", seed_summary,
        "--alerts-dir", args.alerts_dir,
        "--viability-thresholds", viability_thr,
        "--storm-timeseries", args.storm_timeseries,
    ]
    require_file(seed_summary, "seed-summary")
    if seed_analysis:
        step_args += ["--seed-analysis", seed_analysis]
    if conversion_csv:
        step_args += ["--include-conversion", conversion_csv]
    if ibtracs_path:
        step_args += ["--ibtracs", ibtracs_path]
    if args.ibtracs_area:
        step_args += ["--ibtracs-area", args.ibtracs_area]
    if args.ibtracs_normalize_lon:
        step_args += ["--ibtracs-normalize-lon", args.ibtracs_normalize_lon]

    ok, code = run_step("summary", script, step_args)
    if not ok and args.strict:
        return code

    # --- Optional: slow-tick diagnostics on alerts ---
    run_slowtick = bool(args.run_slowtick) and not args.skip_slowtick
    if run_slowtick:
        leads = _parse_leads(args.slowtick_leads)
        if not leads:
            print("[manager] slowtick: no valid leads parsed; skipping.")
        else:
            script = HERE / "slowtick_diagnostics.py"
            out_dir = Path(args.slowtick_out_subdir)
            if not out_dir.is_absolute():
                out_dir = run_dir / out_dir
            slowtick_alerts_dir = args.slowtick_alerts_dir or args.alerts_dir
            slowtick_run = args.slowtick_run_name or args.run_name
            step_args = [
                "--alerts-dir", slowtick_alerts_dir,
                "--run-name", slowtick_run,
                "--leads", *map(str, leads),
                "--flag-col", args.slowtick_flag_col,
                "--out-dir", str(out_dir),
                "--normalize-lon", args.slowtick_normalize_lon,
                "--prefer", args.slowtick_prefer,
                "--bootstrap-B", str(args.slowtick_bootstrap_B),
                "--min-hours-per-lead", str(args.slowtick_min_hours_per_lead),
                "--fft-gap-fill", str(args.slowtick_fft_gap_fill),
            ]
            slowtick_thr = args.slowtick_thresholds
            if slowtick_thr is None and viability_thr and Path(viability_thr).exists():
                slowtick_thr = viability_thr
            if slowtick_thr:
                step_args += [
                    "--thresholds", slowtick_thr,
                    "--threshold-col", args.slowtick_threshold_col,
                ]
            if args.slowtick_prob_col:
                step_args += ["--prob-col", args.slowtick_prob_col]
            if args.slowtick_cache_fallback:
                step_args.append("--cache-fallback")
            fallback_candidates = [
                f"alerts_{slowtick_run}_final.parquet",
                f"alerts_{slowtick_run}_thr.parquet",
                f"alerts_{slowtick_run}_base.parquet",
                f"alerts_{slowtick_run}_final.csv.gz",
                f"alerts_{slowtick_run}_thr.csv.gz",
                f"alerts_{slowtick_run}_base.csv.gz",
                f"alerts_{slowtick_run}_final.csv",
                f"alerts_{slowtick_run}_thr.csv",
                f"alerts_{slowtick_run}_base.csv",
            ]
            for name in fallback_candidates:
                candidate = Path(slowtick_alerts_dir) / name
                if candidate.exists():
                    step_args += ["--fallback-alerts", str(candidate)]
                    break
            if args.slowtick_area:
                step_args += ["--area", args.slowtick_area]
            if args.slowtick_time_format:
                step_args += ["--time-format", args.slowtick_time_format]
            # Always emit coverage_timeseries.csv so diagnostics can compare leads.
            step_args.append("--save-timeseries")
            if args.slowtick_save_hemi_timeseries:
                step_args.append("--save-hemi-timeseries")
            if args.slowtick_debug:
                step_args.append("--debug")
            ok, code = run_step("slowtick-diag", script, step_args)
            if not ok and args.strict:
                return code

    # --- STEP 6: sanity checks over key artifacts ---
    if not args.skip_sanity:
        script = HERE / "report_sanity_checks.py"

        # Files we *expect* after a healthy run
        # - core seed artifacts (original run)
        # - per-run summary
        must_exist = [
            seed_summary,
            patches_csv,
            matches_csv,
        ]
        # summary text lives in run_dir, file name driven by run-name logic
        safe_run = args.run_name.strip().replace(" ", "_")
        summary_txt = run_dir / f"{safe_run}_report.txt"
        must_exist.append(str(summary_txt))

        step_args = [
            "--must-exist",
            *must_exist,
            "--alerts-dir", args.alerts_dir,
            "--require-alerts",
            "--strict",  # internal strict w.r.t its own checks
        ]
        ok, code = run_step("sanity-checks", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 7: report-pack tables ---
    if not args.skip_report_pack:
        script = HERE / "report_pack.py"
        step_args = [
            "--run-name", args.run_name,
            "--out-dir", str(report_pack_out),
            "--objects", args.objects_out,
            "--matches", args.object_matches_out,
        ]
        if report_pack_config:
            step_args += ["--config", report_pack_config]
        if ibtracs_path:
            step_args += ["--tracks", ibtracs_path]
        ok, code = run_step("report-pack", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 8: reporting_v2 summary (MD/JSON + storm pages) ---
    if not args.skip_reporting_v2:
        script = HERE / "reporting_v2.py"
        step_args = [
            "--run-name", args.run_name,
            "--out-dir", str(run_dir),
            "--tables-dir", str(report_pack_out),
            "--objects", args.objects_out,
            "--matches", args.object_matches_out,
        ]
        step_args += [
            "--seed-summary", seed_summary,
            "--alerts-dir", args.alerts_dir,
            "--storm-timeseries", args.storm_timeseries,
            "--viability-thresholds", viability_thr,
            "--seed-union", union_csv,
            "--write-txt",
        ]
        if seed_analysis:
            step_args += ["--seed-analysis", seed_analysis]
        if conversion_csv:
            step_args += ["--conversion-csv", conversion_csv]
        blocked_path = report_pack_out / "blocked.json"
        if blocked_path.exists():
            step_args += ["--blocked", str(blocked_path)]
        if ibtracs_path:
            step_args += ["--tracks", ibtracs_path]
        if ibtracs_path:
            step_args += ["--ibtracs", ibtracs_path]
        if report_pack_config:
            step_args += ["--config", report_pack_config]
        if args.ibtracs_area:
            step_args += ["--ibtracs-area", args.ibtracs_area]
        if args.ibtracs_normalize_lon:
            step_args += [f"--ibtracs-normalize-lon={args.ibtracs_normalize_lon}"]
        ok, code = run_step("reporting-v2", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 9: diagnostics bundle (read-only) ---
    if args.diagnostics:
        script = HERE / "run_diagnostics.py"
        step_args = [
            "--run-dir", str(run_dir),
            "--run-name", args.run_name,
            "--alerts-dir", args.alerts_dir,
            "--objects-by-hour", args.objects_out,
        ]
        if args.diagnostics_out:
            step_args += ["--out-dir", args.diagnostics_out]
        if args.diagnostics_train_table:
            step_args += ["--train-table", args.diagnostics_train_table]
        if args.diagnostics_train_keys:
            step_args += ["--train-keys", args.diagnostics_train_keys]
        if args.diagnostics_val_keys:
            step_args += ["--val-keys", args.diagnostics_val_keys]
        if args.diagnostics_metrics_json:
            step_args += ["--metrics-json", args.diagnostics_metrics_json]
        if args.diagnostics_feature_meta:
            step_args += ["--feature-metadata", args.diagnostics_feature_meta]
        if args.diagnostics_fail_on_checks:
            step_args.append("--fail-on-checks")
        ok, code = run_step("run-diagnostics", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 10: provenance summary ---
    git_commit = _git_commit(REPO_ROOT)
    source_inputs: List[Dict[str, Any]] = []
    source_input_paths = [
        Path(union_csv),
        Path(patches_csv),
        Path(matches_csv),
        Path(seed_summary),
        Path(objects_in) if objects_in else None,
        Path(args.objects_out) if args.objects_out else None,
        Path(args.object_matches_out) if args.object_matches_out else None,
        Path(objects_rejects_out) if objects_rejects_out else None,
    ]
    if seed_analysis:
        source_input_paths.append(Path(seed_analysis))
    if conversion_csv:
        source_input_paths.append(Path(conversion_csv))
    if ibtracs_path:
        source_input_paths.append(Path(ibtracs_path))
    source_inputs = [_file_signature(p) for p in _collect_files([p for p in source_input_paths if p])]

    artifacts: List[Path] = []
    artifacts.extend([p for p in source_input_paths if p is not None])
    artifacts.extend(_collect_alert_files(Path(args.alerts_dir), args.run_name))
    artifacts.extend(_collect_metrics_files(args.run_name))
    artifacts.extend([p for p in report_pack_out.rglob("*") if p.is_file()])
    artifacts.extend([p for p in (maps_dir).rglob("*") if p.is_file()])
    if object_maps_dir:
        artifacts.extend([p for p in object_maps_dir.rglob("*") if p.is_file()])
    artifacts.extend([p for p in run_dir.rglob("*") if p.is_file()])

    prov_path = _write_provenance(
        run_dir,
        args.run_name,
        config_sha,
        source_inputs,
        artifacts,
        git_commit,
    )
    print(f"[manager] provenance -> {prov_path}")

    print(f"\n[manager] Completed. Run artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
