#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_contracts.py

Contract and pre/post-flight helpers for the orchestrated pipeline.
Encodes per-step inputs, outputs, column expectations, and dependencies so
run_pipeline (and tooling like pipeline_doctor) can fail fast when a required
file/column is missing or steps are ordered incorrectly.
"""

from __future__ import annotations

import dataclasses
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pq = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


# --------------------------- core column contract ---------------------------
# Columns that should be present in the main merged feature table prior to
# modelling, drawn from AGENTS.md. Used for downstream schema guards.
CORE_FEATURE_COLUMNS: List[str] = [
    "time", "lat", "lon",
    "msl", "t2m",
    "u10", "v10", "wspd",
    "zeta", "div", "S",
    "shear10_def", "S3", "S3_src",
    "gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap",
    "gka_F", "gka_msl_nd", "gka_knee_ratio", "gka_chirality",
    "gka_Q", "gka_dir_var", "gka_vortdiv_ratio",
    "sph_center", "sph_radial_signed", "sph_radial_abs", "SFI",
]

GKA_NEW_COLUMNS: List[str] = [
    "gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap",
    "gka_F", "gka_msl_nd", "gka_knee_ratio",
    "gka_chirality", "gka_Q", "gka_dir_var", "gka_vortdiv_ratio",
]

GKA_MS_COLUMNS: List[str] = [
    "gka_spin_coh",
    "gka_build",
    "gka_relax",
    "gka_shear_quench",
    "gka_knee_ms",
    "gka_score",
]

SPHERICAL_COLUMNS: List[str] = [
    "sph_center",
    "sph_radial_signed",
    "sph_radial_abs",
    "sph_vdr_std",
    "t2m_anom_local",
    "pdrop_nd",
    "thermo_shear",
    "SFI",
    "SFI2",
]

OBJECTS_BY_HOUR_COLUMNS: List[str] = [
    "object_id",
    "time",
    "obj_area_cells",
    "obj_centroid_lat",
    "obj_centroid_lon",
    "obj_lat_min",
    "obj_lat_max",
    "obj_lon_min",
    "obj_lon_max",
    "obj_score_max",
    "obj_score_mean",
    "obj_score_topk_mean",
    "obj_core_lat",
    "obj_core_lon",
    "obj_axis_bearing_deg",
    "obj_axis_elongation",
]

OBJECT_MATCH_COLUMNS: List[str] = [
    "storm_id",
    "track_time",
    "track_lat",
    "track_lon",
    "track_bearing_deg",
    "track_speed_kmh",
    "vmax",
    "object_id",
    "obj_track_id",
    "object_time",
    "obj_centroid_lat",
    "obj_centroid_lon",
    "obj_score",
    "obj_area_cells",
    "obj_motion_bearing_deg",
    "obj_motion_speed_kmh",
    "obj_motion_uncertainty_deg",
    "obj_flow_bearing_deg",
    "obj_flow_speed_kmh",
    "obj_axis_bearing_deg",
    "obj_axis_align_motion",
    "obj_corepull_align_motion",
    "obj_persist",
    "obj_track_len_h",
    "d_km",
    "dt_hours",
    "match_score",
    "match_score_gauss",
    "match_is_primary",
    "score_prob",
    "score_dist",
    "score_compact",
    "score_persist",
    "score_area_penalty",
    "match_rank",
]


# --------------------------- data structures ---------------------------
@dataclass
class PathSpec:
    path_keys: Tuple[str, ...]
    kind: str  # "file" | "glob" | "dir"
    required: bool = True
    required_columns: Tuple[str, ...] = ()
    description: str | None = None
    produced_by: Tuple[str, ...] = ()

    def resolve(self, step: Dict[str, Any]) -> tuple[Optional[Path], Optional[str]]:
        for key in self.path_keys:
            val = step.get(key)
            if isinstance(val, str) and val.strip():
                return Path(val), key
        return None, None


@dataclass
class StepContract:
    section: str
    mode: str
    inputs: List[PathSpec] = field(default_factory=list)
    outputs: List[PathSpec] = field(default_factory=list)
    required_input_columns: List[str] = field(default_factory=list)
    output_columns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    input_column_rule: Optional[Any] = None
    output_column_rule: Optional[Any] = None

    def expected_input_columns(self, step: Dict[str, Any]) -> List[str]:
        cols: List[str] = list(self.required_input_columns)
        if self.input_column_rule:
            extra = self.input_column_rule(step)
            if extra:
                cols.extend(extra)
        # dedupe while preserving order
        seen = set()
        uniq: List[str] = []
        for c in cols:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def expected_output_columns(self, step: Dict[str, Any]) -> List[str]:
        cols: List[str] = list(self.output_columns)
        if self.output_column_rule:
            extra = self.output_column_rule(step)
            if extra:
                cols.extend(extra)
        seen = set()
        uniq: List[str] = []
        for c in cols:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq


@dataclass
class PreflightResult:
    contract: StepContract
    input_schemas: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    expected_output_columns: List[str] = field(default_factory=list)
    input_issues: List["InputIssue"] = field(default_factory=list)


@dataclass
class InputIssue:
    spec: PathSpec
    reason: str
    used_key: Optional[str]
    path: Optional[Path]
    detail: str


# --------------------------- schema helpers ---------------------------
def _peek_columns(path: Path) -> List[str]:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()

    if ext == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to inspect parquet schemas.")
        try:
            return list(pq.ParquetFile(path).schema.names)
        except Exception as exc:  # pragma: no cover - diagnostic path
            raise SystemExit(f"Failed to read parquet schema for {path}: {exc}")
    if ext in {".csv", ".csv.gz"}:
        if pd is None:
            raise SystemExit("pandas is required to inspect CSV schemas.")
        try:
            return list(pd.read_csv(path, nrows=0, compression="infer").columns)
        except Exception as exc:  # pragma: no cover - diagnostic path
            raise SystemExit(f"Failed to read CSV header for {path}: {exc}")
    return []


def _row_count(path: Path) -> int:
    suffixes = "".join(path.suffixes[-2:]).lower()
    ext = suffixes if suffixes in {".csv.gz", ".parquet"} else path.suffix.lower()

    if ext == ".parquet":
        if pq is None:
            raise SystemExit("pyarrow is required to inspect parquet row counts.")
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:
            raise SystemExit(f"Failed to open parquet file for row counts: {exc}")
        meta = pf.metadata
        meta_rows = int(meta.num_rows) if meta is not None else 0
        if meta_rows:
            return meta_rows
        try:
            return int(sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups))) if meta else 0
        except Exception:
            try:
                return sum(len(b) for b in pf.iter_batches(batch_size=200_000, columns=[]))
            except Exception:
                return 0
    if ext in {".csv", ".csv.gz"}:
        opener = gzip.open if ext == ".csv.gz" else open
        try:
            row_idx = -1
            with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
                for row_idx, _ in enumerate(fh):
                    pass
            return max(0, row_idx)
        except Exception:
            return 0
    # Fallback for non-tabular artifacts (e.g., model pickle): treat non-empty file as row_count=1
    try:
        return 1 if path.exists() and path.stat().st_size > 0 else 0
    except Exception:
        return 0


# --------------------------- contract registry ---------------------------
def _build_output_columns(step: Dict[str, Any]) -> List[str]:
    cols = ["time", "lat", "lon", "wspd", "msl", "t2m"]
    export_uv = bool(step.get("export_uv") or step.get("export-uv"))
    if export_uv or {"u10", "v10"}.intersection(step.get("require_vars", []) or []):
        cols += ["u10", "v10"]
    if step.get("with_vortdiv"):
        cols += ["zeta", "div", "S", "agree"]
    if step.get("emit_grid_index"):
        cols += ["ilat", "ilon"]
    return cols


def _join_features_output_columns(_: Dict[str, Any]) -> List[str]:
    # Join of base + shear tables; at minimum we expect join keys + shear vars
    return ["time", "lat", "lon", "shear_low", "shear_deep", "S3"]


def _gka_output_columns(_: Dict[str, Any]) -> List[str]:
    return list(GKA_NEW_COLUMNS)


def _integrate_output_columns(_: Dict[str, Any]) -> List[str]:
    return ["time", "lat", "lon", "u10", "v10", "msl", "t2m"]


def _spherical_output_columns(_: Dict[str, Any]) -> List[str]:
    return list(SPHERICAL_COLUMNS)


def _gka_ms_output_columns(_: Dict[str, Any]) -> List[str]:
    return list(GKA_MS_COLUMNS)

def _first_key(step: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in step and step.get(k) not in (None, ""):
            return str(step.get(k))
    return None

def _parse_csv_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        items = [str(v) for v in val]
    else:
        items = [str(val)]
    out: List[str] = []
    for tok in items:
        for part in tok.replace(",", " ").split():
            if part.strip():
                out.append(part.strip())
    return out

def _alert_required_cols(step: Dict[str, Any]) -> List[str]:
    cols = ["time", "lat", "lon"]
    prob = _first_key(step, ("prob_col", "prob-col", "score_col", "score-col"))
    flag = _first_key(step, ("flag_col", "flag-col"))
    if prob:
        cols.append(prob)
    if flag:
        cols.append(flag)
    return cols

def _hourly_rollup_cols(step: Dict[str, Any]) -> List[str]:
    cols = ["time", "lat", "lon"]
    risk = _first_key(step, ("risk_cols", "risk-cols"))
    for c in _parse_csv_list(risk):
        cols.append(c)
    return cols

def _viability_leads_cols(step: Dict[str, Any]) -> List[str]:
    cols = ["t_to_storm_min_h"]
    target = _first_key(step, ("target",))
    if target:
        cols.append(target)
    return cols


STEP_CONTRACTS: Dict[Tuple[str, str], StepContract] = {
    ("features", "build"): StepContract(
        section="features",
        mode="build",
        inputs=[PathSpec(("nc_glob", "nc"), kind="glob", required=True)],
        outputs=[PathSpec(("out", "out_features"), kind="file", required=True)],
        output_column_rule=_build_output_columns,
    ),
    ("features", "bulk-shear"): StepContract(
        section="features",
        mode="bulk-shear",
        inputs=[PathSpec(("pl_glob", "pl_globs"), kind="glob", required=True)],
        outputs=[PathSpec(("out",), kind="file", required=True)],
        output_columns=["time", "lat", "lon", "shear_low", "shear_deep", "S3"],
    ),
    ("features", "join-features"): StepContract(
        section="features",
        mode="join-features",
        inputs=[
            PathSpec(("left",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("features.build",)),
            PathSpec(("right",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("features.bulk-shear",)),
        ],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        output_column_rule=_join_features_output_columns,
        dependencies=["build", "bulk-shear"],
    ),
    ("features", "patch"): StepContract(
        section="features",
        mode="patch",
        inputs=[PathSpec(
            ("in", "infile"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon", "msl", "zeta", "div", "S", "u10", "v10"),
            produced_by=("features.join-features",),
        )],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        output_columns=[
            "msl_d1h", "msl_d3h",
            "zeta_mean3h", "zeta_std3h",
            "div_mean3h", "div_std3h",
            "S_mean3h", "S_std3h",
            "shear10_def", "shear_proxy", "S3", "S3_src",
            "dS_dt", "drelax_dt", "dagree_dt",
            "msl_grad",
        ],
        dependencies=["join-features"],
    ),
    ("features", "gka"): StepContract(
        section="features",
        mode="gka",
        inputs=[PathSpec(
            ("infile", "in"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon", "u10", "v10", "zeta", "div", "S", "S3", "msl"),
            produced_by=("features.patch",),
        )],
        outputs=[PathSpec(("outfile", "out"), kind="file", required=True)],
        output_column_rule=_gka_output_columns,
        dependencies=["patch"],
    ),
    ("features", "integrate-thermo"): StepContract(
        section="features",
        mode="integrate-thermo",
        inputs=[
            PathSpec(("features", "infile", "in"), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("features.gka",)),
            PathSpec(("thermo_glob", "nc_glob"), kind="glob", required=True, produced_by=("fetch.era5",)),
        ],
        outputs=[PathSpec(("out",), kind="file", required=True)],
        output_column_rule=_integrate_output_columns,
        dependencies=["gka"],
    ),
    ("features", "spherical-feedback"): StepContract(
        section="features",
        mode="spherical-feedback",
        inputs=[PathSpec(
            ("infile", "labelled", "in"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon", "msl", "msl_d1h", "u10", "v10"),
            produced_by=("features.integrate-thermo",),
        )],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        output_column_rule=_spherical_output_columns,
        dependencies=["integrate-thermo"],
    ),
    ("features", "gka-ms"): StepContract(
        section="features",
        mode="gka-ms",
        inputs=[PathSpec(
            ("infile", "in"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon", "zeta_mean3h", "div_mean3h", "S3", "dS_dt"),
            produced_by=("features.spherical-feedback",),
        )],
        outputs=[PathSpec(("outfile", "out"), kind="file", required=True)],
        output_column_rule=_gka_ms_output_columns,
        dependencies=["spherical-feedback"],
    ),
    ("features", "join-labels-grid"): StepContract(
        section="features",
        mode="join-labels-grid",
        inputs=[
            PathSpec(("features", "infile", "in"), kind="file", required=True, required_columns=tuple(CORE_FEATURE_COLUMNS), produced_by=("features.gka-ms",)),
            PathSpec(("labels",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        output_columns=["storm_point", "storm_window", "storm", "near_storm", "pregen", "t_to_storm_min_h"],
        dependencies=["gka-ms"],
    ),
    ("data_stage", "add-ids"): StepContract(
        section="data_stage",
        mode="add-ids",
        inputs=[PathSpec(
            ("infile", "in"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon"),
            produced_by=("features.join-labels-grid",),
        )],
        outputs=[PathSpec(("outfile", "out"), kind="file", required=True)],
        output_columns=["row_id"],
    ),
    ("data_stage", "viability-targets"): StepContract(
        section="data_stage",
        mode="viability-targets",
        inputs=[PathSpec(("panel", "infile", "in"), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("data_stage.gse-states",))],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
    ),
    ("data_stage", "train-viability"): StepContract(
        section="data_stage",
        mode="train-viability",
        inputs=[PathSpec(("train",), kind="file", required=True, produced_by=("data_stage.viability-targets",))],
        outputs=[
            PathSpec(("model_out", "model-out"), kind="file", required=True),
            PathSpec(("metrics_json", "metrics-json"), kind="file", required=False),
            PathSpec(("coefs_csv", "coefs-csv"), kind="file", required=False),
        ],
        dependencies=["viability-targets"],
    ),
    ("data_stage", "state-transitions"): StepContract(
        section="data_stage",
        mode="state-transitions",
        inputs=[PathSpec(("infile", "in"), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("data_stage.add-ids",))],
        outputs=[PathSpec(("outfile", "out"), kind="file", required=True)],
        dependencies=["add-ids"],
    ),
    # training section
    ("training", "train-base"): StepContract(
        section="training",
        mode="train-base",
        inputs=[PathSpec(
            ("train", "infile", "features"),
            kind="file",
            required=True,
            required_columns=("time", "lat", "lon", "storm"),
            produced_by=("data_stage.state-transitions",),
        )],
        outputs=[
            PathSpec(("model_out", "model-out", "base_model"), kind="file", required=True),
            PathSpec(("calibration_out", "calibration-out", "base_calibrator", "base-calibrator"), kind="file", required=False),
            PathSpec(("metrics_out", "metrics-out", "base_metrics", "base-metrics"), kind="file", required=False),
        ],
    ),
    ("training", "train-alert-specialist"): StepContract(
        section="training",
        mode="train-alert-specialist",
        inputs=[
            PathSpec(("train", "features"), kind="file", required=True, required_columns=("time", "lat", "lon", "storm"), produced_by=("data_stage.state-transitions",)),
            PathSpec(("alerts",), kind="file", required=True, produced_by=("alerts_logic.viability-pipeline",)),
        ],
        outputs=[
            PathSpec(("model_out", "model-out", "specialist_model", "specialist-model"), kind="file", required=True),
            PathSpec(("calibration_out", "calibration-out", "specialist_calibrator", "specialist-calibrator"), kind="file", required=False),
            PathSpec(("metrics_out", "metrics-out", "specialist_metrics", "specialist-metrics"), kind="file", required=False),
        ],
        dependencies=["train-base"],
    ),
    ("training", "predict-alerts"): StepContract(
        section="training",
        mode="predict-alerts",
        inputs=[
            PathSpec(("features", "infile"), kind="file", required=True, produced_by=("data_stage.state-transitions",)),
            PathSpec(("base_model", "base-model"), kind="file", required=True, produced_by=("training.train-base",)),
            PathSpec(("base_calibrator", "base-calibrator"), kind="file", required=False, produced_by=("training.train-base",)),
            PathSpec(("specialist_model", "specialist-model"), kind="file", required=False, produced_by=("training.train-alert-specialist",)),
            PathSpec(("specialist_calibrator", "specialist-calibrator"), kind="file", required=False, produced_by=("training.train-alert-specialist",)),
        ],
        outputs=[PathSpec(("outfile", "out"), kind="file", required=True)],
        dependencies=["train-base", "train-alert-specialist"],
    ),
    ("training", "track-objects"): StepContract(
        section="training",
        mode="track-objects",
        inputs=[PathSpec(("infile",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("training.predict-alerts",))],
        outputs=[
            PathSpec(("objects_out", "objects-out"), kind="file", required=False),
            PathSpec(("join_out", "join-out"), kind="file", required=False),
        ],
        dependencies=["predict-alerts"],
    ),
    # alerts logic section
    ("alerts_logic", "apply-thresholds"): StepContract(
        section="alerts_logic",
        mode="apply-thresholds",
        inputs=[
            PathSpec(("labelled", "features", "infile"), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("features.join-labels-grid",)),
            PathSpec(("model",), kind="file", required=True, produced_by=("data_stage.train-viability", "training.train-base")),
        ],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
    ),
    ("alerts_logic", "throttle"): StepContract(
        section="alerts_logic",
        mode="throttle",
        inputs=[PathSpec(("alerts",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("alerts_logic.apply-thresholds",))],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        input_column_rule=_alert_required_cols,
        dependencies=["apply-thresholds"],
    ),
    ("alerts_logic", "denoise"): StepContract(
        section="alerts_logic",
        mode="denoise",
        inputs=[PathSpec(("alerts",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("alerts_logic.throttle",))],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        input_column_rule=_alert_required_cols,
        dependencies=["throttle"],
    ),
    ("alerts_logic", "viability-pipeline"): StepContract(
        section="alerts_logic",
        mode="viability-pipeline",
        inputs=[
            PathSpec(("labelled",), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("data_stage.viability-targets",)),
            PathSpec(("model",), kind="file", required=True, produced_by=("data_stage.train-viability",)),
        ],
        outputs=[
            PathSpec(("base_out", "base-out"), kind="file", required=True),
            PathSpec(("thr_out", "thr-out"), kind="file", required=True),
            PathSpec(("out", "outfile"), kind="file", required=True),
        ],
    ),
    # eval section
    ("eval", "viability-leads"): StepContract(
        section="eval",
        mode="viability-leads",
        inputs=[
            PathSpec(("panel",), kind="file", required=True, produced_by=("data_stage.viability-targets",)),
            PathSpec(("model",), kind="file", required=True, produced_by=("data_stage.train-viability",)),
        ],
        outputs=[PathSpec(("out",), kind="file", required=True)],
        input_column_rule=_viability_leads_cols,
    ),
    ("eval", "hourly-metrics"): StepContract(
        section="eval",
        mode="hourly-metrics",
        inputs=[PathSpec(("alerts",), kind="file", required=True, produced_by=("alerts_logic.viability-pipeline",))],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        input_column_rule=_alert_required_cols,
    ),
    ("eval", "hourly-rollup"): StepContract(
        section="eval",
        mode="hourly-rollup",
        inputs=[PathSpec(("alerts",), kind="file", required=True, produced_by=("alerts_logic.viability-pipeline",))],
        outputs=[PathSpec(("out", "outfile"), kind="file", required=True)],
        input_column_rule=_hourly_rollup_cols,
    ),
    # seeds section
    ("seeds", "from-alerts"): StepContract(
        section="seeds",
        mode="from-alerts",
        inputs=[PathSpec(("alerts",), kind="glob", required=True, produced_by=("alerts_logic.viability-pipeline",))],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
        input_column_rule=_alert_required_cols,
    ),
    ("seeds", "starts-vs-tracks"): StepContract(
        section="seeds",
        mode="starts-vs-tracks",
        inputs=[
            PathSpec(("seeds",), kind="file", required=True, produced_by=("seeds.from-alerts",)),
            PathSpec(("tracks",), kind="file", required=True, produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
    ),
    ("seeds", "proto-outcomes"): StepContract(
        section="seeds",
        mode="proto-outcomes",
        inputs=[
            PathSpec(("seeds",), kind="file", required=True, produced_by=("seeds.from-alerts",)),
            PathSpec(("ibtracs",), kind="file", required=True, produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
    ),
    ("seeds", "gse-tracks"): StepContract(
        section="seeds",
        mode="gse-tracks",
        inputs=[PathSpec(("panel",), kind="file", required=True, produced_by=("data_stage.viability-targets",))],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
    ),
    ("seeds", "analyze"): StepContract(
        section="seeds",
        mode="analyze",
        inputs=[PathSpec(("seeds",), kind="file", required=True, produced_by=("seeds.from-alerts",))],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
    ),
    # report section (object-based)
    ("report", "objects-by-hour"): StepContract(
        section="report",
        mode="objects-by-hour",
        inputs=[PathSpec(("infile", "objects_in", "objects-in"), kind="file", required=True, required_columns=("time", "lat", "lon"), produced_by=("alerts_logic.viability-pipeline",))],
        outputs=[
            PathSpec(("objects_out", "objects-out"), kind="file", required=True),
            PathSpec(("cells_out", "cells-out"), kind="file", required=False),
        ],
        output_columns=OBJECTS_BY_HOUR_COLUMNS,
    ),
    ("report", "object-matches"): StepContract(
        section="report",
        mode="object-matches",
        inputs=[
            PathSpec(
                ("objects", "objects_out", "objects-out"),
                kind="file",
                required=True,
                required_columns=("object_id", "obj_centroid_lat", "obj_centroid_lon", "obj_score_max", "obj_axis_bearing_deg", "obj_core_lat", "obj_core_lon"),
                produced_by=("report.objects-by-hour",),
            ),
            PathSpec(("tracks", "ibtracs"), kind="file", required=True, produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out", "object_matches_out", "object-matches-out"), kind="file", required=True)],
        output_columns=OBJECT_MATCH_COLUMNS,
        dependencies=["objects-by-hour"],
    ),
    ("report", "object-maps"): StepContract(
        section="report",
        mode="object-maps",
        inputs=[
            PathSpec(
                ("matches", "object_matches_out", "object-matches-out"),
                kind="file",
                required=True,
                required_columns=("storm_id", "object_time", "obj_centroid_lat", "obj_centroid_lon"),
                produced_by=("report.object-matches",),
            ),
            PathSpec(("tracks", "ibtracs"), kind="file", required=True, produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out_dir", "out-dir", "object-maps-dir"), kind="dir", required=True)],
        dependencies=["object-matches"],
    ),
    ("report", "report-pack"): StepContract(
        section="report",
        mode="report-pack",
        inputs=[
            PathSpec(("config", "report_pack_config", "report-pack-config"), kind="file", required=False),
            PathSpec(("objects", "objects_out", "objects-out"), kind="file", required=False),
            PathSpec(("matches", "object_matches_out", "object-matches-out"), kind="file", required=False),
        ],
        outputs=[PathSpec(("out_dir", "out-dir", "report-pack-out-dir"), kind="dir", required=True)],
        dependencies=["object-matches"],
    ),
    ("report", "reporting-v2"): StepContract(
        section="report",
        mode="reporting-v2",
        inputs=[
            PathSpec(("tables_dir", "tables-dir"), kind="dir", required=False, produced_by=("report.report-pack",)),
            PathSpec(("matches", "object_matches_out", "object-matches-out"), kind="file", required=False, produced_by=("report.object-matches",)),
            PathSpec(("tracks", "ibtracs"), kind="file", required=False, produced_by=("fetch.ibtracs",)),
        ],
        outputs=[PathSpec(("out_dir", "out-dir"), kind="dir", required=True)],
        dependencies=["report-pack"],
    ),
    ("report", "bundle"): StepContract(
        section="report",
        mode="bundle",
        inputs=[],
        outputs=[PathSpec(("out_dir", "out-dir", "run_dir", "run-dir"), kind="dir", required=False)],
    ),
}


def contract_for(section: str, mode: str) -> Optional[StepContract]:
    return STEP_CONTRACTS.get((section, mode))


# --------------------------- dependency + planning ---------------------------
def order_steps(section: str, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Topologically sort steps for a section using contract-defined dependencies.
    Raises SystemExit if a dependency is missing.
    """
    enabled_steps = [s for s in steps if s is not None and s.get("enabled", True) is not False]
    mode_to_step = {str(s.get("mode", "")).strip(): s for s in enabled_steps}

    # Validate dependencies before sorting
    for step in enabled_steps:
        mode = str(step.get("mode", "")).strip()
        ct = contract_for(section, mode)
        if not ct or not ct.dependencies:
            continue
        missing = [d for d in ct.dependencies if d not in mode_to_step]
        if missing:
            miss = ", ".join(missing)
            raise SystemExit(
                f"[plan] Step {section}.{mode} requires {miss} but they are not enabled in this section."
            )

    # Build graph
    edges: Dict[str, set[str]] = {}
    for step in enabled_steps:
        mode = str(step.get("mode", "")).strip()
        ct = contract_for(section, mode)
        deps = set(ct.dependencies) if ct else set()
        edges[mode] = deps

    # Kahn topological sort
    out: List[Dict[str, Any]] = []
    ready = [m for m, deps in edges.items() if not deps]
    processed: set[str] = set()
    while ready:
        m = ready.pop(0)
        processed.add(m)
        if m in mode_to_step:
            out.append(mode_to_step[m])
        for nxt, deps in list(edges.items()):
            if m in deps:
                deps.remove(m)
                edges[nxt] = deps
                if not deps and nxt not in processed and nxt not in ready:
                    ready.append(nxt)

    if len(out) != len(enabled_steps):
        raise SystemExit(f"[plan] Dependency cycle or missing node detected in {section} steps.")
    return out


# --------------------------- pre/post-flight ---------------------------
def _format_step_label(section: str, mode: str) -> str:
    return f"{section}.{mode}"

def _format_produced_by(spec: PathSpec) -> str:
    if not spec.produced_by:
        return ""
    hints: List[str] = []
    for item in spec.produced_by:
        if not item:
            continue
        if "." in item:
            sec, mode = item.split(".", 1)
            hints.append(f"{sec}.steps[*].mode == {mode}")
        else:
            hints.append(item)
    if not hints:
        return ""
    uniq = ", ".join(dict.fromkeys(hints))
    return f" Produced by {', '.join(spec.produced_by)}; enable {uniq}."


def _validate_required_columns(path: Path, required: Iterable[str]) -> List[str]:
    if not required:
        return []
    cols = _peek_columns(path)
    missing = [c for c in required if c not in cols]
    return missing


def preflight_step(section: str, step: Dict[str, Any]) -> PreflightResult:
    mode = str(step.get("mode", "")).strip() or "(unknown)"
    contract = contract_for(section, mode) or StepContract(section, mode)
    errors: List[str] = []
    input_schemas: Dict[str, List[str]] = {}
    input_issues: List[InputIssue] = []

    for spec in contract.inputs:
        path, used_key = spec.resolve(step)
        label = _format_step_label(section, mode)
        if path is None:
            if spec.required:
                errors.append(f"{label}: missing required input key(s) {spec.path_keys}.{_format_produced_by(spec)}")
                input_issues.append(InputIssue(spec, "missing_key", used_key=None, path=None, detail="missing input key"))
            continue
        if spec.kind == "glob":
            matches = sorted(Path().glob(str(path)))
            if not matches:
                if spec.required:
                    errors.append(f"{label}: glob '{path}' matched no files.{_format_produced_by(spec)}")
                    input_issues.append(InputIssue(spec, "missing_glob", used_key, path, detail="glob matched no files"))
                continue
            target = matches[0]
            if spec.required:
                non_empty = None
                for mp in matches:
                    if _row_count(mp) > 0:
                        non_empty = mp
                        break
                if non_empty is None:
                    errors.append(f"{label}: glob '{path}' matched only empty files.{_format_produced_by(spec)}")
                    input_issues.append(InputIssue(spec, "empty_glob", used_key, path, detail="glob matched only empty files"))
                    continue
                target = non_empty
        else:
            target = path
            if not target.exists():
                if spec.required:
                    errors.append(f"{label}: required input not found ({used_key}={path}).{_format_produced_by(spec)}")
                    input_issues.append(InputIssue(spec, "missing_file", used_key, path, detail="required input file missing"))
                continue
            if spec.kind == "dir":
                if not target.is_dir():
                    errors.append(f"{label}: expected input dir {used_key or path} is not a directory.{_format_produced_by(spec)}")
                    input_issues.append(InputIssue(spec, "invalid_dir", used_key, path, detail="input path is not a directory"))
                    continue
                if spec.required:
                    try:
                        has_child = any(target.iterdir())
                    except Exception:
                        has_child = False
                    if not has_child:
                        errors.append(f"{label}: required input dir {used_key or path} is empty.{_format_produced_by(spec)}")
                        input_issues.append(InputIssue(spec, "empty_dir", used_key, path, detail="input directory empty"))
                continue
            if spec.required:
                rows = _row_count(target)
                if rows <= 0:
                    errors.append(f"{label}: required input {used_key or path} is empty or unreadable.{_format_produced_by(spec)}")
                    input_issues.append(InputIssue(spec, "empty_file", used_key, path, detail="input file empty or unreadable"))
                    continue

        if spec.required_columns:
            missing_cols = _validate_required_columns(target, spec.required_columns)
            if missing_cols:
                errors.append(
                    f"{label}: missing columns {missing_cols} in {target}. "
                    f"Upstream step may not have produced them.{_format_produced_by(spec)}"
                )
                input_issues.append(InputIssue(spec, "missing_columns", used_key, target, detail="missing required columns"))
            else:
                input_schemas[used_key or str(path)] = _peek_columns(target)
        else:
            # still capture schema when cheap
            try:
                input_schemas[used_key or str(path)] = _peek_columns(target)
            except Exception:
                pass

    expected_outputs = contract.expected_output_columns(step)
    return PreflightResult(
        contract=contract,
        input_schemas=input_schemas,
        errors=errors,
        warnings=[],
        expected_output_columns=expected_outputs,
        input_issues=input_issues,
    )


def postflight_step(
    section: str,
    step: Dict[str, Any],
    expected_output_columns: Optional[List[str]] = None,
) -> None:
    mode = str(step.get("mode", "")).strip() or "(unknown)"
    contract = contract_for(section, mode) or StepContract(section, mode)
    outputs = contract.outputs or [PathSpec(("out", "outfile"), kind="file", required=True)]
    label = _format_step_label(section, mode)
    for spec in outputs:
        path, used_key = spec.resolve(step)
        if path is None:
            if spec.required:
                raise SystemExit(f"{label}: missing required output key(s) {spec.path_keys}.")
            continue
        if spec.kind == "glob":
            matches = sorted(Path().glob(str(path)))
            if not matches and spec.required:
                raise SystemExit(f"{label}: expected output glob '{path}' matched no files.")
            continue
        if not path.exists():
            if spec.required:
                raise SystemExit(f"{label}: expected output {used_key or path} not found after run.")
            # optional output absent -> skip validation
            continue
        if spec.kind == "dir":
            if not path.is_dir():
                raise SystemExit(f"{label}: expected output dir {used_key or path} is not a directory.")
            continue
        rows = _row_count(path)
        if rows <= 0:
            mode = str(step.get("mode", "")).strip()
            if section == "training" and mode == "track-objects":
                print(f"[warn] {label}: output {path} is empty (no objects found); continuing.")
                continue
            raise SystemExit(f"{label}: output {path} is empty or unreadable.")
        needed_cols = expected_output_columns or contract.expected_output_columns(step)
        needed_cols = [c for c in needed_cols if c]  # drop blanks
        if needed_cols:
            cols = _peek_columns(path)
            missing = [c for c in needed_cols if c not in cols]
            if missing:
                raise SystemExit(f"{label}: output {path} missing columns {missing}.")


# --------------------------- reporting ---------------------------
def summarize_step(section: str, step: Dict[str, Any], pref: Optional[PreflightResult] = None) -> str:
    mode = str(step.get("mode", "")).strip() or "(unknown)"
    parts = [f"{section}.{mode}"]
    if pref and pref.errors:
        parts.append(f"ERRORS={len(pref.errors)}")
    if pref and pref.expected_output_columns:
        cols = ", ".join(pref.expected_output_columns[:6])
        more = len(pref.expected_output_columns) - 6
        if more > 0:
            cols += f", +{more} more"
        parts.append(f"outputs: {cols}")
    return " | ".join(parts)
