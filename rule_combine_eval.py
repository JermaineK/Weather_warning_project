#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_pipeline.py  — end-to-end grid alert pipeline with overwrite controls.

Usage:
  python run_pipeline.py --config config/pipeline.yaml [--overwrite]

YAML (optional per-stage overwrite):
overwrite:
  integrate: true
  train: true
  calibrate: false
  thresholds: true
  alerts: true
  reports: true
"""

import argparse, os, subprocess, sys, textwrap, math
from pathlib import Path
import pandas as pd

try:
    import yaml
except Exception:
    print("Please `pip install pyyaml`.", file=sys.stderr)
    raise


# ---------------- util helpers ----------------

def sh(cmd, check=True):
    print(f"\n$ {' '.join(map(str, cmd))}")
    return subprocess.run(cmd, check=check)

def exists(p: Path | str) -> bool:
    return Path(p).exists()

def _safe_unlink(path: Path | str):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        print(f"[warn] failed to remove existing file: {path} ({e})")

def maybe_run(msg, out_path, cmd, overwrite=False):
    """
    Run a shell command that produces out_path.
    - If out_path is missing -> run.
    - If out_path exists:
        - overwrite=True -> delete and run.
        - overwrite=False -> skip.
    """
    print(f"\n== {msg} ==")
    if out_path and exists(out_path):
        if overwrite:
            print(f"[overwrite] removing existing: {out_path}")
            _safe_unlink(out_path)
            sh(cmd)
        else:
            print(f"[skip] found: {out_path}")
    else:
        sh(cmd)

def sanitize_time(path, time_col="time"):
    """Rewrite CSV timestamps as tz-naive UTC in a canonical format to avoid downstream drift."""
    p = str(path)
    df = pd.read_csv(p, compression="infer")
    # Handle BOM on header if present
    if time_col not in df.columns:
        bom = "\ufeff" + time_col
        if bom in df.columns:
            df = df.rename(columns={bom: time_col})
        else:
            raise ValueError(f"{p}: missing '{time_col}' column")
    # Parse to tz-naive UTC
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None)
    bad = int(t.isna().sum())
    if bad:
        print(f"[SANITIZE] {bad:,} invalid {time_col} in {os.path.basename(p)}; dropping.", flush=True)
        df = df.loc[t.notna()].copy()
        df[time_col] = t[t.notna()]
    else:
        df[time_col] = t
    df.to_csv(
        p,
        index=False,
        compression="gzip" if str(p).lower().endswith(".gz") else "infer",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    print(f"[SANITIZE] normalized timestamps -> {p}", flush=True)


# ---------------- diagnostics helpers ----------------

def _infer_grid(df: pd.DataFrame, time_col="time", flag_col="alert"):
    """Infer H,W,T and grid completeness; return dict and cleaned df."""
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
    bad_t = int(t.isna().sum())
    df = df.loc[t.notna()].copy()
    df[time_col] = t[t.notna()]
    H = df["lat"].nunique()
    W = df["lon"].nunique()
    T = df[time_col].nunique()
    expected = int(H * W * T)
    rows = len(df)
    complete = (rows == expected)
    return {
        "rows": rows, "bad_time": bad_t,
        "H": H, "W": W, "T": T,
        "expected_rows": expected,
        "grid_complete": bool(complete),
    }, df

def _dup_keys(df: pd.DataFrame, time_col="time"):
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None)
    return int(df.assign(_t=t).duplicated(subset=["_t","lat","lon"]).sum())

def _coverage(df: pd.DataFrame, flag_col):
    if flag_col not in df.columns:
        return math.nan, 0
    v = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)
    return float(v.mean()), int(v.sum())

def _top_hours(df: pd.DataFrame, flag_col="alert_final", k=5):
    if "time" not in df.columns or flag_col not in df.columns:
        return []
    tt = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
    byh = pd.DataFrame({"t": tt, "v": pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)}) \
            .groupby("t", sort=True)["v"].sum().sort_values(ascending=False)
    return [(str(idx), int(val)) for idx, val in byh.head(k).items()]

def collect_alert_diagnostics(base_path: Path, den_path: Path, thr_path: Path, flag_col="alert_final"):
    """Return dict of diagnostics across stages for one lead."""
    out = {
        "file_base": base_path.name if base_path.exists() else "",
        "file_denoised": den_path.name if den_path.exists() else "",
        "file_throttled": thr_path.name if thr_path.exists() else "",
    }

    # Stage 1: base alerts
    if base_path.exists():
        dfb = pd.read_csv(base_path)
        g1, dfb = _infer_grid(dfb, "time", flag_col="alert" if "alert" in dfb.columns else flag_col)
        out.update({f"base_{k}": v for k, v in g1.items()})
        out["base_dup_keys"] = _dup_keys(dfb, "time")
        cov1, act1 = _coverage(dfb, "alert" if "alert" in dfb.columns else flag_col)
        out["base_cov"] = cov1; out["base_active"] = act1
    else:
        return out

    # Stage 2: denoised
    if den_path.exists():
        dfd = pd.read_csv(den_path)
        g2, dfd = _infer_grid(dfd, "time", flag_col=flag_col)
        out.update({f"den_{k}": v for k, v in g2.items()})
        out["den_dup_keys"] = _dup_keys(dfd, "time")
        cov2, act2 = _coverage(dfd, flag_col)
        out["den_cov"] = cov2; out["den_active"] = act2
        out["retained_after_denoise"] = (act2 / act1) if act1 else math.nan
        out["den_top_hours"] = "; ".join(f"{t}={n}" for t, n in _top_hours(dfd, flag_col))
    else:
        out["retained_after_denoise"] = math.nan

    # Stage 3: throttled
    if thr_path.exists():
        dft = pd.read_csv(thr_path)
        g3, dft = _infer_grid(dft, "time", flag_col=flag_col)
        out.update({f"thr_{k}": v for k, v in g3.items()})
        out["thr_dup_keys"] = _dup_keys(dft, "time")
        cov3, act3 = _coverage(dft, flag_col)
        out["thr_cov"] = cov3; out["thr_active"] = act3
        base_or_den = out.get("den_active", act1) or 1
        out["retained_after_throttle"] = (act3 / base_or_den) if base_or_den else math.nan
        out["thr_top_hours"] = "; ".join(f"{t}={n}" for t, n in _top_hours(dft, flag_col))
    else:
        out["retained_after_throttle"] = math.nan

    # Quick red flags
    flags = []
    if out.get("base_bad_time", 0) > 0:
        flags.append(f"base_bad_time={out['base_bad_time']}")
    if not out.get("base_grid_complete", True):
        flags.append("base_incomplete_grid")
    if out.get("den_grid_complete") is False:
        flags.append("den_incomplete_grid")
    if out.get("thr_grid_complete") is False:
        flags.append("thr_incomplete_grid")
    if out.get("retained_after_denoise", 1.0) < 0.5:
        flags.append("large_drop_after_denoise")
    if out.get("retained_after_throttle", 1.0) < 0.5:
        flags.append("large_drop_after_throttle")
    if out.get("base_dup_keys", 0) > 0:
        flags.append(f"base_dup_keys={out['base_dup_keys']}")
    if out.get("den_dup_keys", 0) > 0:
        flags.append(f"den_dup_keys={out['den_dup_keys']}")
    if out.get("thr_dup_keys", 0) > 0:
        flags.append(f"thr_dup_keys={out['thr_dup_keys']}")
    out["flags"] = ", ".join(flags)

    return out


# ---------------- main pipeline ----------------

def main():
    ap = argparse.ArgumentParser(description="End-to-end grid alert pipeline")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--overwrite", action="store_true",
                    help="Force re-run stages even if outputs exist (overrides per-stage settings).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_name = cfg.get("run_name", "run")

    # Per-stage overwrite from config (defaults False); CLI --overwrite wins
    ow_cfg = cfg.get("overwrite", {})
    OW_ALL = bool(args.overwrite)  # global override
    OW_INT = OW_ALL or bool(ow_cfg.get("integrate", False))
    OW_TRN = OW_ALL or bool(ow_cfg.get("train", False))
    OW_CAL = OW_ALL or bool(ow_cfg.get("calibrate", False))
    OW_THR = OW_ALL or bool(ow_cfg.get("thresholds", False))
    OW_ALR = OW_ALL or bool(ow_cfg.get("alerts", False))     # apply→denoise→throttle
    OW_RPT = OW_ALL or bool(ow_cfg.get("reports", False))

    labelled = cfg["labelled_csv"]
    target   = cfg.get("target", "pregen")

    # 0) ERA5 integrate (optional)
    era = cfg.get("era5", {})
    if era.get("enabled", False):
        integrate_out = era["out_csv"]
        integrate_cmd = [
            sys.executable, "integrate_era5_thermo.py",
            "--labelled", labelled,
            "--nc-glob", era["nc_glob"],
            "--out", integrate_out,
        ]
        if era.get("nearest", True):
            integrate_cmd += ["--nearest", "--nearest-maxdeg", str(era.get("nearest_maxdeg", 0.4))]
        maybe_run("ERA5 -> integrate thermo", integrate_out, integrate_cmd, overwrite=OW_INT)
        labelled = integrate_out  # downstream uses integrated file

    # 1) Train
    model_out = cfg["model_out"]
    train_cmd = [
        sys.executable, "train_grid_logit_from_csv.py",
        "--labelled", labelled,
        "--target", target,
        "--test-size", str(cfg.get("test_size", 0.10)),
        "--impute", cfg.get("impute", "median")),
        "--clip-quantile", str(cfg.get("clip_quantile", 0.999)),
        "--model-out", model_out,
        "--auto-clean",
    ]
    # ^^^ NOTE: If your train script does not accept a trailing ')' after --impute, remove it.
    # Keeping compatibility with earlier versions:
    if train_cmd[8].endswith(")"):  # defensive cleanup if copied from old snippet
        train_cmd[8] = train_cmd[8][:-1]

    cw = cfg.get("class_weight", None)
    if cw:
        train_cmd += ["--class-weight", cw]
    maybe_run("Train model", model_out, train_cmd, overwrite=OW_TRN)

    # 2) Calibrate
    model_out_cal = cfg["model_out_cal"]
    cal_cmd = [
        sys.executable, "calibrate_grid.py",
        "--labelled", labelled,
        "--model-in", model_out,
        "--model-out", model_out_cal,
        "--target", target,
        "--method", "isotonic",
        "--test-size", str(cfg.get("test_size", 0.10)),
    ]
    maybe_run("Calibrate model", model_out_cal, cal_cmd, overwrite=OW_CAL)

    # 3) Lead metrics (progress eval) — always rerun (no single output to gate)
    leads = [str(x) for x in cfg.get("leads", [24, 48, 72])]
    eval_cmd = [
        sys.executable, "eval_leadtime_grid_progress.py",
        "--labelled", labelled,
        "--model", model_out_cal,
        "--target", target,
        "--lead-hours", *leads,
        "--chunk-rows", str(cfg.get("chunk_rows", 2_000_000)),
        "--window-mode", cfg.get("window_mode", "time"),
    ]
    sh(eval_cmd)

    # 4) Constrained thresholds (Fβ)
    cons     = cfg.get("constraints", {})
    thr_csv  = cons["out_csv"]
    thr_save = cons.get("save_curve")
    thr_cmd = [
        sys.executable, "find_best_f1_thresholds_constrained.py",
        "--labelled", labelled,
        "--model", model_out_cal,
        "--target", target,
        "--leads", *leads,
        "--min-precision", str(cons.get("min_precision", 0.15)),
        "--max-coverage", str(cons.get("max_coverage", 0.10)),
        "--fbeta", str(cons.get("fbeta", 0.5)),
        "--out", thr_csv,
    ]
    if thr_save:
        thr_cmd += ["--save-table", thr_save]
    if cons.get("verbose", False):
        thr_cmd += ["--verbose"]
    maybe_run("Find constrained thresholds", thr_csv, thr_cmd, overwrite=OW_THR)

    # parse thresholds (expects columns: lead_h + thr_Fbeta or thr_f1 or thr)
    thr_df = pd.read_csv(thr_csv)
    if "thr_Fbeta" in thr_df.columns:
        use_col = "thr_Fbeta"
    elif "thr_f1" in thr_df.columns:
        use_col = "thr_f1"
    elif "thr" in thr_df.columns:
        use_col = "thr"
    else:
        raise ValueError(f"{thr_csv}: no threshold column found.")
    per_lead_thr = {}
    for _, r in thr_df.iterrows():
        lh = int(r["lead_h"])
        try:
            per_lead_thr[lh] = float(r[use_col])
        except Exception:
            pass

    # 5) Alerts per lead -> sanitize -> INTENSITY -> denoise -> throttle -> eval
    alrt_cfg = cfg["alerts"]
    out_dir  = Path(alrt_cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    keepq_map = {int(k): float(v) for k, v in alrt_cfg.get("throttle", {}).get("per_lead_keep_quantile", {}).items()}
    flag_col  = alrt_cfg.get("flag_col", "alert_final")

    summary_lines = []
    for lead in [int(x) for x in leads]:
        thr = per_lead_thr.get(lead)
        if thr is None:
            print(f"[warn] no threshold for lead={lead}; skipping alerts for this lead.")
            continue

        base_out = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}.csv.gz"
        den_out  = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}_denoised.csv.gz"
        top_out  = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}_throttled.csv.gz"

        # apply thresholds (BASE alerts with 'prob')
        apply_cmd = [
            sys.executable, "apply_thresholds.py",
            "--labelled", labelled,
            "--model", model_out_cal,
            "--lead-hours", str(lead),
            "--thr", f"{thr:.6f}",
            "--out", str(base_out),
        ]
        maybe_run(f"Apply thresholds (lead={lead})", base_out, apply_cmd, overwrite=OW_ALR)

        # sanitize times (fixes parsing issues for denoise & intensity)
        sanitize_time(base_out, time_col="time")

        # ---- intensity analysis (optional; runs on BASE alerts with 'prob') ----
        int_cfg = cfg.get("intensity", {})
        if int_cfg.get("enabled", False) and int_cfg.get("tracks"):
            int_out_dir = Path(int_cfg.get("out_dir", "results/intensity"))
            int_out_dir.mkdir(parents=True, exist_ok=True)
            int_out = int_out_dir / f"intensity_{run_name}_lead{lead}.csv"

            int_cmd = [
                sys.executable, "intensity_analysis.py",
                "--alerts", str(base_out),            # BASE alerts (has 'prob')
                "--tracks", str(int_cfg["tracks"]),
                "--radius-deg", str(int_cfg.get("radius_deg", 0.75)),
                "--time-tol-hours", str(int_cfg.get("time_tolerance_hours", 2.0)),
                "--agg", int_cfg.get("agg", "max"),
                "--out-csv", str(int_out),
            ]
            # Optional prob column override
            if "prob_col" in int_cfg:
                int_cmd += ["--prob-col", str(int_cfg["prob_col"])]

            maybe_run(f"Intensity match (lead={lead})", int_out, int_cmd,
                      overwrite=(OW_ALR or bool(int_cfg.get("overwrite", False))))

        # denoise
        den_cfg = alrt_cfg.get("denoise", {})
        den_cmd = [
            sys.executable, "denoise_alerts.py",
            "--alerts", str(base_out),
            "--persist-hours", str(den_cfg.get("persist_hours", 1)),
            "--min-neighbors", str(den_cfg.get("min_neighbors", 3)),
            "--connectivity", str(den_cfg.get("connectivity", 4)),
            "--min-area", str(den_cfg.get("min_area", 0)),
            "--out", str(den_out),
        ]
        if den_cfg.get("sparse_output", False):
            den_cmd.append("--sparse-output")
        if den_cfg.get("overwrite", False) or OW_ALR:
            den_cmd.append("--overwrite")

        maybe_run(f"Denoise alerts (lead={lead})", den_out, den_cmd, overwrite=OW_ALR)

        # throttle
        keepq = keepq_map.get(lead, 0.90)
        thr_cmd2 = [
            sys.executable, "throttle_by_percentile.py",
            "--alerts", str(den_out),
            "--keep-quantile", str(keepq),
            "--only-alerts",
            "--flag-col", flag_col,
            "--out", str(top_out),
        ]
        maybe_run(f"Throttle alerts (lead={lead}, keep={keepq})", top_out, thr_cmd2, overwrite=OW_ALR)

        # eval alerts
        eval_cmd2 = [
            sys.executable, "eval_alert_hits.py",
            "--labelled", labelled,
            "--alerts", str(top_out),
            "--lead-hours", str(lead),
            "--flag-col", flag_col,
        ]
        print(f"\n== Evaluate alert hits (lead={lead}) ==")
        sh(eval_cmd2)

        summary_lines.append(f"- Lead {lead}h -> {top_out.name}")

    # 6) Summary + quick stats
    summary_md = cfg.get("summary_md", "results/summary_report.md")

    # Collect quick stats from throttled outputs
    stats_rows = []
    for lead in [int(x) for x in leads]:
        thr = per_lead_thr.get(lead)
        if thr is None:
            continue
        top_out = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}_throttled.csv.gz"
        if not top_out.exists():
            continue
        df_top = pd.read_csv(top_out, usecols=["time", "lat", "lon", flag_col])
        cov = df_top[flag_col].mean()
        active = int(df_top[flag_col].sum())
        total  = len(df_top)
        hours = pd.to_datetime(df_top["time"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h").nunique()
        stats_rows.append({
            "lead_h": lead,
            "threshold": per_lead_thr[lead],
            "file": top_out.name,
            "hours": hours,
            "active_cells": active,
            "total_cells": total,
            "coverage": cov,
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("lead_h").reset_index(drop=True)

    # 7) (Optional) Slow-tick diagnostics on throttled alerts
    st_cfg = cfg.get("slowtick", {})
    if st_cfg.get("enabled", True):
        st_out = Path(st_cfg.get("out_dir", "results/slowtick"))
        st_out.mkdir(parents=True, exist_ok=True)

        st_cmd = [
            sys.executable, "slowtick_diagnostics.py",
            "--alerts-dir", str(out_dir),
            "--run-name", run_name,
            "--leads", *[str(int(x)) for x in leads],
            "--flag-col", flag_col,
            "--out-dir", str(st_out),
        ]
        print("\n== Slow-tick diagnostics ==")
        sh(st_cmd)

        # Append a tiny summary into the markdown if present
        knee_csv = st_out / "knee_fit.csv"
        parity_csv = st_out / "parity_summary.csv"
        spec_csv = st_out / "spectrum_summary.csv"
        add_lines = []

        if knee_csv.exists():
            kdf = pd.read_csv(knee_csv)
            if not kdf.empty:
                p, lo, hi = kdf.loc[0, ["p","p_lo","p_hi"]]
                add_lines.append(f"- **Knee law**: p = {p:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")

        if parity_csv.exists():
            pdf = pd.read_csv(parity_csv).sort_values("lead_h")
            if not pdf.empty:
                s = ", ".join(f"{int(r.lead_h)}h: Δcov={r.mean_diff:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]"
                              for _, r in pdf.iterrows())
                add_lines.append(f"- **Parity (N−S)**: {s}")

        if spec_csv.exists():
            sdf = pd.read_csv(spec_csv).sort_values("lead_h")
            if not sdf.empty:
                s = ", ".join(f"{int(r.lead_h)}h: f*={r.peak_freq:.3f} cph"
                              for _, r in sdf.iterrows() if not pd.isna(r.peak_freq))
                if s:
                    add_lines.append(f"- **Slow-tick ridge**: {s}")

        if add_lines:
            Path(summary_md).parent.mkdir(parents=True, exist_ok=True)
            with open(summary_md, "a", encoding="utf-8") as f:
                f.write("\n\n## Slow-tick diagnostics\n")
                for line in add_lines:
                    f.write(f"{line}\n")

    # Diagnostics across leads
    diag_rows = []
    for lead in [int(x) for x in leads]:
        thr = per_lead_thr.get(lead)
        if thr is None:
            continue
        base_out = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}.csv.gz"
        den_out  = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}_denoised.csv.gz"
        top_out  = out_dir / f"alerts_{run_name}_lead{lead}_thr{thr:.4f}_throttled.csv.gz"
        d = collect_alert_diagnostics(base_out, den_out, top_out, flag_col=flag_col)
        d["lead_h"] = lead; d["threshold"] = thr
        diag_rows.append(d)
    diag_df = pd.DataFrame(diag_rows).sort_values("lead_h").reset_index(drop=True)
    diag_csv = Path(cfg.get("diagnostics_csv", "results/diagnostics.csv"))
    diag_csv.parent.mkdir(parents=True, exist_ok=True)
    if not diag_df.empty:
        if OW_RPT and diag_csv.exists():
            _safe_unlink(diag_csv)
        diag_df.to_csv(diag_csv, index=False)

    Path(summary_md).parent.mkdir(parents=True, exist_ok=True)
    # Write markdown (UTF-8 to avoid Windows cp1252 issues)
    if OW_RPT and Path(summary_md).exists():
        _safe_unlink(summary_md)

    Path(summary_md).write_text(textwrap.dedent(f"""
        # Pipeline summary – {run_name}

        **Labelled:** `{labelled}`  
        **Model:** `{model_out_cal}`

        **Leads:** {', '.join(leads)}  
        **Thresholds column used:** {use_col} (from `{thr_csv}`)

        ## Outputs
        {os.linesep.join(f"- Lead {r['lead_h']}h -> {r['file']}" for _, r in stats_df.iterrows()) or "- (none)"}

        ## Coverage (post-throttle)
        | Lead (h) | Threshold | Hours | Active cells | Total cells | Coverage |
        |---------:|----------:|------:|-------------:|------------:|---------:|
        {os.linesep.join(
            f"| {int(r.lead_h):>7} | {r.threshold:>9.4f} | {int(r.hours):>5} | {int(r.active_cells):>12,} | {int(r.total_cells):>11,} | {r.coverage:>7.3f} |"
            for _, r in stats_df.iterrows()
        ) or "|  – | – | – | – | – | – |"}

        ## Pipeline Diagnostics
        {("(see `" + str(diag_csv) + "` for CSV)") if not diag_df.empty else "_No diagnostics available._"}
    """).strip()+"\n", encoding="utf-8")

    # Append a compact diagnostics table to the MD file
    if not diag_df.empty:
        with open(summary_md, "a", encoding="utf-8") as f:
            f.write("\n\n| Lead | thr | base_rows | base_bad_time | base_cov | den_cov | thr_cov | keep_denoise | keep_throttle | flags |\n")
            f.write("|----:|----:|----------:|-------------:|---------:|--------:|--------:|------------:|--------------:|:------|\n")
            for _, r in diag_df.iterrows():
                f.write(
                    f"| {int(r['lead_h'])} | {r['threshold']:.4f} | {int(r.get('base_rows',0))} | {int(r.get('base_bad_time',0))} | "
                    f"{(r.get('base_cov',math.nan)):.3f} | {(r.get('den_cov',math.nan)):.3f} | {(r.get('thr_cov',math.nan)):.3f} | "
                    f"{(r.get('retained_after_denoise',math.nan)):.3f} | {(r.get('retained_after_throttle',math.nan)):.3f} | "
                    f"{r.get('flags','')} |\n"
                )

    # Also write a plain-text version
    summary_txt = Path(summary_md).with_suffix(".txt")
    if OW_RPT and summary_txt.exists():
        _safe_unlink(summary_txt)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Pipeline summary - {run_name}\n\n")
        f.write(f"Labelled: {labelled}\nModel: {model_out_cal}\n\n")
        f.write(f"Leads: {', '.join(leads)}\nThresholds file: {thr_csv} (col={use_col})\n\n")
        if not stats_df.empty:
            f.write("Coverage (post-throttle):\n")
            for _, r in stats_df.iterrows():
                f.write(f"  Lead {int(r.lead_h)}h  thr={r.threshold:.4f}  hours={int(r.hours)}  "
                        f"active={int(r.active_cells):,}/{int(r.total_cells):,}  cov={r.coverage:.3f}\n")
        else:
            f.write("No throttled outputs found.\n")
        if not diag_df.empty:
            f.write("\nDiagnostics:\n")
            for _, r in diag_df.iterrows():
                f.write(
                    f"  Lead {int(r['lead_h'])}h thr={r['threshold']:.4f} "
                    f"| base_rows={int(r.get('base_rows',0))}, base_bad_time={int(r.get('base_bad_time',0))}, "
                    f"base_cov={r.get('base_cov',math.nan):.3f}, den_cov={r.get('den_cov',math.nan):.3f}, thr_cov={r.get('thr_cov',math.nan):.3f}, "
                    f"keep_denoise={r.get('retained_after_denoise',math.nan):.3f}, keep_throttle={r.get('retained_after_throttle',math.nan):.3f} "
                    f"| flags: {r.get('flags','')}\n"
                )

    # Optional PDF with quick plots
    pdf_path = cfg.get("report_pdf", None)
    if pdf_path:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
            if OW_RPT and Path(pdf_path).exists():
                _safe_unlink(pdf_path)

            with PdfPages(pdf_path) as pdf:
                # Figure 1: Coverage by lead
                plt.figure(figsize=(8, 4.5))
                plt.title(f"Coverage by Lead - {run_name}")
                if not stats_df.empty:
                    plt.bar(stats_df["lead_h"].astype(int), stats_df["coverage"])
                    plt.xlabel("Lead (hours)")
                    plt.ylabel("Coverage (fraction of cells active)")
                    plt.xticks(stats_df["lead_h"].astype(int))
                else:
                    plt.text(0.5, 0.5, "No outputs", ha="center", va="center")
                pdf.savefig(bbox_inches="tight")
                plt.close()

                # Figure 2: Alerts per hour for the last lead (if available)
                if not stats_df.empty:
                    last_row = stats_df.iloc[-1]
                    last_file = Path(cfg["alerts"]["out_dir"]) / last_row["file"]
                    if last_file.exists():
                        fcol = cfg["alerts"].get("flag_col","alert_final")
                        dfl = pd.read_csv(last_file, usecols=["time", fcol])
                        tt = pd.to_datetime(dfl["time"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")
                        by_hour = dfl.assign(t=tt).groupby("t")[fcol].sum()
                        plt.figure(figsize=(8, 4.5))
                        plt.title(f"Active cells per hour - lead {int(last_row['lead_h'])}h")
                        if len(by_hour) > 0:
                            plt.plot(by_hour.index, by_hour.values)
                            plt.xlabel("Time")
                            plt.ylabel("Active cells")
                        else:
                            plt.text(0.5, 0.5, "No data", ha="center", va="center")
                        pdf.savefig(bbox_inches="tight")
                        plt.close()

            print(f"\nReport saved: {pdf_path}")
        except Exception as e:
            print(f"[report] Skipped PDF (matplotlib not available or error: {e})")

    print(f"\nDone.\n- Markdown -> {summary_md}\n- Text     -> {summary_txt}\n" + (f"- PDF      -> {pdf_path}\n" if pdf_path else ""))


if __name__ == "__main__":
    main()