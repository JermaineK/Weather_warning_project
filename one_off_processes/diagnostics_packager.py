#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diagnostics_packager.py — sweep pipeline artifacts, compute light diagnostics,
and emit: (1) a Markdown summary, (2) a machine-readable JSON manifest,
(3) an optional one-page PDF of quick plots, and (4) a CSV of per-lead alert stats.

It tolerates missing pieces and keeps going.

Examples
--------
python data_subprocess/diagnostics_packager.py ^
  --run-name coral_sea_demo ^
  --reports-dir results/reports ^
  --alerts-dir results/alerts ^
  --models-dir results/models ^
  --seedmaps-dir results/seedmaps ^
  --eval-csv models/eval.csv ^
  --thresholds-csv results/best_fbeta_thresholds.csv ^
  --labels-csv data/tracks/tracks_subset.csv ^
  --labels-time-col time --labels-lat-col lat --labels-lon-col lon --labels-flag-col event ^
  --write-pdf --write-zip
"""

from __future__ import annotations
import argparse, json, math, re, sys, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib only if --write-pdf is passed
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# --------------------------- utils ---------------------------

def _exists(p: Path | str) -> bool:
    return Path(p).exists()

def _maybe_read_csv(path: Path, **kw):
    if not _exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        try:
            return pd.read_csv(path, compression="infer", low_memory=False, **kw)
        except Exception:
            return pd.DataFrame()

def _maybe_read_parquet(path: Path, columns=None):
    if not _exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.DataFrame()

def _glob(paths: List[str]) -> List[Path]:
    out = []
    for pat in paths:
        for p in Path().glob(pat):
            out.append(p)
    return sorted(set(out))

def _to_utc_naive(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_convert(None)

def _fmt_float(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


# --------------------------- schema ---------------------------

@dataclass
class LeadStats:
    lead: int
    n_rows: int
    n_alerts: int
    alert_rate: float

@dataclass
class Diagnostics:
    run_name: str
    alerts_dir: str
    models_dir: str
    seedmaps_dir: str
    reports_dir: str
    eval_csv: Optional[str]
    thresholds_csv: Optional[str]
    labels_csv: Optional[str]
    leads: List[int]
    per_lead: List[LeadStats]
    auc_mean: Optional[float]
    ap_mean: Optional[float]
    brier_mean: Optional[float]
    logloss_mean: Optional[float]
    f1_best_mean: Optional[float]
    seed_summary_path: Optional[str]
    seed_summary_preview: Optional[str]


# --------------------------- core collectors ---------------------------

def collect_per_lead_alert_stats(alerts_dir: Path, run_name: str) -> Tuple[List[LeadStats], List[Path]]:
    """
    Scans alerts_dir for files like alerts_<run_name>_leadXX_thr*.csv.gz and/or *_base.csv.gz,
    computes per-file n_rows + n_alerts (if flag col present), and aggregates by lead.
    """
    base_pat = str(alerts_dir / f"alerts_{run_name}_lead*_base.csv.gz")
    thr_pat  = str(alerts_dir / f"alerts_{run_name}_lead*_thr*.csv.gz")
    files = _glob([base_pat, thr_pat])
    by_lead: Dict[int, Dict[str, int]] = {}

    lead_re = re.compile(rf"alerts_{re.escape(run_name)}_lead(\d+)_")
    for fp in files:
        m = lead_re.search(fp.name)
        if not m:
            continue
        L = _safe_int(m.group(1))
        if L is None:
            continue
        df = _maybe_read_csv(fp)
        if df.empty:
            n_rows = 0
            n_alerts = 0
        else:
            n_rows = len(df)
            flag_col = None
            for c in ["alert_final", "flag", "is_event", "label"]:
                if c in df.columns:
                    flag_col = c
                    break
            if flag_col:
                v = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)
                n_alerts = int(v.sum())
            else:
                n_alerts = 0
        cell = by_lead.setdefault(L, {"rows": 0, "alerts": 0})
        cell["rows"] += n_rows
        cell["alerts"] += n_alerts

    stats: List[LeadStats] = []
    for L, d in sorted(by_lead.items()):
        rows = d["rows"]
        alerts = d["alerts"]
        rate = (alerts / rows) if rows > 0 else 0.0
        stats.append(LeadStats(lead=int(L), n_rows=rows, n_alerts=alerts, alert_rate=rate))

    return stats, files

def collect_eval_metrics(eval_csv: Optional[Path]) -> Dict[str, float]:
    if not eval_csv or not _exists(eval_csv):
        return {"auc": None, "ap": None, "brier": None, "logloss": None, "f1_best": None}
    df = _maybe_read_csv(eval_csv)
    if df.empty:
        return {"auc": None, "ap": None, "brier": None, "logloss": None, "f1_best": None}
    agg = {}
    for k in ["auc", "ap", "brier", "logloss", "f1_best"]:
        if k in df.columns:
            col = pd.to_numeric(df[k], errors="coerce").dropna()
            agg[k] = float(col.mean()) if not col.empty else None
        else:
            # support *_cal columns if present but raw missing
            k_cal = f"{k}_cal"
            if k_cal in df.columns:
                col = pd.to_numeric(df[k_cal], errors="coerce").dropna()
                agg[k] = float(col.mean()) if not col.empty else None
            else:
                agg[k] = None
    return agg

def parse_seed_summary(seedmaps_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    # Prefer seed_summary.txt; fallback any *_seed_summary*.txt
    pref = seedmaps_dir / "seed_summary.txt"
    if _exists(pref):
        txt = pref.read_text(encoding="utf-8", errors="ignore")
        return pref, _summarize_seed_txt(txt)
    candidates = list(seedmaps_dir.glob("*seed_summary*.txt"))
    if candidates:
        txt = candidates[0].read_text(encoding="utf-8", errors="ignore")
        return candidates[0], _summarize_seed_txt(txt)
    return None, None

def _summarize_seed_txt(txt: str) -> Optional[str]:
    """
    Extract a few salient lines: Unique pts, Hours, Seed starts, Seed patches, Matched patches.
    """
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    keys = ["Unique pts", "Hours", "Seed starts (points)", "Seed patches", "Matched patches"]
    found = []
    for k in keys:
        for ln in lines:
            if ln.lower().startswith(k.lower()):
                found.append(ln)
                break
    return " | ".join(found[:5]) if found else None


# --------------------------- labels join (optional quick precision) ---------------------------

def quick_precision_estimate(alert_files: List[Path],
                             labels_csv: Optional[Path],
                             tcol="time", ycol="event",
                             latc="lat", lonc="lon",
                             flagc="alert_final") -> Optional[float]:
    """
    Coarse precision estimate: for a sample of alert files, join on exact hour + cell and compute P.
    Only if labels file provided and reasonably small.
    """
    if not labels_csv or not _exists(labels_csv):
        return None

    # Load labels (CSV or Parquet)
    lab = None
    low = labels_csv.name.lower()
    try:
        if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
            lab = pd.read_parquet(labels_csv)
        else:
            lab = pd.read_csv(labels_csv, compression="infer", low_memory=False)
    except Exception:
        return None
    if lab is None or lab.empty:
        return None

    # Normalize
    for c in [tcol, latc, lonc, ycol]:
        if c not in lab.columns:
            return None
    lab = lab[[tcol, latc, lonc, ycol]].copy()
    lab[tcol] = _to_utc_naive(lab[tcol])
    lab[tcol] = lab[tcol].dt.floor("H")
    lab[ycol] = pd.to_numeric(lab[ycol], errors="coerce").fillna(0).astype(int)

    # Take up to 3 alert files to keep it light
    sample_files = alert_files[:3]
    if not sample_files:
        return None

    prs = []
    for fp in sample_files:
        df = _maybe_read_csv(fp)
        if df.empty:
            continue
        # pick a plausible flag column if not present
        fcol = flagc if flagc in df.columns else next(
            (c for c in ["flag", "is_event", "label"] if c in df.columns),
            None,
        )
        if fcol is None or "time" not in df.columns or "lat" not in df.columns or "lon" not in df.columns:
            continue

        tmp = df.copy()
        tmp["time"] = _to_utc_naive(tmp["time"]).dt.floor("H")
        tmp = tmp[["time", "lat", "lon", fcol]]
        tmp[fcol] = pd.to_numeric(tmp[fcol], errors="coerce").fillna(0).astype(int)
        tmp = tmp[tmp[fcol] == 1]

        if tmp.empty:
            continue

        m = tmp.merge(lab, left_on=["time", "lat", "lon"], right_on=[tcol, latc, lonc], how="left")
        hit = pd.to_numeric(m[ycol], errors="coerce").fillna(0).astype(int)
        if len(m) == 0:
            continue
        prs.append(float(hit.mean()))
    if not prs:
        return None
    return float(np.mean(prs))


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Package diagnostics and summaries from pipeline artifacts.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--reports-dir", default="results/reports")
    ap.add_argument("--alerts-dir", default="results/alerts")
    ap.add_argument("--models-dir", default="results/models")
    ap.add_argument("--seedmaps-dir", default="results/seedmaps")

    ap.add_argument("--eval-csv", default=None)
    ap.add_argument("--thresholds-csv", default=None)

    # Optional labels for coarse precision estimate
    ap.add_argument("--labels-csv", default=None)
    ap.add_argument("--labels-time-col", default="time")
    ap.add_argument("--labels-lat-col", default="lat")
    ap.add_argument("--labels-lon-col", default="lon")
    ap.add_argument("--labels-flag-col", default="event")

    ap.add_argument("--write-pdf", action="store_true")
    ap.add_argument("--write-zip", action="store_true")

    args = ap.parse_args()

    run = args.run_name
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = Path(args.alerts_dir)
    models_dir = Path(args.models_dir)
    seed_dir = Path(args.seedmaps_dir)

    # 1) Alerts sweep
    lead_stats, alert_files = collect_per_lead_alert_stats(alerts_dir, run)
    leads = [s.lead for s in lead_stats]

    # 2) Model eval
    eval_metrics = collect_eval_metrics(Path(args.eval_csv) if args.eval_csv else None)

    # 3) Seed summary (text)
    seed_summary_path, seed_summary_preview = parse_seed_summary(seed_dir)

    # 4) Quick precision estimate (optional)
    prec_est = quick_precision_estimate(
        alert_files,
        Path(args.labels_csv) if args.labels_csv else None,
        tcol=args.labels_time_col,
        ycol=args.labels_flag_col,
        latc=args.labels_lat_col,
        lonc=args.labels_lon_col,
    )

    # 5) Thresholds snapshot (optional small table for MD)
    thr_df = _maybe_read_csv(Path(args.thresholds_csv)) if args.thresholds_csv else pd.DataFrame()
    thr_preview = ""
    if not thr_df.empty:
        # small preview of first 10 rows/cols
        keep = [c for c in thr_df.columns[:5]]
        thr_preview = thr_df[keep].head(10).to_csv(index=False)

    # 6) Build manifest
    diag = Diagnostics(
        run_name=run,
        alerts_dir=str(alerts_dir),
        models_dir=str(models_dir),
        seedmaps_dir=str(seed_dir),
        reports_dir=str(reports_dir),
        eval_csv=args.eval_csv,
        thresholds_csv=args.thresholds_csv,
        labels_csv=args.labels_csv,
        leads=leads,
        per_lead=lead_stats,
        auc_mean=eval_metrics["auc"],
        ap_mean=eval_metrics["ap"],
        brier_mean=eval_metrics["brier"],
        logloss_mean=eval_metrics["logloss"],
        f1_best_mean=eval_metrics["f1_best"],
        seed_summary_path=(str(seed_summary_path) if seed_summary_path else None),
        seed_summary_preview=seed_summary_preview,
    )

    manifest_path = reports_dir / f"manifest_{run}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({**asdict(diag), "precision_estimate": prec_est}, f, indent=2)

    # 7) Per-lead CSV
    perlead_path = reports_dir / f"alerts_perlead_{run}.csv"
    if lead_stats:
        rows = [
            {
                "lead": s.lead,
                "n_rows": s.n_rows,
                "n_alerts": s.n_alerts,
                "alert_rate": s.alert_rate,
            }
            for s in lead_stats
        ]
        pd.DataFrame(rows).to_csv(perlead_path, index=False)

    # 8) Markdown report (compact)
    md_path = reports_dir / f"diagnostics_{run}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Diagnostics – {run}\n\n")
        f.write("## High-level\n")
        f.write(f"- Alerts dir: `{alerts_dir}`\n")
        f.write(f"- Models dir: `{models_dir}`\n")
        f.write(f"- Seedmaps dir: `{seed_dir}`\n")
        f.write(f"- Eval CSV: `{args.eval_csv or '(none)'}`\n")
        f.write(f"- Thresholds CSV: `{args.thresholds_csv or '(none)'}`\n")
        f.write(f"- Labels (for quick P): `{args.labels_csv or '(none)'}`\n\n")

        f.write("## Model metrics (mean across leads if applicable)\n")
        f.write(f"- AUC: {_fmt_float(diag.auc_mean)}\n")
        f.write(f"- AP : {_fmt_float(diag.ap_mean)}\n")
        f.write(f"- Brier: {_fmt_float(diag.brier_mean)}\n")
        f.write(f"- LogLoss: {_fmt_float(diag.logloss_mean)}\n")
        f.write(f"- F1* (best on PR curve): {_fmt_float(diag.f1_best_mean)}\n\n")

        if prec_est is not None:
            f.write(f"**Quick precision estimate (small sample join)**: {_fmt_float(prec_est)}\n\n")

        f.write("## Alerts per lead\n\n")
        if lead_stats:
            f.write("| lead | rows | alerts | rate |\n|---:|---:|---:|---:|\n")
            for s in lead_stats:
                f.write(f"| {s.lead} | {s.n_rows:,} | {s.n_alerts:,} | {_fmt_float(s.alert_rate)} |\n")
            f.write("\n")
        else:
            f.write("_No alert files found._\n\n")

        f.write("## Seeds & tracks\n\n")
        if seed_summary_preview:
            f.write(f"{seed_summary_preview}\n\n")
        else:
            f.write("_No seed summary found._\n\n")

        if thr_preview:
            f.write("## Thresholds preview (first 10 rows / cols <=5)\n\n")
            f.write("```\n")
            f.write(thr_preview)
            f.write("```\n")

        f.write("\n---\n")
        f.write(f"_Manifest_: `{manifest_path}`\n")

    # 9) Optional 1-page PDF
    pdf_path = None
    if args.write_pdf:
        if not _HAVE_MPL:
            print("[warn] matplotlib not available; skipping PDF.")
        else:
            pdf_path = reports_dir / f"diagnostics_{run}.pdf"
            with PdfPages(pdf_path) as pdf:
                # Plot alert_rate by lead if present
                if lead_stats:
                    ld = pd.DataFrame([asdict(s) for s in lead_stats]).sort_values("lead")
                    plt.figure(figsize=(8, 4.0))
                    plt.title("Alert rate by lead")
                    plt.plot(ld["lead"].to_numpy(), ld["alert_rate"].to_numpy(), marker="o")
                    plt.xlabel("Lead (h)")
                    plt.ylabel("Alert rate")
                    pdf.savefig(bbox_inches="tight")
                    plt.close()

                # Basic table-like text page for metrics
                plt.figure(figsize=(8, 4.0))
                plt.axis("off")
                txt = (
                    f"AUC mean: {_fmt_float(diag.auc_mean)}\n"
                    f"AP mean: {_fmt_float(diag.ap_mean)}\n"
                    f"Brier mean: {_fmt_float(diag.brier_mean)}\n"
                    f"LogLoss mean: {_fmt_float(diag.logloss_mean)}\n"
                    f"F1* mean: {_fmt_float(diag.f1_best_mean)}"
                )
                plt.text(0.02, 0.98, txt, va="top", ha="left", family="monospace")
                pdf.savefig(bbox_inches="tight")
                plt.close()

    # 10) Optional zip bundle of key outputs
    zip_path = None
    if args.write_zip:
        zip_path = reports_dir / f"diagnostics_{run}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            def _try_add(p: Path, arcname=None):
                if _exists(p):
                    z.write(p, arcname=str(arcname or p.name))
            _try_add(md_path)
            _try_add(manifest_path)
            if pdf_path:
                _try_add(pdf_path)
            if _exists(perlead_path):
                _try_add(perlead_path)

    # Console footer
    print("[done] diagnostics package")
    print(f"- Markdown : {md_path}")
    print(f"- Manifest : {manifest_path}")
    if _exists(perlead_path):
        print(f"- Per-lead : {perlead_path}")
    if pdf_path and _exists(pdf_path):
        print(f"- PDF      : {pdf_path}")
    if zip_path and _exists(zip_path):
        print(f"- ZIP      : {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass