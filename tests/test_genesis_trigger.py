"""
tests/test_genesis_trigger.py — synthetic end-to-end test of the pipeline
genesis-trigger step. No real data required.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "alerts_logic_subprocess" / "genesis_trigger.py"


def _make_labelled(path: Path, n_cells=40, n_hours=400, seed=3):
    """Small synthetic labelled grid: half the cells 'tighten' (favourable
    features + near_storm=1 late), half fizzle. All cells pregen==1 plus a
    band of non-spiral background cells."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-02-01", periods=n_hours, freq="h")
    rows = []
    for c in range(n_cells):
        tighten = c < n_cells // 2
        ilat, ilon = divmod(c, 8)
        lat, lon = -20.0 + ilat * 0.25, 130.0 + ilon * 0.25
        # favourable == low shear_quench / low msl_nd for tightens
        base_sq = 0.3 if tighten else 0.8
        base_msl = -0.8 if tighten else 0.2
        for i, t in enumerate(times):
            rows.append({
                "time": t, "lat": lat, "lon": lon, "ilat": ilat, "ilon": ilon,
                "gka_shear_quench": base_sq + rng.normal(0, 0.05),
                "gka_msl_nd": base_msl + rng.normal(0, 0.1),
                "gka_SII": (0.5 if tighten else 0.3) + rng.normal(0, 0.05),
                "gka_knee_ratio": 1.0 + rng.normal(0, 0.1),
                "pregen": 1,
                "near_storm": int(tighten),
                "t_to_storm_min_h": float(max(0, n_hours - i)) if tighten else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def test_genesis_trigger_end_to_end(tmp_path):
    labelled = tmp_path / "labelled.parquet"
    out = tmp_path / "alerts.parquet"
    _make_labelled(labelled)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--labelled", str(labelled), "--out", str(out),
         "--train-end", "2025-02-09",
         "--stripes", "2", "--false-alarm", "0.10",
         "--max-train-rows", "50000"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"trigger failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists(), "alerts parquet not written"

    alerts = pd.read_parquet(out)
    # schema contract for seeds.from_alerts
    for col in ("time", "lat", "lon", "prob_genesis", "alert_genesis"):
        assert col in alerts.columns, f"missing column {col}"

    # planted signal: tighten cells must alert far more than fizzle cells
    tight_rate = alerts.loc[alerts.near_storm == 1, "alert_genesis"].mean()
    fiz_rate = alerts.loc[alerts.near_storm == 0, "alert_genesis"].mean()
    assert tight_rate > 0.5, f"tighten cells under-alerted ({tight_rate:.2f})"
    assert fiz_rate < 0.25, f"fizzle false-alarm too high ({fiz_rate:.2f})"

    # model card written and self-consistent
    card = json.loads((tmp_path / "alerts.parquet.model_card.json").read_text())
    assert card["trigger_feature"] == "acc48"
    assert len(card["w"]) == len(card["features"]) + 1  # bias + coefs
    assert np.isfinite(card["threshold"])
