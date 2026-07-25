#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geomval_seasons.py — shared multi-season loading + cross-validation blocking
for the genesis analyses.

Why
    The single-season results (LOSO 0.739, the 24-48h accumulation advantage,
    the shape filter, the 0.854 precursor score) rest on 8-10 storms from ONE
    season. With per-season slim files (data/genesis_<year>_slim.parquet) the
    same analyses can run across 2021-2025 (~30 storms).

Blocking
    Storms inside one season share the large-scale environment, so plain
    leave-one-storm-out can leak across storms of the same year. This module
    therefore supports two CV modes:

      cv="season"  (default, strict)  fit on all OTHER seasons; score every
                                      storm of the held-out season.
      cv="storm"                      fit on all other storms regardless of
                                      season (looser, more training data).

Files are given as a glob or comma-separated list, e.g.
    --panels "data/genesis_*_slim.parquet"
    --tracks "data/tracks/tracks_2021.parquet,...,data/tracks/tracks_geomval.parquet"
or simply --tracks "data/tracks/tracks_*.parquet" (auto-matched by season year).
"""
from __future__ import annotations

import glob as _glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_files(spec: str) -> list[str]:
    """Glob and/or comma-list -> sorted unique existing paths."""
    out: list[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        hits = sorted(_glob.glob(part))
        out.extend(hits if hits else ([part] if Path(part).exists() else []))
    seen, uniq = set(), []
    for p in out:
        rp = str(Path(p))
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def _season_of_path(p: str) -> int | None:
    m = re.search(r"(20\d\d)", Path(p).name)
    return int(m.group(1)) if m else None


def read_tracks(paths: list[str]) -> pd.DataFrame:
    """Concatenate track files, tagging each row with its season year."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p) if p.lower().endswith((".parquet", ".parq", ".pq")) \
            else pd.read_csv(p, low_memory=False)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        for c in ("lat", "lon"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["time", "lat", "lon"])
        if df.empty:
            continue
        df["season"] = df["time"].dt.year
        frames.append(df)
    if not frames:
        raise SystemExit("[seasons] no usable track files.")
    tr = pd.concat(frames, ignore_index=True)
    # a storm may straddle a year boundary; pin it to its first observation
    first_season = tr.groupby(tr["storm_id"].astype(str))["season"].transform("min")
    tr["season"] = first_season
    return tr


def load_storm_crops(panel_specs: str, track_specs: str, feats: list[str],
                     extra_cols: list[str] | None = None,
                     pre_h: float = 120.0, pad_deg: float = 6.0,
                     max_cells: int = 4000, min_points: int = 3,
                     verbose: bool = True) -> dict[str, dict]:
    """Return {storm_key: {"df":DataFrame, "season":int, "name":str}}.

    Each season's panel file is opened once and cropped per storm of that season,
    so a five-season run touches each file exactly once.
    """
    panels = resolve_files(panel_specs)
    tracks = resolve_files(track_specs)
    if not panels:
        raise SystemExit(f"[seasons] no panel files matched: {panel_specs}")
    tr = read_tracks(tracks)

    by_season: dict[int, str] = {}
    for p in panels:
        s = _season_of_path(p)
        if s is not None:
            by_season[s] = p
    if verbose:
        print(f"[seasons] panels: { {k: Path(v).name for k, v in sorted(by_season.items())} }")

    need = list(dict.fromkeys(
        ["time", "lat", "lon", "ilat", "ilon", "pregen", "near_storm", "t_to_storm_min_h"]
        + list(feats) + list(extra_cols or [])))

    rng = np.random.default_rng(0)
    crops: dict[str, dict] = {}
    for season, panel in sorted(by_season.items()):
        st = tr[tr["season"] == season]
        if st.empty:
            if verbose:
                print(f"[seasons] {season}: no storms in tracks; skipping")
            continue
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(panel).schema.names)
        cols = [c for c in need if c in have]
        missing = [c for c in need if c not in have]
        if missing and verbose:
            print(f"[seasons] {season}: missing cols {missing}")
        panel_df = pd.read_parquet(panel, columns=cols)
        panel_df["time"] = pd.to_datetime(panel_df["time"])
        for sid, g in st.groupby(st["storm_id"].astype(str)):
            g = g.sort_values("time")
            if len(g) < min_points:
                continue
            name = str(g["name"].iloc[0]) if "name" in g.columns else sid
            t0 = g["time"].min() - pd.Timedelta(hours=pre_h)
            t1 = g["time"].max() + pd.Timedelta(hours=6)
            m = panel_df[(panel_df["time"] >= t0) & (panel_df["time"] < t1)]
            m = m[(m["lat"].between(g["lat"].min() - pad_deg, g["lat"].max() + pad_deg))
                  & (m["lon"].between(g["lon"].min() - pad_deg, g["lon"].max() + pad_deg))]
            if m.empty:
                continue
            cells = m[["ilat", "ilon"]].drop_duplicates()
            if len(cells) > max_cells:
                cells = cells.iloc[rng.choice(len(cells), max_cells, replace=False)]
                m = m.merge(cells, on=["ilat", "ilon"], how="inner")
            key = f"{season}:{name}"
            i = 1
            while key in crops:
                i += 1
                key = f"{season}:{name}#{i}"
            crops[key] = {"df": m.copy(), "season": season, "name": name}
            if verbose:
                print(f"[seasons] {key}: rows={len(m):,}")
        del panel_df
    if not crops:
        raise SystemExit("[seasons] no storm crops built.")
    if verbose:
        seasons = sorted({v['season'] for v in crops.values()})
        print(f"[seasons] {len(crops)} storms across {len(seasons)} seasons {seasons}")
    return crops


def train_keys_for(crops: dict[str, dict], held_key: str, cv: str = "season") -> list[str]:
    """Keys to TRAIN on when `held_key` is the test storm."""
    if cv == "storm":
        return [k for k in crops if k != held_key]
    held_season = crops[held_key]["season"]
    return [k for k in crops if crops[k]["season"] != held_season]


# ---------------------------------------------------------------------------
# Strict genesis labelling
# ---------------------------------------------------------------------------
# `near_storm` is generous: any cell within ~5 deg / 12h of ANY track point of a
# tracked system, including a fully mature storm. That inflates apparent skill
# and lets "lead 0" mean "inside a hurricane" rather than "about to form".
#
# The strict label instead targets the GENESIS EVENT itself:
#     genesis := first track time at which vmax >= thresh_kt (default 34 kt, TS)
#     positive := spiral cell within radius_deg of the GENESIS POINT,
#                 at a time in [t_g - max_lead_h, t_g)      (strictly BEFORE)
#     negative := spiral cell that is not positive for any storm AND has
#                 near_storm == 0  (i.e. never part of a tracked system) -> fizzle
#     excluded := post-genesis cells, cells near a mature storm, and storms that
#                 never reached TS intensity (ambiguous outcome)
#
# This removes the mature-storm inflation and makes "lead" mean time-to-formation.

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def genesis_events(tr: pd.DataFrame, thresh_kt: float = 34.0) -> pd.DataFrame:
    """First TS-intensity crossing per storm: (storm_id, name, season, t_g, lat_g, lon_g)."""
    rows = []
    for sid, g in tr.groupby(tr["storm_id"].astype(str)):
        g = g.sort_values("time")
        v = pd.to_numeric(g.get("vmax"), errors="coerce")
        if v is None or not (v >= thresh_kt).any():
            continue
        i = (v >= thresh_kt).idxmax()
        r = g.loc[i]
        rows.append({
            "storm_id": sid,
            "name": str(r["name"]) if "name" in g.columns else sid,
            "season": int(r["season"]) if "season" in g.columns else int(r["time"].year),
            "t_g": r["time"], "lat_g": float(r["lat"]), "lon_g": float(r["lon"]),
        })
    return pd.DataFrame(rows)


def add_strict_labels(df: pd.DataFrame, events: pd.DataFrame,
                      radius_deg: float = 3.0, max_lead_h: float = 72.0):
    """Add gen_pos / gen_lead_h / gen_storm. Returns (df, n_pos).

    A cell may fall in more than one genesis window; the nearest-in-time one wins.
    """
    n = len(df)
    gen_pos = np.zeros(n, dtype=np.int8)
    gen_lead = np.full(n, np.nan)
    gen_storm = np.array([""] * n, dtype=object)
    if events.empty:
        return df.assign(gen_pos=gen_pos, gen_lead_h=gen_lead, gen_storm=gen_storm), 0

    t = pd.to_datetime(df["time"]).to_numpy()
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
    radius_km = radius_deg * 111.32

    for ev in events.itertuples(index=False):
        lead = (np.datetime64(ev.t_g) - t) / np.timedelta64(1, "h")
        in_time = (lead > 0) & (lead <= max_lead_h)
        if not in_time.any():
            continue
        d = np.full(n, np.inf)
        d[in_time] = haversine_km(lat[in_time], lon[in_time], ev.lat_g, ev.lon_g)
        hit = in_time & (d <= radius_km)
        if not hit.any():
            continue
        # nearest-in-time genesis wins where windows overlap
        better = hit & (np.isnan(gen_lead) | (lead < gen_lead))
        gen_pos[better] = 1
        gen_lead[better] = lead[better]
        gen_storm[better] = f"{ev.season}:{ev.name}"

    out = df.assign(gen_pos=gen_pos, gen_lead_h=gen_lead, gen_storm=gen_storm)
    return out, int(gen_pos.sum())
