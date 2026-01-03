#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
animate_frames.py
Stitch a sequence of PNG/JPG frames into a GIF or MP4.
"""

from __future__ import annotations

# Agent: add a lightweight frame-to-animation utility for report maps.

import argparse
import glob
from pathlib import Path
from typing import List


def _collect_frames(patterns: List[str]) -> List[Path]:
    frames: List[Path] = []
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            fp = Path(p)
            key = str(fp.resolve()) if fp.exists() else str(fp)
            if key in seen:
                continue
            seen.add(key)
            frames.append(fp)
    frames = [p for p in frames if p.exists()]
    frames.sort(key=lambda x: str(x))
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a GIF/MP4 from frame images.")
    ap.add_argument("--frames", nargs="+", required=True, help="Glob patterns for frames (e.g., maps/seed_*_YYYYMMDDHH.png).")
    ap.add_argument("--out", required=True, help="Output path (.gif or .mp4).")
    ap.add_argument("--fps", type=float, default=6.0, help="Frames per second.")
    ap.add_argument("--loop", type=int, default=0, help="GIF loop count (0 = infinite).")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 disables).")
    ap.add_argument("--reverse", action="store_true", help="Reverse frame order.")
    args = ap.parse_args()

    frames = _collect_frames(args.frames)
    if not frames:
        raise SystemExit("[animate] no frames matched.")
    if args.reverse:
        frames = list(reversed(frames))
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        raise SystemExit(f"[animate] imageio not available: {exc}. Try: pip install imageio")

    if ext == ".gif":
        duration = 1.0 / max(float(args.fps), 0.1)
        writer = imageio.get_writer(str(out_path), mode="I", duration=duration, loop=int(args.loop))
    elif ext == ".mp4":
        writer = imageio.get_writer(str(out_path), fps=float(args.fps))
    else:
        raise SystemExit("[animate] output must be .gif or .mp4")

    for p in frames:
        frame = imageio.imread(p)
        writer.append_data(frame)
    writer.close()
    print(f"[animate] wrote {out_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
