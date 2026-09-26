#!/usr/bin/env python3
"""Stream caches into a standalone cloud in native prediction coordinates.

The evaluator does not use this utility: metric voxelization must happen only
after prediction-to-GT scale initialization in ``evaluate_long3d.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_utils import (
    build_prediction_points,
    discover_cache_files,
    voxel_downsample_numpy,
    write_json,
    write_point_cloud,
)


def optional_float(value: str):
    if value.lower() in {"none", "off", "disabled"}:
        return None
    return float(value)


def optional_int(value: str):
    if value.lower() in {"none", "all"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone native-coordinate prediction cloud. "
            "Do not use this output as a metric cloud before alignment."
        )
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_frames", type=optional_int, default=-1)
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Standalone output voxel size in native prediction units.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=optional_float,
        default=None,
        help="Disabled by default, matching the reference evaluator.",
    )
    parser.add_argument("--preview_path", type=Path, default=None)
    parser.add_argument("--preview_voxel_size", type=float, default=0.10)
    parser.add_argument("--flush_points", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_frames is None or args.max_frames < 0 else args.max_frames
    files = discover_cache_files(args.cache_dir, max_frames)
    points, stats = build_prediction_points(
        files,
        point_stride=args.point_stride,
        voxel_size=args.voxel_size,
        conf_thresh=args.conf_thresh,
        flush_points=args.flush_points,
    )
    write_point_cloud(args.output_path, points)
    if args.preview_path is not None:
        preview = voxel_downsample_numpy(points, args.preview_voxel_size)
        write_point_cloud(args.preview_path, preview)
    metadata = {
        **stats.as_dict(),
        "cache_dir": str(args.cache_dir.resolve()),
        "first_cache": files[0].name,
        "last_cache": files[-1].name,
        "point_stride": args.point_stride,
        "voxel_size": args.voxel_size,
        "voxel_origin": [0.0, 0.0, 0.0],
        "voxel_representative": "mean_of_points_in_voxel",
        "conf_thresh": args.conf_thresh,
        "preview_voxel_size": args.preview_voxel_size if args.preview_path else None,
    }
    write_json(args.output_path.with_suffix(".json"), metadata)
    print(f"Wrote {len(points):,} native-coordinate output points to {args.output_path}")


if __name__ == "__main__":
    main()
