#!/usr/bin/env python3
"""Inspect Long3D caches and GT without constructing the full prediction cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_utils import (
    discover_cache_files,
    extract_prediction,
    load_point_cloud,
    tensor_to_numpy,
    torch_load_cpu,
)
from build_long3d_pointcloud import optional_int


def describe(points: np.ndarray) -> dict:
    flat = np.asarray(points).reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    good = flat[finite]
    result = {
        "points": len(flat),
        "finite_points": len(good),
        "invalid_points": int(len(flat) - len(good)),
    }
    if len(good):
        minimum, maximum = good.min(axis=0), good.max(axis=0)
        result.update(
            minimum=minimum.tolist(),
            maximum=maximum.tolist(),
            extent=(maximum - minimum).tolist(),
            max_extent=float(np.max(maximum - minimum)),
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--gt_path", type=Path, default=None)
    parser.add_argument("--max_frames", type=optional_int, default=-1)
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=8,
        help="Inspection-only spatial stride; does not affect evaluation.",
    )
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_frames = None if args.max_frames is None or args.max_frames < 0 else args.max_frames
    files = discover_cache_files(args.cache_dir, max_frames)
    samples = []
    conf_min, conf_max = np.inf, -np.inf
    invalid_total = 0
    raw_total = 0
    first_structure = None
    for index, path in enumerate(files):
        cache = torch_load_cpu(path)
        points, conf = extract_prediction(cache)
        raw_total += points.shape[0] * points.shape[1]
        invalid_total += int((~np.isfinite(points).all(axis=-1)).sum())
        samples.append(points[:: args.sample_stride, :: args.sample_stride].reshape(-1, 3))
        if conf is not None:
            finite_conf = conf[np.isfinite(conf)]
            if len(finite_conf):
                conf_min = min(conf_min, float(finite_conf.min()))
                conf_max = max(conf_max, float(finite_conf.max()))
        if index == 0:
            first_structure = {
                outer: {
                    key: list(tensor_to_numpy(value).shape) if hasattr(value, "shape") else type(value).__name__
                    for key, value in section.items()
                }
                for outer, section in cache.items()
                if isinstance(section, dict)
            }
        del cache
    sampled = np.concatenate(samples, axis=0)
    result = {
        "cache_dir": str(args.cache_dir.resolve()),
        "frames": len(files),
        "first_cache": files[0].name,
        "last_cache": files[-1].name,
        "raw_prediction_points": int(raw_total),
        "raw_invalid_prediction_points": int(invalid_total),
        "inspection_sample_stride": args.sample_stride,
        "prediction_sample": describe(sampled),
        "confidence_range": None if not np.isfinite(conf_min) else [conf_min, conf_max],
        "first_cache_structure": first_structure,
    }
    if args.gt_path is not None:
        result["gt"] = describe(load_point_cloud(args.gt_path))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
