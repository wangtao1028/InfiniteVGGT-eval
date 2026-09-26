#!/usr/bin/env python3
"""Independent evaluation adapter for InfiniteVGGT caches and Long3D scans."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_long3d_pointcloud import optional_float, optional_int
from eval_utils import (
    align_point_clouds,
    build_prediction_points,
    discover_cache_files,
    evaluate_metrics,
    load_point_cloud,
    read_transform,
    voxel_downsample_numpy,
    write_json,
    write_point_cloud,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cached InfiniteVGGT global points against Long3D GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--gt_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_frames", type=optional_int, default=-1)
    parser.add_argument("--point_stride", type=int, default=4)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument(
        "--conf_thresh",
        type=optional_float,
        default=None,
        help="Disabled by default; pass a number to opt into confidence filtering.",
    )
    parser.add_argument(
        "--alignment_mode",
        choices=["none", "rigid_icp", "scale_then_icp", "sim3_icp"],
        default="scale_then_icp",
    )
    parser.add_argument("--icp_threshold", type=float, default=0.1)
    parser.add_argument(
        "--init_transform",
        type=Path,
        default=None,
        help="Optional JSON 4x4 prediction-to-GT transform.",
    )
    parser.add_argument(
        "--alignment_voxel_size",
        type=float,
        default=0.0,
        help="Optional alignment-only Open3D downsample; metrics still use evaluation points.",
    )
    parser.add_argument(
        "--preview_voxel_size",
        type=float,
        default=0.10,
        help="Only controls the three saved visualization PLY files.",
    )
    parser.add_argument("--flush_points", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_frames = None if args.max_frames is None or args.max_frames < 0 else args.max_frames
    files = discover_cache_files(args.cache_dir, max_frames)

    pred, build_stats = build_prediction_points(
        files,
        point_stride=args.point_stride,
        voxel_size=args.voxel_size,
        conf_thresh=args.conf_thresh,
        flush_points=args.flush_points,
    )
    gt_raw = load_point_cloud(args.gt_path)
    gt = voxel_downsample_numpy(gt_raw, args.voxel_size)
    initial_transform = read_transform(args.init_transform) if args.init_transform else None
    pred_aligned, alignment = align_point_clouds(
        pred,
        gt,
        mode=args.alignment_mode,
        icp_threshold=args.icp_threshold,
        initial_transform=initial_transform,
        alignment_voxel_size=args.alignment_voxel_size,
    )

    metrics = evaluate_metrics(pred_aligned, gt)
    parameters = {
        "cache_dir": str(args.cache_dir.resolve()),
        "gt_path": str(args.gt_path.resolve()),
        "frames": len(files),
        "first_cache": files[0].name,
        "last_cache": files[-1].name,
        "point_stride": args.point_stride,
        "voxel_size": args.voxel_size,
        "voxel_origin": [0.0, 0.0, 0.0],
        "voxel_representative": "mean_of_points_in_voxel",
        "conf_thresh": args.conf_thresh,
        "pred_points_count": len(pred),
        "gt_points_count": len(gt),
        "gt_points_count_before_voxel": len(gt_raw),
        "preview_voxel_size": args.preview_voxel_size,
        "build_stats": build_stats.as_dict(),
        "alignment": alignment,
    }
    metrics.update(parameters)
    write_json(args.output_dir / "alignment.json", {**alignment, "parameters": parameters})
    write_json(args.output_dir / "metrics.json", metrics)

    # Preview PLYs are explicitly not the metric point sets unless sizes happen to match.
    pred_preview = voxel_downsample_numpy(pred, args.preview_voxel_size)
    aligned_preview = voxel_downsample_numpy(pred_aligned, args.preview_voxel_size)
    write_point_cloud(args.output_dir / "pred_before_alignment.ply", pred_preview)
    write_point_cloud(args.output_dir / "pred_aligned.ply", aligned_preview)
    # Unlike the two prediction previews, this is the exact GT metric point set.
    write_point_cloud(args.output_dir / "gt_used.ply", gt)

    summary_lines = [
        "InfiniteVGGT Long3D evaluation",
        "",
        f"Frames: {len(files)}",
        f"Prediction evaluation points: {len(pred):,}",
        f"GT evaluation points: {len(gt):,}",
        f"Accuracy mean / median: {metrics['accuracy_mean']:.9g} / {metrics['accuracy_median']:.9g}",
        f"Completeness mean / median: {metrics['completeness_mean']:.9g} / {metrics['completeness_median']:.9g}",
        f"Normal consistency mean / median: {metrics['normal_consistency_mean']:.9g} / {metrics['normal_consistency_median']:.9g}",
        f"Chamfer distance: {metrics['chamfer_distance']:.9g}",
        "",
        f"Alignment mode: {args.alignment_mode}",
        f"Scale: {alignment['scale']:.12g}",
        f"ICP fitness: {alignment['icp_fitness']}",
        f"ICP inlier RMSE: {alignment['icp_inlier_rmse']}",
        f"Point stride: {args.point_stride}",
        f"Evaluation voxel size: {args.voxel_size}",
        f"Confidence threshold: {args.conf_thresh}",
        f"Preview voxel size (not used for metrics): {args.preview_voxel_size}",
        "",
        "WARNING: Long3D's public paper does not disclose scale initialization, crop,",
        "confidence filtering, or downsampling. See README.md before comparing to paper values.",
    ]
    (args.output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
