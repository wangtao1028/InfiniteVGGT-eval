#!/usr/bin/env python3
"""Independent evaluation adapter for InfiniteVGGT caches and Long3D scans."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_long3d_pointcloud import optional_float, optional_int
from eval_utils import (
    apply_transform,
    build_transformed_prediction_points,
    decompose_similarity,
    discover_cache_files,
    estimate_initial_transform,
    evaluate_metrics,
    load_point_cloud,
    read_transform,
    refine_alignment,
    scan_prediction_points,
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
        default=0.10,
        help=(
            "Alignment-only voxel size in GT/world units (a Long3D reproduction "
            "assumption). This is independent of metric and visualization "
            "sampling; use 0 to reuse metric clouds."
        ),
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

    if args.voxel_size < 0:
        raise ValueError("voxel_size must be >= 0")
    if args.alignment_voxel_size < 0:
        raise ValueError("alignment_voxel_size must be >= 0")
    if args.preview_voxel_size < 0:
        raise ValueError("preview_voxel_size must be >= 0")

    # Pass 1 keeps only counts and bounds. No native-scale metric voxelization.
    scan = scan_prediction_points(files, args.point_stride, args.conf_thresh)
    gt_raw = load_point_cloud(args.gt_path)
    supplied_transform = read_transform(args.init_transform) if args.init_transform else None
    initial_transform, init_source = estimate_initial_transform(
        scan.minimum,
        scan.maximum,
        gt_raw,
        args.alignment_mode,
        supplied_transform,
    )

    # Pass 2 transforms every surviving prediction point into GT/world units
    # before either the metric or alignment voxel grid sees it.
    pred_metric_initial, pred_alignment, after_transform = (
        build_transformed_prediction_points(
            files,
            point_stride=args.point_stride,
            conf_thresh=args.conf_thresh,
            initial_transform=initial_transform,
            metric_voxel_size=args.voxel_size,
            alignment_voxel_size=args.alignment_voxel_size,
            flush_points=args.flush_points,
        )
    )
    if after_transform != scan.after_filter:
        raise RuntimeError(
            "Prediction cache contents changed between streaming passes: "
            f"{scan.after_filter} points in pass 1, {after_transform} in pass 2"
        )

    gt_metric = voxel_downsample_numpy(gt_raw, args.voxel_size)
    gt_alignment = (
        voxel_downsample_numpy(gt_raw, args.alignment_voxel_size)
        if args.alignment_voxel_size > 0
        else gt_metric
    )
    icp_transform, fitness, inlier_rmse, estimator = refine_alignment(
        pred_alignment,
        gt_alignment,
        mode=args.alignment_mode,
        icp_threshold=args.icp_threshold,
    )
    final_transform = icp_transform @ initial_transform
    pred_metric = apply_transform(pred_metric_initial, icp_transform)
    scale, rotation, translation = decompose_similarity(final_transform)
    alignment = {
        "alignment_mode": args.alignment_mode,
        "initialization": init_source,
        "initial_transform": initial_transform.tolist(),
        "icp_transform": icp_transform.tolist(),
        "transformation": final_transform.tolist(),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "icp_threshold": float(args.icp_threshold),
        "icp_fitness": fitness,
        "icp_inlier_rmse": inlier_rmse,
        "icp_estimator": estimator,
        "alignment_voxel_size": float(args.alignment_voxel_size),
        "sampling_units": "GT/world units after initial transform",
    }

    metrics = evaluate_metrics(pred_metric, gt_metric)
    point_counts = {
        "prediction": {
            "raw": scan.raw,
            "after_stride": scan.after_stride,
            "after_transform": after_transform,
            "after_metric_voxel": len(pred_metric_initial),
            "alignment_points": len(pred_alignment),
            "metric_points": len(pred_metric),
        },
        "ground_truth": {
            "raw": len(gt_raw),
            "after_stride": len(gt_raw),
            "after_transform": len(gt_raw),
            "after_metric_voxel": len(gt_metric),
            "alignment_points": len(gt_alignment),
            "metric_points": len(gt_metric),
        },
    }
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
        "metric_voxel_units": "GT/world units",
        "conf_thresh": args.conf_thresh,
        "alignment_voxel_size": args.alignment_voxel_size,
        "preview_voxel_size": args.preview_voxel_size,
        "prediction_filtering": scan.filtering_dict(),
        "point_counts": point_counts,
        "reproduction_assumptions": [
            "Long3D scale initialization uses maximum AABB extent and AABB centers when no transform is supplied.",
            "Long3D alignment uses point-to-point ICP with the configured threshold.",
            "No Long3D crop is applied and confidence filtering is disabled unless explicitly requested.",
            "Metric, alignment, and visualization voxel sizes are independent sampling choices.",
        ],
        "alignment": alignment,
    }
    metrics.update(parameters)
    write_json(args.output_dir / "alignment.json", {**alignment, "parameters": parameters})
    write_json(args.output_dir / "metrics.json", metrics)

    # Preview PLYs are explicitly not the metric point sets unless sizes happen to match.
    pred_initial_preview = voxel_downsample_numpy(
        pred_metric_initial, args.preview_voxel_size
    )
    aligned_preview = voxel_downsample_numpy(pred_metric, args.preview_voxel_size)
    gt_preview = voxel_downsample_numpy(gt_metric, args.preview_voxel_size)
    write_point_cloud(args.output_dir / "pred_initial_aligned_preview.ply", pred_initial_preview)
    write_point_cloud(args.output_dir / "pred_aligned_preview.ply", aligned_preview)
    write_point_cloud(args.output_dir / "gt_preview.ply", gt_preview)

    summary_lines = [
        "InfiniteVGGT Long3D evaluation",
        "",
        f"Frames: {len(files)}",
        f"Prediction metric points: {len(pred_metric):,}",
        f"GT metric points: {len(gt_metric):,}",
        f"Prediction / GT alignment points: {len(pred_alignment):,} / {len(gt_alignment):,}",
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
        f"Alignment voxel size: {args.alignment_voxel_size}",
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
