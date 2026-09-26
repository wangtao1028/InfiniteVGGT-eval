"""Utilities for an independent Long3D point-cloud evaluator.

The metric definitions intentionally mirror StreamVGGT/CUT3R's mv_recon/utils.py.
Long3D-specific preprocessing and alignment choices are kept separate and are
always returned as metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree as KDTree


def _as_points(points: Any, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """StreamVGGT/CUT3R accuracy: prediction -> nearest GT distance."""
    gt_points = _as_points(gt_points, "gt_points")
    rec_points = _as_points(rec_points, "rec_points")
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)
    acc_median = np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        gt_normals = np.asarray(gt_normals)
        rec_normals = np.asarray(rec_normals)
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)
    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """StreamVGGT/CUT3R completion: GT -> nearest prediction distance."""
    gt_points = _as_points(gt_points, "gt_points")
    rec_points = _as_points(rec_points, "rec_points")
    rec_points_kd_tree = KDTree(rec_points)
    distances, idx = rec_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)
    if gt_normals is not None and rec_normals is not None:
        gt_normals = np.asarray(gt_normals)
        rec_normals = np.asarray(rec_normals)
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)
    return comp, comp_median


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required. Install it in the InfiniteVGGT environment with "
            "`pip install open3d`."
        ) from exc
    return o3d


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read InfiniteVGGT .pt caches. Run this script "
            "inside the InfiniteVGGT environment."
        ) from exc
    return torch


def torch_load_cpu(path: Path):
    torch = require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location="cpu")


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def discover_cache_files(cache_dir: Path, max_frames: Optional[int]) -> list[Path]:
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {cache_dir}")
    if max_frames is not None and max_frames >= 0:
        files = files[:max_frames]
    if not files:
        raise ValueError("max_frames selected zero cache files")
    return files


def extract_prediction(cache: dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pred = cache.get("pred")
    if not isinstance(pred, dict):
        raise KeyError("cache has no dictionary at key 'pred'")
    if "pts3d_in_other_view" not in pred:
        raise KeyError("cache pred has no 'pts3d_in_other_view'")
    points = np.squeeze(tensor_to_numpy(pred["pts3d_in_other_view"]))
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            "pts3d_in_other_view must reduce to (H, W, 3), "
            f"got {points.shape}"
        )
    conf = pred.get("conf")
    if conf is not None:
        conf = np.squeeze(tensor_to_numpy(conf))
        if conf.shape != points.shape[:2]:
            raise ValueError(
                f"conf shape {conf.shape} does not match points {points.shape[:2]}"
            )
    return points.astype(np.float64, copy=False), conf


def _reduce_voxels(
    keys: np.ndarray, sums: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort voxel records and combine equal integer keys."""
    if len(keys) == 0:
        return keys, sums, counts
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    sums = sums[order]
    counts = counts[order]
    starts_mask = np.empty(len(keys), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.flatnonzero(starts_mask)
    return (
        keys[starts],
        np.add.reduceat(sums, starts, axis=0),
        np.add.reduceat(counts, starts, axis=0),
    )


class StreamingPointAccumulator:
    """Bounded-buffer point collector with deterministic, origin-anchored voxels."""

    def __init__(self, voxel_size: float, flush_points: int = 2_000_000):
        if voxel_size < 0:
            raise ValueError("voxel_size must be >= 0")
        self.voxel_size = float(voxel_size)
        self.flush_points = int(flush_points)
        self._chunks: list[np.ndarray] = []
        self._buffer_count = 0
        self._keys = np.empty((0, 3), dtype=np.int64)
        self._sums = np.empty((0, 3), dtype=np.float64)
        self._counts = np.empty((0,), dtype=np.int64)

    def add(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            return
        self._chunks.append(points.copy())
        self._buffer_count += len(points)
        if self.voxel_size > 0 and self._buffer_count >= self.flush_points:
            self._flush_voxels()

    def _flush_voxels(self) -> None:
        if not self._chunks:
            return
        points = np.concatenate(self._chunks, axis=0)
        self._chunks.clear()
        self._buffer_count = 0
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        counts = np.ones(len(points), dtype=np.int64)
        keys, sums, counts = _reduce_voxels(keys, points, counts)
        if len(self._keys):
            keys = np.concatenate([self._keys, keys], axis=0)
            sums = np.concatenate([self._sums, sums], axis=0)
            counts = np.concatenate([self._counts, counts], axis=0)
            keys, sums, counts = _reduce_voxels(keys, sums, counts)
        self._keys, self._sums, self._counts = keys, sums, counts

    def finalize(self) -> np.ndarray:
        if self.voxel_size > 0:
            self._flush_voxels()
            if not len(self._keys):
                raise ValueError("No valid prediction points survived filtering")
            return self._sums / self._counts[:, None]
        if not self._chunks:
            raise ValueError("No valid prediction points survived filtering")
        # This path is exact but can be very large. Callers choose it explicitly.
        return np.concatenate(self._chunks, axis=0)


@dataclass
class BuildStats:
    frames: int = 0
    raw_points: int = 0
    after_stride_points: int = 0
    finite_points: int = 0
    confidence_rejected: int = 0
    evaluation_points: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def build_prediction_points(
    files: Iterable[Path],
    point_stride: int,
    voxel_size: float,
    conf_thresh: Optional[float],
    flush_points: int = 2_000_000,
) -> tuple[np.ndarray, BuildStats]:
    if point_stride < 1:
        raise ValueError("point_stride must be >= 1")
    accumulator = StreamingPointAccumulator(voxel_size, flush_points=flush_points)
    stats = BuildStats()
    for path in files:
        cache = torch_load_cpu(path)
        points, conf = extract_prediction(cache)
        stats.frames += 1
        stats.raw_points += int(points.shape[0] * points.shape[1])
        points = points[::point_stride, ::point_stride, :].reshape(-1, 3)
        if conf is not None:
            conf = conf[::point_stride, ::point_stride].reshape(-1)
        stats.after_stride_points += len(points)
        valid = np.isfinite(points).all(axis=1)
        if conf is not None:
            valid &= np.isfinite(conf)
        finite_before_conf = int(valid.sum())
        stats.finite_points += finite_before_conf
        if conf_thresh is not None:
            if conf is None:
                raise ValueError(
                    f"conf_thresh was set but {path.name} contains no confidence tensor"
                )
            valid &= conf > conf_thresh
            stats.confidence_rejected += finite_before_conf - int(valid.sum())
        accumulator.add(points[valid])
        del cache, points, conf
    result = accumulator.finalize()
    stats.evaluation_points = len(result)
    return result, stats


def voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return np.asarray(points, dtype=np.float64)
    accumulator = StreamingPointAccumulator(voxel_size)
    accumulator.add(points)
    return accumulator.finalize()


def make_point_cloud(points: np.ndarray):
    o3d = require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(_as_points(points, "points"))
    return cloud


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    o3d = require_open3d()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), make_point_cloud(points)):
        raise IOError(f"Open3D failed to write {path}")


def load_point_cloud(path: Path) -> np.ndarray:
    o3d = require_open3d()
    if not path.is_file():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    return _as_points(points, f"point cloud {path}")


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {transform.shape}")
    return points @ transform[:3, :3].T + transform[:3, 3]


def extent_scale_initialization(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Reproduction choice: match maximum axis extent, then AABB centers."""
    pred_min, pred_max = pred.min(axis=0), pred.max(axis=0)
    gt_min, gt_max = gt.min(axis=0), gt.max(axis=0)
    pred_extent = float(np.max(pred_max - pred_min))
    gt_extent = float(np.max(gt_max - gt_min))
    if pred_extent <= 0 or not math.isfinite(pred_extent):
        raise ValueError(f"Invalid prediction extent: {pred_extent}")
    scale = gt_extent / pred_extent
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = (gt_min + gt_max) / 2.0 - scale * (pred_min + pred_max) / 2.0
    return transform


def decompose_similarity(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    det = float(np.linalg.det(linear))
    scale = float(np.cbrt(abs(det)))
    if scale <= 0:
        raise ValueError("Alignment transform has zero scale")
    rotation_approx = linear / scale
    u, _, vt = np.linalg.svd(rotation_approx)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return scale, rotation, np.asarray(transform, dtype=np.float64)[:3, 3]


def read_transform(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    matrix = data.get("transformation", data.get("transform", data)) if isinstance(data, dict) else data
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Initial transform in {path} must be a finite 4x4 matrix")
    return matrix


def align_point_clouds(
    pred: np.ndarray,
    gt: np.ndarray,
    mode: str,
    icp_threshold: float,
    initial_transform: Optional[np.ndarray] = None,
    alignment_voxel_size: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align prediction to GT; ICP thresholds are interpreted in GT units."""
    o3d = require_open3d()
    pred = _as_points(pred, "pred")
    gt = _as_points(gt, "gt")
    if icp_threshold <= 0:
        raise ValueError("icp_threshold must be > 0")
    if mode not in {"none", "rigid_icp", "scale_then_icp", "sim3_icp"}:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    if initial_transform is not None:
        init = np.asarray(initial_transform, dtype=np.float64)
        init_source = "user_json"
    elif mode in {"scale_then_icp", "sim3_icp"}:
        init = extent_scale_initialization(pred, gt)
        init_source = "max_extent_and_aabb_center"
    else:
        init = np.eye(4, dtype=np.float64)
        init_source = "identity"

    fitness = None
    inlier_rmse = None
    icp_transform = np.eye(4, dtype=np.float64)
    if mode == "none":
        final = init if initial_transform is not None else np.eye(4, dtype=np.float64)
    else:
        pred_init = apply_transform(pred, init)
        source = make_point_cloud(pred_init)
        target = make_point_cloud(gt)
        if alignment_voxel_size > 0:
            source = source.voxel_down_sample(alignment_voxel_size)
            target = target.voxel_down_sample(alignment_voxel_size)
        with_scaling = mode == "sim3_icp"
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling
        )
        registration = o3d.pipelines.registration.registration_icp(
            source,
            target,
            icp_threshold,
            np.eye(4, dtype=np.float64),
            estimator,
        )
        icp_transform = np.asarray(registration.transformation, dtype=np.float64)
        final = icp_transform @ init
        fitness = float(registration.fitness)
        inlier_rmse = float(registration.inlier_rmse)

    aligned = apply_transform(pred, final)
    scale, rotation, translation = decompose_similarity(final)
    metadata = {
        "alignment_mode": mode,
        "initialization": init_source,
        "initial_transform": init.tolist(),
        "icp_transform": icp_transform.tolist(),
        "transformation": final.tolist(),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "icp_threshold": float(icp_threshold),
        "icp_fitness": fitness,
        "icp_inlier_rmse": inlier_rmse,
        "icp_estimator": (
            None
            if mode == "none"
            else "point_to_point_with_scaling"
            if mode == "sim3_icp"
            else "point_to_point"
        ),
        "alignment_voxel_size": float(alignment_voxel_size),
    }
    return aligned, metadata


def evaluate_metrics(pred_aligned: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Estimate Open3D-default normals, then apply reference metric definitions."""
    pred_cloud = make_point_cloud(pred_aligned)
    gt_cloud = make_point_cloud(gt)
    # Deliberately no search parameter: this matches the reference evaluator.
    pred_cloud.estimate_normals()
    gt_cloud.estimate_normals()
    pred_points = np.asarray(pred_cloud.points)
    gt_points = np.asarray(gt_cloud.points)
    pred_normals = np.asarray(pred_cloud.normals)
    gt_normals = np.asarray(gt_cloud.normals)
    acc, acc_med, nc_pred, nc_pred_med = accuracy(
        gt_points, pred_points, gt_normals, pred_normals
    )
    comp, comp_med, nc_gt, nc_gt_med = completion(
        gt_points, pred_points, gt_normals, pred_normals
    )
    return {
        "accuracy_mean": float(acc),
        "accuracy_median": float(acc_med),
        "completeness_mean": float(comp),
        "completeness_median": float(comp_med),
        "normal_consistency_mean": float((nc_pred + nc_gt) / 2.0),
        "normal_consistency_median": float((nc_pred_med + nc_gt_med) / 2.0),
        "normal_consistency_pred_to_gt_mean": float(nc_pred),
        "normal_consistency_pred_to_gt_median": float(nc_pred_med),
        "normal_consistency_gt_to_pred_mean": float(nc_gt),
        "normal_consistency_gt_to_pred_median": float(nc_gt_med),
        "chamfer_distance": float((acc + comp) / 2.0),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
