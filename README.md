# InfiniteVGGT Long3D evaluation adapter

This is an inference-free adapter for per-frame caches produced by InfiniteVGGT.
It does **not** import or modify InfiniteVGGT model code. In particular,
`pred.pts3d_in_other_view` is treated as an already-global point prediction and
is never transformed by the cached extrinsic matrix.

## What the reference evaluator actually does

The implementation was checked against these public files:

- StreamVGGT `src/eval/mv_recon/{launch.py,utils.py,criterion.py}`
- CUT3R `eval/mv_recon/{launch.py,utils.py,criterion.py}`
- InfiniteVGGT `run_inference.py`, `demo_viser.py`, and paper arXiv:2601.02281

For 7Scenes/NeuralRGBD, both reference pipelines take
`pts3d_in_other_view`, use the dataset's per-pixel `valid_mask`, crop the center
224x224 region, and perform scale-and-z-shift normalization using paired
per-pixel GT pointmaps. CUT3R then restores the GT z shift to both point sets and
transforms both from the first-camera frame back into the scene frame. In the
current StreamVGGT fork those restoration/transform lines are absent or
commented, so its ICP operates on the normalized first-camera-frame clouds.
Both then run Open3D point-to-point ICP from the identity with threshold 0.1.
The declared confidence threshold is not used: the filtering line is commented
out. There is no voxel downsample in either evaluator. Normals are estimated
with a bare Open3D `estimate_normals()` call.

Metrics reproduce `utils.py`:

- Accuracy: prediction-to-nearest-GT Euclidean distance, mean and median.
- Completion: GT-to-nearest-prediction Euclidean distance, mean and median.
- Directional NC: absolute dot product of corresponding nearest-neighbor
  normals. Reported NC is the average of the two directional values.
- CD: `(accuracy_mean + completeness_mean) / 2`, as stated by the
  InfiniteVGGT paper.

## Missing Long3D protocol details

The public InfiniteVGGT paper says only that predicted and scanner GT point
clouds are aligned with ICP. It does not disclose the scale initialization,
coordinate initialization, ICP threshold, crop, confidence threshold, or
downsampling. The public Long3D data description exposes a dense scanner cloud,
not the paired per-pixel GT pointmaps required by the CUT3R/StreamVGGT
scale-and-shift code. Therefore that normalization cannot be faithfully copied.
Rigid ICP alone cannot repair an arbitrary scale mismatch.

The following are explicitly **implementation assumptions / reproduction
choices**, not claims about the unpublished official Long3D evaluator:

1. `scale_then_icp` (default) initializes uniform scale using the ratio of the
   two maximum axis-aligned extents and aligns AABB centers, then runs the same
   point-to-point rigid ICP used by StreamVGGT. This is deterministic and does
   not use paper target numbers.
2. `sim3_icp` uses the same initialization, then allows Open3D point-to-point
   ICP to refine uniform scale. This is an alternative sensitivity experiment,
   not the default.
3. No Long3D crop is applied. Confidence filtering is disabled by default.
4. To make 432 million cached points tractable, the default evaluation point
   set uses a 2D pixel stride of 4 and a 0.02-unit, origin-anchored voxel grid.
   A voxel is represented by the mean of its points. GT is passed through the
   same voxel rule. These choices materially affect metrics and are always
   recorded. Set `--point_stride 1 --voxel_size 0` only if sufficient RAM is
   available; this exact path may require many gigabytes.
5. Preview PLYs use `--preview_voxel_size` and are never used for metrics.

For a defensible reproduction, report the chosen alignment mode and run
sensitivity checks. Do not describe these results as matching the paper's
official protocol unless the authors publish the missing details.

## Alignment modes

- `none`: no alignment (unless `--init_transform` is supplied).
- `rigid_icp`: identity or supplied initialization, then rigid point-to-point
  ICP. This will not solve the observed scale mismatch by itself.
- `scale_then_icp`: max-extent scale + AABB-center initialization, then rigid
  point-to-point ICP. This is the default reproduction choice.
- `sim3_icp`: the same initialization followed by point-to-point ICP with
  Open3D uniform scaling enabled.

An externally justified prediction-to-GT transform can be supplied as a JSON
4x4 matrix via `--init_transform`. This is the preferred route if scanner poses
or camera correspondences become available.

## Install

Run inside the InfiniteVGGT environment so its PyTorch installation is reused:

```bash
cd /root/InfiniteVGGT-eval
pip install -r requirements.txt
```

## Inspect

Inspection samples pixels only for statistics; it does not create an evaluation
cloud or change later evaluation settings.

```bash
python inspect_long3d.py \
  --cache_dir /root/autodl-tmp/InfiniteVGGT-data/results/exp001_classroom_100/frame_cache \
  --gt_path /root/autodl-tmp/InfiniteVGGT-data/datasets/Long3D/Classroom/dense_cloud_map.pcd \
  --max_frames 100
```

## 100-frame smoke test

This validates `.pt -> point cloud -> alignment -> metrics`. Because the GT is
the complete 2,128-frame Classroom scan, these numbers must not be compared to
the paper's Classroom row.

```bash
python evaluate_long3d.py \
  --cache_dir /root/autodl-tmp/InfiniteVGGT-data/results/exp001_classroom_100/frame_cache \
  --gt_path /root/autodl-tmp/InfiniteVGGT-data/datasets/Long3D/Classroom/dense_cloud_map.pcd \
  --output_dir /root/InfiniteVGGT-eval/outputs/classroom_100_smoke \
  --max_frames 100 \
  --point_stride 4 \
  --voxel_size 0.02 \
  --conf_thresh none \
  --alignment_mode scale_then_icp \
  --icp_threshold 0.1
```

## Full Classroom command (not run automatically)

```bash
python evaluate_long3d.py \
  --cache_dir /root/autodl-tmp/InfiniteVGGT-data/results/exp002_classroom_full/frame_cache \
  --gt_path /root/autodl-tmp/InfiniteVGGT-data/datasets/Long3D/Classroom/dense_cloud_map.pcd \
  --output_dir /root/InfiniteVGGT-eval/outputs/classroom_full_scale_then_icp \
  --max_frames -1 \
  --point_stride 4 \
  --voxel_size 0.02 \
  --conf_thresh none \
  --alignment_mode scale_then_icp \
  --icp_threshold 0.1
```

The output directory contains `metrics.json`, `summary.txt`,
`alignment.json`, `pred_before_alignment.ply`, `pred_aligned.ply`, and
`gt_used.ply`. The two prediction PLYs are visualization previews at
`preview_voxel_size`; `gt_used.ply` is the exact post-evaluation-voxel GT point
set. Exact metric counts and all filtering/alignment parameters are stored in
JSON.

## Fair comparisons

Use identical CLI settings for baseline and modified predictions. Keep the
cache set, frame count, stride, evaluation voxel, confidence setting, alignment
mode, ICP threshold, and any initial transform unchanged. Runtime and memory
should be measured separately from reconstruction metrics; this adapter does
not rerun or time model inference.
