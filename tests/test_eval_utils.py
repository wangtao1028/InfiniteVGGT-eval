import unittest

import numpy as np

from eval_utils import (
    StreamingPointAccumulator,
    accuracy,
    apply_transform,
    completion,
    extent_scale_initialization,
)


class EvalUtilsTests(unittest.TestCase):
    def test_reference_directional_metrics(self):
        gt = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.assertEqual(accuracy(gt, pred), (0.5, 0.5))
        self.assertEqual(completion(gt, pred), (0.5, 0.5))

    def test_extent_initialization_recovers_scale_and_translation(self):
        gt = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
        pred = gt / 4.0 + np.array([10.0, -3.0, 2.0])
        transform = extent_scale_initialization(pred, gt)
        np.testing.assert_allclose(apply_transform(pred, transform), gt, atol=1e-12)

    def test_streaming_voxels_average_across_flushes(self):
        accumulator = StreamingPointAccumulator(voxel_size=1.0, flush_points=2)
        accumulator.add(np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]))
        accumulator.add(np.array([[0.5, 0.5, 0.5], [1.2, 0.0, 0.0]]))
        result = accumulator.finalize()
        np.testing.assert_allclose(
            result,
            np.array([[0.3, 0.3, 0.3], [1.2, 0.0, 0.0]]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
