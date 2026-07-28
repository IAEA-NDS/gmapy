import unittest
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.mapping_elements_tf import (
    PiecewiseLinearInterpolation
)
from gmapy.mappings.tf.vectorized_helpers import (
    piecewise_linear_interp_matrix
)


class TestPiecewiseLinearInterpMatrix(unittest.TestCase):

    def _compare(self, src_en, tar_en, seed=42):
        rng = np.random.default_rng(seed)
        y = rng.uniform(1., 10., size=len(src_en))
        ref = PiecewiseLinearInterpolation(src_en, tar_en)(
            tf.constant(y, dtype=tf.float64)
        ).numpy()
        rows, cols, vals = piecewise_linear_interp_matrix(src_en, tar_en)
        mat = np.zeros((len(tar_en), len(src_en)))
        np.add.at(mat, (rows, cols), vals)
        res = mat @ y
        self.assertTrue(np.allclose(res, ref, rtol=1e-14, atol=1e-14))

    def test_sorted_mesh_interior_points(self):
        src = np.array([1., 2., 5., 10.])
        tar = np.array([1.5, 2., 3., 9.99])
        self._compare(src, tar)

    def test_unsorted_source_mesh(self):
        src = np.array([5., 1., 10., 2.])
        tar = np.array([1.5, 3., 7.])
        self._compare(src, tar)

    def test_target_points_outside_mesh_are_zero(self):
        src = np.array([1., 2., 5.])
        tar = np.array([0.5, 1., 5., 7.])
        self._compare(src, tar)
        rows, _, _ = piecewise_linear_interp_matrix(src, tar)
        self.assertNotIn(0, rows)
        self.assertNotIn(3, rows)

    def test_target_on_mesh_points(self):
        src = np.array([1., 2., 5., 10.])
        self._compare(src, src.copy())

    def test_target_between_all_segments_random(self):
        rng = np.random.default_rng(7)
        src = np.sort(rng.uniform(0., 20., size=50))
        tar = rng.uniform(-1., 21., size=200)
        self._compare(src, tar)

    def test_unsorted_random_mesh_random_targets(self):
        rng = np.random.default_rng(11)
        src = rng.permutation(np.unique(rng.uniform(0., 20., size=40)))
        tar = rng.uniform(0., 20., size=100)
        self._compare(src, tar)

    def test_single_point_mesh_raises(self):
        with self.assertRaises(NotImplementedError):
            piecewise_linear_interp_matrix([1.], [1., 2.])


if __name__ == '__main__':
    unittest.main()
