import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestWeightedRowHessian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data-2017-07-26.gma').resolve().as_posix()
        rawdb = read_gma_database(dbpath)
        priortable = create_prior_table(rawdb['prior_list'])
        exptable = create_experiment_table(rawdb['datablock_list'])
        priortable = attach_shape_prior((priortable, exptable))
        cls._vecmap = VectorizedCompoundMap(
            (priortable, exptable), reduce=True
        )
        cls._x = priortable['PRIOR'].to_numpy() + 1e-5
        rng = np.random.default_rng(17)
        cls._w = rng.uniform(-1., 1., size=len(exptable))

    def _weighted_sum_gradient(self, x):
        # exact gradient of sum(w * f) restricted to the algebraic
        # part, via autodiff
        vecmap = self._vecmap
        w = tf.constant(self._w, dtype=tf.float64)
        xt = tf.constant(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(xt)
            num, den, g = vecmap._algebraic_parts(xt)
            res = tf.reduce_sum(w * g * num / den)
        return tape.gradient(res, xt).numpy()

    def test_hessian_is_symmetric(self):
        hess = tf.sparse.to_dense(
            self._vecmap.weighted_row_hessian(self._x, self._w)
        ).numpy()
        self.assertTrue(np.allclose(hess, hess.T, rtol=1e-13, atol=1e-13))
        self.assertGreater(np.count_nonzero(hess), 0)

    def test_hessian_matches_finite_difference_hvp(self):
        hess = tf.sparse.to_dense(
            self._vecmap.weighted_row_hessian(self._x, self._w)
        ).numpy()
        rng = np.random.default_rng(23)
        eps = 1e-6
        for _ in range(3):
            r = rng.normal(size=len(self._x))
            r /= np.linalg.norm(r)
            gplus = self._weighted_sum_gradient(self._x + eps * r)
            gminus = self._weighted_sum_gradient(self._x - eps * r)
            hvp_fd = (gplus - gminus) / (2. * eps)
            hvp = hess @ r
            scale = np.max(np.abs(hvp_fd))
            self.assertTrue(
                np.allclose(hvp, hvp_fd, rtol=1e-5, atol=1e-6 * scale),
                msg=f'max abs diff {np.max(np.abs(hvp - hvp_fd))} '
                    f'at scale {scale}'
            )


if __name__ == '__main__':
    unittest.main()
