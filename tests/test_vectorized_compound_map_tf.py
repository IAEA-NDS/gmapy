import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestVectorizedCompoundMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data-2017-07-26.gma').resolve().as_posix()
        rawdb = read_gma_database(dbpath)
        priortable = create_prior_table(rawdb['prior_list'])
        exptable = create_experiment_table(rawdb['datablock_list'])
        priortable = attach_shape_prior((priortable, exptable))
        cls._priortable = priortable
        cls._exptable = exptable
        cls._gmadb = GMADatabase(
            prior_list=rawdb['prior_list'],
            datablock_list=rawdb['datablock_list'],
            remove_dummy=False
        )

    def test_propagate_equals_compound_map_reduce(self):
        refmap = CompoundMapTF(
            (self._priortable, self._exptable), reduce=True
        )
        vecmap = VectorizedCompoundMap(
            (self._priortable, self._exptable), reduce=True
        )
        x = self._priortable['PRIOR'].to_numpy() + 1e-5
        ref = refmap(x).numpy()
        res = vecmap(x).numpy()
        self.assertTrue(
            np.allclose(res, ref, rtol=1e-10, atol=1e-10),
            msg=f'max abs diff {np.max(np.abs(res - ref))}'
        )

    def test_propagate_equals_compound_map_noreduce(self):
        dt = self._gmadb.get_datatable()
        refmap = CompoundMapTF(dt, reduce=False)
        vecmap = VectorizedCompoundMap(dt, reduce=False)
        x = dt['PRIOR'].to_numpy() + 1e-5
        ref = refmap(x).numpy()
        res = vecmap(x).numpy()
        self.assertTrue(
            np.allclose(res, ref, rtol=1e-10, atol=1e-10),
            msg=f'max abs diff {np.max(np.abs(res - ref))}'
        )

    def test_jacobian_equals_compound_map_reduce(self):
        refmap = CompoundMapTF(
            (self._priortable, self._exptable), reduce=True
        )
        vecmap = VectorizedCompoundMap(
            (self._priortable, self._exptable), reduce=True
        )
        x = self._priortable['PRIOR'].to_numpy() + 1e-5
        refjac = tf.sparse.to_dense(refmap.jacobian(x)).numpy()
        resjac = tf.sparse.to_dense(vecmap.jacobian(x)).numpy()
        self.assertTrue(
            np.allclose(resjac, refjac, rtol=1e-8, atol=1e-10),
            msg=f'max abs diff {np.max(np.abs(resjac - refjac))}'
        )

    def test_propagate_inside_tf_function_with_gradient(self):
        vecmap = VectorizedCompoundMap(
            (self._priortable, self._exptable), reduce=True
        )
        x = tf.constant(
            self._priortable['PRIOR'].to_numpy() + 1e-5, dtype=tf.float64
        )

        @tf.function
        def sum_and_grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                res = tf.reduce_sum(vecmap.propagate(x))
            return res, tape.gradient(res, x)

        val, grad = sum_and_grad(x)
        self.assertTrue(np.isfinite(val.numpy()))
        self.assertTrue(np.all(np.isfinite(grad.numpy())))


if __name__ == '__main__':
    unittest.main()
