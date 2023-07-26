import unittest
import pathlib
import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
from gmapy.mappings.energy_dependent_usu_map import attach_endep_usu_df


class TestCompoundMapWithErrorMapsTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._dbpath = dbpath
        cls._gmadb = GMADatabase(dbpath, use_relative_errors=True)

    def test_proper_propagation_of_endep_usu_errors(self):
        dt = self._gmadb.get_datatable()
        dt = attach_endep_usu_df(
            dt, ['MT:1-R1:8'], [1, 10, 18], [0.1, 0.2, 0.3]
        )
        refvals = dt.PRIOR.to_numpy(copy=True)
        compmap = CompoundMap(dt)
        compmap_tf = CompoundMapTF(dt)
        sel = (dt.NODE.str.match('endep_usu_1027') &
               (dt.ENERGY.isin([1.0, 10.0, 18.0])))
        idcs = dt.index[sel]
        assert len(idcs) == 3
        refvals[idcs] = np.array((0.11, 0.34, -0.12), dtype=float)
        refvals_tf = tf.Variable(refvals, dtype=tf.float64)
        propvals = compmap.propagate(refvals)
        propvals_tf = compmap_tf(refvals_tf)
        self.assertTrue(np.allclose(propvals, propvals_tf.numpy()))

    def test_proper_jacobian_of_endep_usu_errors(self):
        dt = self._gmadb.get_datatable()
        dt = attach_endep_usu_df(
            dt, ['MT:1-R1:8'], [1, 10, 18], [0.1, 0.2, 0.3]
        )
        refvals = dt.PRIOR.to_numpy(copy=True)
        compmap = CompoundMap(dt)
        compmap_tf = CompoundMapTF(dt)
        sel = (dt.NODE.str.match('endep_usu_1027') &
               (dt.ENERGY.isin([1.0, 10.0, 18.0])))
        idcs = dt.index[sel]
        assert len(idcs) == 3
        refvals[idcs] = np.array((0.11, 0.34, -0.12), dtype=float)
        refvals_tf = tf.Variable(refvals, dtype=tf.float64)
        jac = compmap.jacobian(refvals, with_id=False)
        jac_tf = compmap_tf.jacobian(refvals_tf)
        jac_tf_csr = csr_matrix((jac_tf.values, (
            jac_tf.indices[:, 0], jac_tf.indices[:, 1])), shape=jac_tf.dense_shape
        )
        self.assertTrue(np.allclose(jac.toarray(), jac_tf_csr.toarray()))

    def test_proper_propagation_of_relative_errors(self):
        dt = self._gmadb.get_datatable()
        sel = dt.NODE.str.match('relerr_1027') & (dt.ENERGY == 14.0)
        idx = dt.index[sel]
        assert len(idx) == 1
        refvals = dt.PRIOR.to_numpy(copy=True)
        refvals[idx] = 0.27
        refvals_tf = tf.Variable(refvals, dtype=tf.float64)
        compmap = CompoundMap(dt)
        compmap_tf = CompoundMapTF(dt)
        propvals = compmap.propagate(refvals)
        propvals_tf = compmap_tf(refvals_tf)
        self.assertTrue(np.allclose(propvals, propvals_tf.numpy()))

    def test_proper_jacobian_of_relative_errors(self):
        dt = self._gmadb.get_datatable()
        sel = dt.NODE.str.match('relerr_1027') & (dt.ENERGY == 14.0)
        idx = dt.index[sel]
        assert len(idx) == 1
        refvals = dt.PRIOR.to_numpy(copy=True)
        refvals[idx] = 0.27
        refvals_tf = tf.Variable(refvals, dtype=tf.float64)
        compmap = CompoundMap(dt)
        compmap_tf = CompoundMapTF(dt)
        jac = compmap.jacobian(refvals, with_id=False)
        jac_tf = compmap_tf.jacobian(refvals_tf)
        jac_tf_csr = csr_matrix((jac_tf.values, (
            jac_tf.indices[:, 0], jac_tf.indices[:, 1])), shape=jac_tf.dense_shape
        )
        self.assertTrue(np.allclose(jac.toarray(), jac_tf_csr.toarray()))


if __name__ == '__main__':
    unittest.main()
