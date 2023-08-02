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

    def _permute_dataframe(self, dt, refvals, reset_index=False):
        perm = np.random.permutation(dt.index)
        dt = dt.reindex(perm, copy=True)
        if reset_index:
            refvals = refvals[perm]
            dt.reset_index(drop=True, inplace=True)
        return dt, refvals, perm

    def _propagate_both(self, dt, refvals):
        compmap = CompoundMap(dt)
        compmap_tf = CompoundMapTF(dt)
        refvals_tf = tf.Variable(refvals, dtype=tf.float64)
        propvals = compmap.propagate(refvals)
        propvals_tf = compmap_tf(refvals_tf)
        return propvals, propvals_tf.numpy()

    def _test_shuffled_propagation(self, dt, refvals):
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        dt_shuffled, refvals_shuffled, perm = self._permute_dataframe(
            dt, refvals, reset_index=True
        )
        propvals_shuffled, propvals_tf_shuffled = \
            self._propagate_both(dt_shuffled, refvals_shuffled)
        propvals_rev = np.empty_like(propvals)
        propvals_rev[perm] = propvals_shuffled
        propvals_tf_rev = np.empty_like(propvals_tf)
        propvals_tf_rev[perm] = propvals_tf_shuffled
        self.assertTrue(np.allclose(propvals, propvals_rev))
        self.assertTrue(np.allclose(propvals_tf, propvals_tf_rev))
        self.assertTrue(np.allclose(propvals_rev, propvals_tf_rev))

    def _prepare_propagate_endep_usu_errors(self):
        dt = self._gmadb.get_datatable()
        dt = attach_endep_usu_df(
            dt, ['MT:1-R1:8'], [1, 10, 18], [0.1, 0.2, 0.3]
        )
        refvals = dt.PRIOR.to_numpy(copy=True)
        sel = (dt.NODE.str.match('endep_usu_1027') &
               (dt.ENERGY.isin([1.0, 10.0, 18.0])))
        idcs = dt.index[sel]
        assert len(idcs) == 3
        refvals[idcs] = np.array((0.11, 0.34, -0.12), dtype=float)
        return dt, refvals

    def test_proper_propagation_of_endep_usu_errors_shuffled(self):
        dt, refvals = self._prepare_propagate_endep_usu_errors()
        self._test_shuffled_propagation(dt, refvals)

    def test_proper_propagation_of_endep_usu_errors(self):
        dt, refvals = self._prepare_propagate_endep_usu_errors()
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

    def test_proper_propagation_of_endep_usu_errors_permuted(self):
        dt, refvals = self._prepare_propagate_endep_usu_errors()
        dt, refvals, _ = self._permute_dataframe(dt, refvals, reset_index=False)
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

    def test_proper_propagation_of_endep_usu_errors_permuted_and_index_reset(self):
        dt, refvals = self._prepare_propagate_endep_usu_errors()
        dt, refvals, _ = self._permute_dataframe(dt, refvals, reset_index=True)
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

    def _prepare_propagate_relative_errors(self):
        dt = self._gmadb.get_datatable()
        sel = dt.NODE.str.match('relerr_1027') & (dt.ENERGY == 14.0)
        idx = dt.index[sel]
        assert len(idx) == 1
        refvals = dt.PRIOR.to_numpy(copy=True)
        refvals[idx] = 0.27
        return dt, refvals

    def test_proper_propagation_of_relative_errors_shuffled(self):
        dt, refvals = self._prepare_propagate_relative_errors()
        self._test_shuffled_propagation(dt, refvals)

    def test_proper_propagation_of_relative_errors(self):
        dt, refvals = self._prepare_propagate_relative_errors()
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

    def test_proper_propagation_of_relative_errors_permuted(self):
        dt, refvals = self._prepare_propagate_relative_errors()
        dt, refvals, _ = self._permute_dataframe(dt, refvals, reset_index=False)
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

    def test_proper_propagation_of_relative_errors_permuted_and_index_reset(self):
        dt, refvals = self._prepare_propagate_relative_errors()
        dt, refvals, _ = self._permute_dataframe(dt, refvals, reset_index=True)
        propvals, propvals_tf = self._propagate_both(dt, refvals)
        self.assertTrue(np.allclose(propvals, propvals_tf))

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
