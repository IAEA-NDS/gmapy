import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.compound_map_tf \
    import CompoundMap as CompoundMapTF


class TestCompoundMapWithErrorMapsTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._dbpath = dbpath
        cls._gmadb = GMADatabase(dbpath, use_relative_errors=True)

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


if __name__ == '__main__':
    unittest.main()
