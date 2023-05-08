import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.cross_section_map import CrossSectionMap
from gmapy.mappings.cross_section_map_tf import CrossSectionMap as CrossSectionMapTF


class TestCrossSectionMaps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._dbpath = dbpath
        cls._gmadb = GMADatabase(dbpath)

    def test_cross_section_map_tf_equivalence(self):
        for cur_reduce in (False, True):
            dt = self._gmadb.get_datatable()
            xs_map = CrossSectionMap(dt, reduce=cur_reduce)
            xs_map_tf = CrossSectionMapTF(dt, reduce=cur_reduce)
            if not cur_reduce:
                x = dt['PRIOR'].to_numpy()
            else:
                x = dt.loc[~dt.NODE.str.match('exp_'), 'PRIOR'].to_numpy()
            x_tf = tf.Variable(x, dtype=tf.float64)
            res = xs_map.propagate(x)
            with tf.GradientTape() as tape:
                res_tf = xs_map_tf(x_tf)
            self.assertTrue(np.allclose(res, res_tf.numpy()))
            jac = xs_map.jacobian(x).toarray()
            jac_tf = tape.jacobian(res_tf, x_tf).numpy()
            self.assertTrue(np.allclose(jac, jac_tf))


if __name__ == '__main__':
    unittest.main()
