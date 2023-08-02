import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.cross_section_map import CrossSectionMap
from gmapy.mappings.tf.cross_section_map_tf \
    import CrossSectionMap as CrossSectionMapTF
from gmapy.mappings.cross_section_ratio_map import CrossSectionRatioMap
from gmapy.mappings.tf.cross_section_ratio_map_tf \
    import CrossSectionRatioMap as CrossSectionRatioMapTF
from gmapy.mappings.cross_section_shape_map import CrossSectionShapeMap
from gmapy.mappings.tf.cross_section_shape_map_tf \
    import CrossSectionShapeMap as CrossSectionShapeMapTF
from gmapy.mappings.cross_section_absolute_ratio_map \
    import CrossSectionAbsoluteRatioMap
from gmapy.mappings.tf.cross_section_absolute_ratio_map_tf \
    import CrossSectionAbsoluteRatioMap as CrossSectionAbsoluteRatioMapTF
from gmapy.mappings.cross_section_ratio_shape_map \
    import CrossSectionRatioShapeMap
from gmapy.mappings.tf.cross_section_ratio_shape_map_tf \
    import CrossSectionRatioShapeMap as CrossSectionRatioShapeMapTF
from gmapy.mappings.cross_section_shape_of_ratio_map \
    import CrossSectionShapeOfRatioMap
from gmapy.mappings.tf.cross_section_shape_of_ratio_map_tf \
    import CrossSectionShapeOfRatioMap as CrossSectionShapeOfRatioMapTF
from gmapy.mappings.cross_section_shape_of_sum_map \
    import CrossSectionShapeOfSumMap
from gmapy.mappings.tf.cross_section_shape_of_sum_map_tf \
    import CrossSectionShapeOfSumMap as CrossSectionShapeOfSumMapTF
from gmapy.mappings.cross_section_total_map \
    import CrossSectionTotalMap
from gmapy.mappings.tf.cross_section_total_map_tf \
    import CrossSectionTotalMap as CrossSectionTotalMapTF
from gmapy.mappings.cross_section_ratio_of_sacs_map \
    import CrossSectionRatioOfSacsMap
from gmapy.mappings.tf.cross_section_ratio_of_sacs_map_tf \
    import CrossSectionRatioOfSacsMap as CrossSectionRatioOfSacsMapTF
from gmapy.mappings.cross_section_fission_average_map import CrossSectionFissionAverageMap
from gmapy.mappings.tf.cross_section_fission_average_map_tf \
    import CrossSectionFissionAverageMap as CrossSectionFissionAverageMapTF
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
from gmapy.mappings.helperfuns import mapclass_with_params
import time


class TestCrossSectionMapsTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data_and_sacs.json').resolve().as_posix()
        cls._dbpath = dbpath
        cls._gmadb = GMADatabase(dbpath)

    def _test_mapping_tf_equivalence(self, mapping, mapping_tf):
        for cur_reduce in (False, True):
            dt = self._gmadb.get_datatable()
            xs_map = mapping(dt, reduce=cur_reduce)
            xs_map_tf = mapping_tf(dt, reduce=cur_reduce)
            if not cur_reduce:
                x = dt['PRIOR'].to_numpy()
            else:
                x = dt.loc[~dt.NODE.str.match('exp_'), 'PRIOR'].to_numpy()
            x_tf = tf.Variable(x, dtype=tf.float64)
            res = xs_map.propagate(x)
            res_tf = xs_map_tf(x_tf)
            jac = xs_map.jacobian(x).toarray()
            grad = np.sum(jac, axis=0)

            def myscalarfun(x):
                return tf.reduce_sum(xs_map_tf(x))

            with tf.GradientTape() as tape:
                summed_res_tf = myscalarfun(x_tf)
            grad_tf = tape.gradient(summed_res_tf, x_tf).numpy()
            self.assertTrue(np.allclose(res, res_tf.numpy()))
            self.assertTrue(np.allclose(grad, grad_tf))

    def test_cross_section_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionMap, CrossSectionMapTF
        )

    def test_cross_section_ratio_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionRatioMap, CrossSectionRatioMapTF
        )

    def test_cross_section_shape_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionShapeMap, CrossSectionShapeMapTF
        )

    def test_cross_section_absolute_ratio_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionAbsoluteRatioMap, CrossSectionAbsoluteRatioMapTF
        )

    def test_cross_section_ratio_shape_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionRatioShapeMap, CrossSectionRatioShapeMapTF
        )

    def test_cross_section_shape_of_ratio_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionShapeOfRatioMap, CrossSectionShapeOfRatioMapTF
        )

    def test_cross_section_shape_of_sum_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionShapeOfSumMap, CrossSectionShapeOfSumMapTF
        )

    def test_cross_section_total_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionTotalMap, CrossSectionTotalMapTF
        )

    def test_cross_section_fission_average_map_tf_equivalence(self):
        ModernCrossSectionFissionAverageMap = mapclass_with_params(
            CrossSectionFissionAverageMap,
            legacy_integration=False,
        )
        self._test_mapping_tf_equivalence(
            ModernCrossSectionFissionAverageMap, CrossSectionFissionAverageMapTF
        )

    def test_cross_section_ratio_of_sacs_map_tf_equivalence(self):
        self._test_mapping_tf_equivalence(
            CrossSectionRatioOfSacsMap, CrossSectionRatioOfSacsMapTF
        )

    def test_compound_map_tf_equivalence(self):
        dt = self._gmadb.get_datatable()
        xs_map = CompoundMap(dt, reduce=True)
        xs_map_tf = CompoundMapTF(dt, reduce=True)
        x = dt.loc[~dt.NODE.str.match('exp_'), 'PRIOR'].to_numpy()
        x_tf = tf.Variable(x, dtype=tf.float64)
        y_tf = xs_map_tf(x_tf)
        y = xs_map.propagate(x)
        self.assertTrue(np.allclose(y, y_tf.numpy()))

    def test_compound_map_tf_with_shuffled_dataframe(self):
        dt = self._gmadb.get_datatable()
        xs_map = CompoundMap(dt, reduce=False)
        xs_map_tf = CompoundMapTF(dt, reduce=False)
        perm = np.random.permutation(dt.index)
        dt_shuffled = dt.reindex(perm, copy=False)
        dt_shuffled.reset_index(drop=True, inplace=True)
        xs_map_shuffled = CompoundMap(dt_shuffled, reduce=False)
        xs_map_tf_shuffled = CompoundMapTF(dt_shuffled, reduce=False)
        x = dt.PRIOR.to_numpy()
        x_tf = tf.Variable(x, dtype=tf.float64)
        x_shuffled = x[perm]
        x_tf_shuffled = tf.Variable(x_shuffled, dtype=tf.float64)
        y = xs_map.propagate(x)
        y_tf = xs_map_tf(x_tf)
        y_shuffled = xs_map_shuffled.propagate(x_shuffled)
        y_tf_shuffled = xs_map_tf_shuffled(x_tf_shuffled)
        y_rev = np.empty_like(y)
        y_tf_rev = np.empty_like(y_tf)
        y_rev[perm] = y_shuffled
        y_tf_rev[perm] = y_tf_shuffled
        self.assertTrue(np.allclose(y, y_rev))
        self.assertTrue(np.allclose(y_tf, y_tf_rev))
        self.assertTrue(np.allclose(y_rev, y_tf_rev))

    @unittest.skip
    def test_timings_of_propagate_and_gradient(self):
        for cur_reduce in (True,):
            dt = self._gmadb.get_datatable()
            xs_map = CompoundMap(dt, reduce=cur_reduce)
            xs_map_tf = CompoundMapTF(dt, reduce=cur_reduce)
            if not cur_reduce:
                x = dt['PRIOR'].to_numpy()
            else:
                x = dt.loc[~dt.NODE.str.match('exp_'), 'PRIOR'].to_numpy()
            x_tf = tf.Variable(x, dtype=tf.float64)
            st = time.time()
            res = xs_map.propagate(x)
            ft = time.time()
            print(f'time to evaluate the non-tf propagate {ft-st}')
            st = time.time()
            jac = xs_map.jacobian(x).toarray()
            ft = time.time()
            print(f'time to evaluate the non-tf jacobian {ft-st}')
            st = time.time()
            res_tf = xs_map_tf(x_tf)
            ft = time.time()
            print(f'time first calc: {ft-st}')
            st = time.time()
            res_tf = xs_map_tf(x_tf)
            ft = time.time()
            print(f'time second calc: {ft-st}')

            @tf.function
            def lossfun(x):
                return tf.reduce_sum(xs_map_tf(x))

            @tf.function
            def gradfun(x):
                with tf.GradientTape() as tape:
                    y = lossfun(x)
                g = tape.gradient(y, x)
                return g

            st = time.time()
            grad_res = gradfun(x_tf)
            ft = time.time()
            print(f'time for first tf gradient calc {ft-st}')
            st = time.time()
            grad_res = gradfun(x_tf)
            ft = time.time()
            print(f'time for second tf gradient calc {ft-st}')
            self.assertTrue(np.allclose(res, res_tf.numpy()))


if __name__ == '__main__':
    unittest.main()
