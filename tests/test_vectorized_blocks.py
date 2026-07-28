import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from gmapy.mappings.tf.cross_section_fission_average_map_tf import (
    CrossSectionFissionAverageMap
)
from gmapy.mappings.tf.cross_section_ratio_of_sacs_map_tf import (
    CrossSectionRatioOfSacsMap
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestVectorizedBlocks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data-2017-07-26.gma').resolve().as_posix()
        rawdb = read_gma_database(dbpath)
        priortable = create_prior_table(rawdb['prior_list'])
        exptable = create_experiment_table(rawdb['datablock_list'])
        priortable = attach_shape_prior((priortable, exptable))
        cls._compmap = CompoundMapTF((priortable, exptable), reduce=True)
        rng = np.random.default_rng(31)
        cls._x = rng.uniform(1., 2., size=len(priortable))

    def _propagate_via_blocks(self, curmap, x):
        tar_len = curmap._tar_len
        res = np.zeros(tar_len)
        num_blocks = 0
        for block in curmap.vectorized_blocks():
            num_blocks += 1
            rows = block['tar_idcs']
            r, c, v = block['num']
            num = np.zeros(tar_len)
            np.add.at(num, r, v * x[c])
            vals = num[rows]
            if block['den'] is not None:
                r, c, v = block['den']
                den = np.zeros(tar_len)
                np.add.at(den, r, v * x[c])
                vals = vals / den[rows]
            if block['norm_col'] is not None:
                vals = vals * x[block['norm_col']]
            np.add.at(res, rows, vals)
        return res, num_blocks

    def test_blocks_reproduce_propagate_for_algebraic_maps(self):
        x = self._x
        xt = tf.constant(x, dtype=tf.float64)
        tested = 0
        for curmap in self._compmap._maplist:
            if isinstance(curmap, (CrossSectionFissionAverageMap,
                                   CrossSectionRatioOfSacsMap)):
                continue
            ref = curmap.propagate(xt).numpy()
            res, num_blocks = self._propagate_via_blocks(curmap, x)
            self.assertGreater(num_blocks, 0, msg=type(curmap).__name__)
            self.assertTrue(
                np.allclose(res, ref, rtol=1e-12, atol=1e-12),
                msg=f'{type(curmap).__name__}: max abs diff '
                    f'{np.max(np.abs(res - ref))}'
            )
            tested += 1
        # the test database must exercise a representative set of maps
        self.assertGreaterEqual(tested, 5)

    def test_sacs_maps_refuse_vectorization(self):
        for curmap in self._compmap._maplist:
            if isinstance(curmap, (CrossSectionFissionAverageMap,
                                   CrossSectionRatioOfSacsMap)):
                with self.assertRaises(NotImplementedError):
                    list(curmap.vectorized_blocks())


if __name__ == '__main__':
    unittest.main()
