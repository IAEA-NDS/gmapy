import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)


class TestCompoundMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._dbpath = dbpath
        rawdb = read_gma_database(dbpath)
        priortable = create_prior_table(rawdb['prior_list'])
        exptable = create_experiment_table(rawdb['datablock_list'])
        cls._gmadb = GMADatabase(
            prior_list=rawdb['prior_list'],
            datablock_list=rawdb['datablock_list']
        )

    def test_propagate_with_reduce_option(self):
        dt = self._gmadb.get_datatable()
        compmap1 = CompoundMapTF(dt, reduce=False)
        compmap2 = CompoundMapTF(dt, reduce=True)
        expsel = dt.NODE.str.match('exp')
        x1 = dt['PRIOR'].to_numpy()
        x1 += 1e-5
        x2 = x1[~expsel]
        res1 = compmap1(x1).numpy()[expsel]
        res2 = compmap2(x2).numpy()
        self.assertTrue(np.allclose(res1, res2))

    def test_jacobian_with_reduce_option(self):
        dt = self._gmadb.get_datatable()
        compmap1 = CompoundMapTF(dt, reduce=False)
        compmap2 = CompoundMapTF(dt, reduce=True)
        expsel = dt.NODE.str.match('exp')
        x1 = dt['PRIOR'].to_numpy()
        x1 += 1e-5
        x2 = x1[~expsel]
        jac1 = tf.sparse.to_dense(
            compmap1.jacobian(x1)
        ).numpy()[np.ix_(expsel, ~expsel)]
        jac2 = tf.sparse.to_dense(
            compmap2.jacobian(x2)
        ).numpy()
        self.assertTrue(np.allclose(jac1, jac2))


if __name__ == '__main__':
    unittest.main()
