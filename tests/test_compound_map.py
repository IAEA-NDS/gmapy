import unittest
import pathlib
import numpy as np
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.compound_map import CompoundMap


class TestCompoundMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._dbpath = dbpath
        cls._gmadb = GMADatabase(dbpath)

    def test_propagate_with_reduce_option(self):
        dt = self._gmadb.get_datatable()
        compmap1 = CompoundMap(dt, reduce=False)
        compmap2 = CompoundMap(dt, reduce=True)
        expsel = dt.NODE.str.match('exp')
        x1 = dt['PRIOR'].to_numpy()
        x1 += 1e-5
        x2 = x1[~expsel]
        res1 = compmap1.propagate(x1)[expsel]
        res2 = compmap2.propagate(x2)
        self.assertTrue(np.allclose(res1, res2))

    def test_jacobian_with_reduce_option(self):
        dt = self._gmadb.get_datatable()
        compmap1 = CompoundMap(dt, reduce=False)
        compmap2 = CompoundMap(dt, reduce=True)
        expsel = dt.NODE.str.match('exp')
        x1 = dt['PRIOR'].to_numpy()
        x1 += 1e-5
        x2 = x1[~expsel]
        jac1 = compmap1.jacobian(x1)[np.ix_(expsel, ~expsel)].toarray()
        jac2 = compmap2.jacobian(x2).toarray()
        self.assertTrue(np.allclose(jac1, jac2))


if __name__ == '__main__':
    unittest.main()
