import unittest
import pathlib
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.energy_dependent_usu_map import (
    attach_endep_usu_df
)


class TestCompoundMapWithErrorMaps(unittest.TestCase):

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
        sel = dt.NODE.str.match('endep_usu_1027') & (dt.ENERGY == 18.0)
        idx = dt.index[sel]
        assert len(idx) == 1
        propvals1 = compmap.propagate(refvals)
        refvals[idx] = 1.0
        propvals2 = compmap.propagate(refvals)
        tarsel = dt.NODE.str.match('exp_1027') & (dt.ENERGY == 14.0)
        taridx = dt.index[tarsel]
        assert len(taridx) == 1
        self.assertTrue(propvals1[taridx]*1.5 == propvals2[taridx])

    def test_proper_propagation_of_relative_errors(self):
        dt = self._gmadb.get_datatable()
        dt = attach_endep_usu_df(
            dt, ['MT:1-R1:8'], [1, 10, 18], [0.1, 0.2, 0.3]
        )
        refvals = dt.PRIOR.to_numpy(copy=True)
        compmap = CompoundMap(dt)
        sel = dt.NODE.str.match('relerr_1027') & (dt.ENERGY == 14.0)
        idx = dt.index[sel]
        assert len(idx) == 1
        propvals1 = compmap.propagate(refvals)
        refvals[idx] = 1.0
        propvals2 = compmap.propagate(refvals)
        tarsel = dt.NODE.str.match('exp_1027') & (dt.ENERGY == 14.0)
        taridx = dt.index[tarsel]
        assert len(taridx) == 1
        self.assertTrue(propvals1[taridx]*2.0 == propvals2[taridx])

    def test_proper_propagation_of_combined_errors(self):
        dt = self._gmadb.get_datatable()
        dt = attach_endep_usu_df(
            dt, ['MT:1-R1:8'], [1, 10, 18], [0.1, 0.2, 0.3]
        )
        refvals = dt.PRIOR.to_numpy(copy=True)
        compmap = CompoundMap(dt)
        sel1 = dt.NODE.str.match('relerr_1027') & (dt.ENERGY == 14.0)
        idx1 = dt.index[sel1]
        assert len(idx1) == 1
        sel2 = dt.NODE.str.match('endep_usu_1027') & (dt.ENERGY == 18.0)
        idx2 = dt.index[sel2]
        assert len(idx2) == 1
        propvals1 = compmap.propagate(refvals)
        refvals[idx1] = 1.0
        refvals[idx2] = 2.0
        propvals2 = compmap.propagate(refvals)
        tarsel = dt.NODE.str.match('exp_1027') & (dt.ENERGY == 14.0)
        taridx = dt.index[tarsel]
        assert len(taridx) == 1
        self.assertTrue(propvals1[taridx]*3.0 == propvals2[taridx])


if __name__ == '__main__':
    unittest.main()
