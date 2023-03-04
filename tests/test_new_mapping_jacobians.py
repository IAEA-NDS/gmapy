"""
Testing Jacobians of new mappings beyond legacy GMAP.
"""

import unittest
import numpy as np
import pandas as pd
import pathlib

from gmapy.mappings.helperfuns import numeric_jacobian
from gmapy.mappings.compound_map import mapclass_with_params
from gmapy.data_management.dataset import Dataset
from gmapy.data_management.datablock import Datablock
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.cross_section_ratio_of_sacs_map import (
    CrossSectionRatioOfSacsMap
)
from gmapy import GMADatabase


class TestNewMappingJacobians(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data-2017-08-14.gma').resolve().as_posix()
        gmadb = GMADatabase(dbpath)
        datatable = gmadb.get_datatable()
        cls._base_datatable = datatable
        cls._gmadb = gmadb

    @classmethod
    def tearDownClass(cls):
        del(cls._base_datatable)

    # helper functions for the tests
    def get_error(self, res1, res2, atol=1e-4):
        relerr = np.max(np.abs(res1 - res2) / (np.abs(res2) + atol))
        return relerr

    def create_propagate_wrapper(self, curmapclass, datatable, idcs1, idcs2):
        """Create propagate wrapper with refvals arg being first."""
        def wrapfun(vals):
            allvals = np.full(len(datatable), 0.)
            allvals[idcs1] = vals
            return curmap.propagate(allvals)[idcs2]
        curmap = curmapclass(datatable)
        return wrapfun

    def reduce_table(self, curmapclass, datatable, jacargs=None):
        jacargs = jacargs if jacargs is not None else {}
        refvals = np.full(len(datatable), 10)
        curmap = curmapclass(datatable)
        Smat = curmap.jacobian(refvals, **jacargs).tocoo()
        idcs1 = np.unique(Smat.col)
        idcs2 = np.unique(Smat.row)
        if (len(set(idcs1)) + len(set(idcs2)) !=
                len(set(np.concatenate([idcs1,idcs2])))):
            raise IndexError('idcs1 and idcs2 must be disjoint')
        # also include the fission spectrum
        idcs3 = datatable[datatable['NODE'] == 'fis'].index
        idcs1 = np.sort(np.unique(np.concatenate([idcs1, idcs3])))

        sel = np.concatenate([idcs1, idcs2])
        # create filtered datatable and recreate index
        curdatatable = datatable.loc[sel].reset_index(drop=True)
        idcs1 = np.arange(len(idcs1))
        idcs2 = np.arange(len(idcs1), len(idcs1)+len(idcs2))
        return curdatatable, idcs1, idcs2

    def get_jacobian_testerror(self, datatable, curmapclass, atol=1e-4,
                               jacargs=None):
        jacargs = jacargs if jacargs is not None else {}
        datatable, idcs1, idcs2 = self.reduce_table(
            curmapclass, datatable, jacargs=jacargs)
        propfun = self.create_propagate_wrapper(curmapclass, datatable,
                                                idcs1, idcs2)
        curmap = curmapclass(datatable)
        np.random.seed(15)
        x = np.full(len(idcs1)+len(idcs2), 0.)
        x[idcs1] = np.random.uniform(1, 5, len(idcs1))
        res2 = curmap.jacobian(x, **jacargs)
        res1 = numeric_jacobian(propfun, x[idcs1], o=4, h1=1e-2, v=2)
        res2 = np.array(res2.todense())
        res2 = res2[np.ix_(idcs2, idcs1)]
        if np.all(res1 == 0) or np.all(res2 == 0):
            raise ValueError('Some elements be different from zero')

        relerr = self.get_error(res1, res2, atol=atol)
        abserr = np.max(np.abs(res1-res2))
        return (relerr, abserr, res1, res2)

    def create_ratio_of_sacs_datatable(self):
        # create a dataset with a ratio of sacs measurement (MT 10)
        ds1 = Dataset()
        ds1.define_quantity(10, [1, 2])
        ds1.define_metadata(7259)
        ds1.define_measurements([1.], [3.])
        ds1.add_norm_uncertainty(1.5)
        # create another one with a ratio of sacs measurement
        ds2 = Dataset()
        ds2.define_quantity(10, [2, 3])
        ds2.define_metadata(7260)
        ds2.define_measurements([2.], [5.])
        ds2.add_norm_uncertainty(3.)
        # assemble the datablock
        dblock = Datablock()
        dblock.add_datasets([ds1, ds2])
        datablock_list = [dblock.get_datablock_dic()]
        # create a prior_list
        prior_list = [
          {
            'type': 'legacy-prior-cross-section',
            'ID': 1,
            'CLAB': 'reaction 1',
            'EN': [1, 3, 5, 7],
            'CS': [4, 2, 9, 6]
          },
          {
            'type': 'legacy-prior-cross-section',
            'ID': 2,
            'CLAB': 'reaction 1',
            'EN': [2, 4, 8, 12],
            'CS': [7, 1, 4, 2]
          },
          {
            'type': 'legacy-prior-cross-section',
            'ID': 3,
            'CLAB': 'reaction 1',
            'EN': [2, 4, 8, 12],
            'CS': [7, 1, 4, 2]
          },
          {
            'type': 'legacy-fission-spectrum',
            'ENFIS': [0.5, 2.3, 4.1, 5.8],
            'FIS':   [2,     1,   8,   4]
          }
        ]
        # create the tables
        priortable = create_prior_table(prior_list)
        exptable = create_experiment_table(datablock_list)
        datatable = pd.concat(
            [priortable, exptable], axis=0, ignore_index=True
        )
        return datatable

    def create_relative_error_map_datatable(self):
        dt = self._base_datatable.copy()
        isexp = dt['NODE'].str.match('(exp|norm)_').to_numpy()
        not_isexp = np.logical_not(isexp)
        is_desired_exp = dt['NODE'].str.match('(exp|norm)_(8050|1022)').to_numpy()
        sel = np.logical_or(not_isexp, is_desired_exp)
        dt = dt.loc[sel].copy()
        groupidx_df = dt.groupby('NODE').cumcount().to_frame(name='PTIDX')
        dt = pd.concat([dt, groupidx_df], axis=1)
        # create the relative errors dataframe
        relerr_dt = dt[dt['NODE'].str.match('exp_')][['NODE', 'PTIDX']].copy()
        relerr_dt['NODE'] = relerr_dt['NODE'].str.replace('exp_', 'relerr_', regex=False)
        relerr_dt['PRIOR'] = np.linspace(1, 10, len(relerr_dt))
        # combine the two
        dt = pd.concat([dt, relerr_dt], axis=0, ignore_index=True)
        return dt

    def test_cross_section_ratio_of_sacs_map(self):
        datatable = self.create_ratio_of_sacs_datatable()
        # do the mapping
        curmapclass = mapclass_with_params(
            CrossSectionRatioOfSacsMap,
            rtol=1e-05, atol=1e-05, maxord=20
        )
        relerr, abserr, res1, res2 = self.get_jacobian_testerror(
            datatable, curmapclass, atol=1e-4
        )
        self.assertTrue(relerr < 1e-4 or abserr < 1e-4)

    def test_relative_error_map(self):
        datatable = self.create_relative_error_map_datatable()
        relerr, abserr, res1, res2 = self.get_jacobian_testerror(
            datatable, CompoundMap, jacargs={'with_id': False}
        )
        self.assertTrue(relerr < 1e-8 or abserr < 1e-8)


if __name__ == '__main__':
    unittest.main()
