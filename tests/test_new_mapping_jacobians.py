"""
Testing Jacobians of new mappings beyond legacy GMAP.
"""

import unittest
import pathlib
import numpy as np
import pandas as pd

from gmapi.mappings.helperfuns import numeric_jacobian
from gmapi.data_management.database_IO import read_legacy_gma_database
from gmapi.data_management.uncfuns import create_relunc_vector
from gmapi.data_management.dataset import Dataset
from gmapi.data_management.datablock import Datablock
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.mappings.cross_section_ratio_of_sacs_map import CrossSectionRatioOfSacsMap


class TestNewMappingJacobians(unittest.TestCase):

    # helper functions for the tests
    def get_error(self, res1, res2, atol=1e-7):
        relerr = np.max(np.abs(res1 - res2) / (np.abs(res2) + atol))
        return relerr

    def create_propagate_wrapper(self, curmap, datatable, idcs1, idcs2):
        """Create propagate wrapper with refvals arg being first."""
        def wrapfun(vals):
            allvals = np.full(len(datatable), 0.)
            allvals[idcs1] = vals
            return curmap.propagate(datatable, allvals)[idcs2]
        return wrapfun

    def reduce_table(self, curmap, datatable):
        refvals = np.full(len(datatable), 10)
        Sdic = curmap.jacobian(datatable, refvals, ret_mat=False)
        idcs1 = np.unique(Sdic['idcs1'])
        idcs2 = np.unique(Sdic['idcs2'])
        if (len(set(idcs1)) + len(set(idcs2)) !=
                len(set(np.concatenate([idcs1,idcs2])))):
            raise IndexError('idcs1 and idcs2 must be disjoint')
        # also include the fission spectrum
        idcs3 = datatable[datatable['NODE'] == 'fis'].index

        sel = np.concatenate([idcs1, idcs2, idcs3])
        # create filtered datatable and recreate index
        curdatatable = datatable.loc[sel].reset_index(drop=True)
        idcs1 = np.arange(len(idcs1))
        idcs2 = np.arange(len(idcs1), len(idcs1)+len(idcs2))
        return curdatatable, idcs1, idcs2

    def get_jacobian_testerror(self, curmap):
        datatable, idcs1, idcs2 = self.reduce_table(curmap, self._datatable)
        propfun = self.create_propagate_wrapper(curmap, datatable,
                                                idcs1, idcs2)
        np.random.seed(15)
        x = np.full(len(idcs1)+len(idcs2), 0.)
        x[idcs1] = np.random.uniform(1, 5, len(idcs1))
        res2 = curmap.jacobian(datatable, x, ret_mat=True)
        res1 = numeric_jacobian(propfun, x[idcs1], o=4, h1=1e-2, v=2)
        res2 = np.array(res2.todense())
        res2 = res2[np.ix_(idcs2, idcs1)]
        if np.all(res1 == 0) or np.all(res2 == 0):
            raise ValueError('Some elements be different from zero')

        relerr = self.get_error(res1, res2)
        return (relerr, res1, res2)

    def test_cross_section_ratio_of_sacs_map(self):
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
        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        self._datatable = datatable

        # do the mapping
        curmap = CrossSectionRatioOfSacsMap(rtol=1e-05, atol=1e-05, maxord=20)
        relerr, res1, res2 = self.get_jacobian_testerror(curmap)
        self.assertLess(relerr, 1e-4)


if __name__ == '__main__':
    unittest.main()
