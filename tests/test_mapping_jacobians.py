import unittest
import pathlib
import numpy as np
import pandas as pd

from gmapi.legacy.database_reading import read_gma_database
from gmapi.legacy.conversion_utils import (sanitize_datablock, sanitize_prior)
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.data_management.uncfuns import create_relunc_vector
from gmapi.mappings.priortools import attach_shape_prior
from gmapi.mappings.helperfuns import numeric_jacobian

from gmapi.mappings.compound_map import CompoundMap
from gmapi.mappings.cross_section_absolute_ratio_map import CrossSectionAbsoluteRatioMap
from gmapi.mappings.cross_section_fission_average_map import CrossSectionFissionAverageMap
from gmapi.mappings.cross_section_map import CrossSectionMap
from gmapi.mappings.cross_section_ratio_map import CrossSectionRatioMap
from gmapi.mappings.cross_section_ratio_shape_map import CrossSectionRatioShapeMap
from gmapi.mappings.cross_section_shape_map import CrossSectionShapeMap
from gmapi.mappings.cross_section_shape_of_ratio_map import CrossSectionShapeOfRatioMap
from gmapi.mappings.cross_section_shape_of_sum_map import CrossSectionShapeOfSumMap
from gmapi.mappings.cross_section_total_map import CrossSectionTotalMap


class TestMappingJacobians(unittest.TestCase):

    # helper functions for the tests
    def get_error(self, res1, res2):
        absthres = 1e-6
        sgn1 = np.sign(res1)
        sgn2 = np.sign(res2)
        sgn1[sgn1==0] = 1
        sgn2[sgn2==0] = 1

        x1 = sgn1 * np.maximum(np.abs(res1), absthres)
        x2 = sgn2 * np.maximum(np.abs(res2), absthres)
        reldiff = (np.max(np.abs(x1-x2) /
            np.maximum(np.abs(x1), np.abs(x2)))) 
        return reldiff

    def create_propagate_wrapper(self, curmap, priortable, exptable):  
        """Create propagate wrapper with refvals arg being first."""
        def wrapfun(vals):
            return curmap.propagate(priortable, exptable, vals)
        return wrapfun

    def reduce_tables(self, curmap, priortable, exptable):
        refvals = np.full(len(priortable), 10)
        Sdic = curmap.jacobian(priortable, exptable, refvals, ret_mat=False)
        sel1 = np.unique(Sdic['idcs1'])
        sel2 = np.unique(Sdic['idcs2'])
        curpriortable = priortable.loc[sel1].reset_index()
        curexptable = exptable.loc[sel2].reset_index()
        return (curpriortable, curexptable)

    def get_jacobian_testerror(self, curmap):
        priortable, exptable = self.reduce_tables(curmap, self._priortable,
                                                  self._exptable) 
        propfun = self.create_propagate_wrapper(curmap, priortable, exptable)
        np.random.seed(15)
        x = np.random.uniform(1, 5, len(priortable))
        res1 = numeric_jacobian(propfun, x, o=4, h1=1e-2, v=2)
        res2 = curmap.jacobian(priortable, exptable, x, ret_mat=True)
        res2 = np.array(res2.todense())
        relerr = self.get_error(res1, res2)
        return (relerr, res1, res2)

    # set up the data for the tests
    @classmethod
    def setUpClass(cls):
        cls._dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-08-14.gma').resolve().as_posix()
        dbdic = read_gma_database(cls._dbpath)
        priorlist = sanitize_prior(dbdic['APR']) 
        datablocklist = [sanitize_datablock(b)
                for b in dbdic['datablock_list']]
        priortable = create_prior_table(priorlist)
        exptable = create_experiment_table(datablocklist)
        refvals = priortable['PRIOR']
        uncs = create_relunc_vector(datablocklist)
        compmap = CompoundMap()
        priortable = attach_shape_prior(priortable, compmap, exptable,
                refvals, uncs)
        cls._priortable = priortable
        cls._exptable = exptable
        cls._datablocklist = datablocklist

    @classmethod
    def tearDownClass(cls):
        del(cls._priortable)
        del(cls._exptable)
        del(cls._datablocklist)

    def test_cross_section_absolute_ratio_map(self): 
        curmap = CrossSectionAbsoluteRatioMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_fission_average_map(self): 
        curmap = CrossSectionFissionAverageMap() 
        priortable, exptable = self.reduce_tables(curmap, self._priortable,
                                                  self._exptable) 
        fistable = self._priortable.copy()
        fistable = fistable[fistable['NODE']=='fis']
        priortable = pd.concat([priortable, fistable], ignore_index=True)
        numel = len(priortable) - len(fistable)
        def propfun(x):
            refvals = priortable['PRIOR'].to_numpy()
            refvals[:numel] = x
            return curmap.propagate(priortable, exptable, refvals)
        np.random.seed(15)
        x = np.random.uniform(1, 5, numel)
        res1 = numeric_jacobian(propfun, x, o=4, h1=1e-2, v=2)
        res2 = curmap.jacobian(priortable, exptable, x, ret_mat=True)
        res2 = np.array(res2.todense())
        res2 = res2[:, :numel]
        relerr = self.get_error(res1, res2)
        msg = f'Maximum relative error in SACS Jacobian is {relerr}'
        self.assertLess(relerr, 1e-7, msg)

    def test_cross_section_map(self):
        curmap = CrossSectionMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_ratio_map(self):
        curmap = CrossSectionRatioMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_ratio_shape_map(self):
        curmap = CrossSectionRatioShapeMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_map(self):
        curmap = CrossSectionShapeMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_of_ratio_map(self):
        curmap = CrossSectionShapeOfRatioMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_of_sum_map(self):
        curmap = CrossSectionShapeOfSumMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)

    def test_cross_section_total_map(self):
        curmap = CrossSectionTotalMap()
        relerr, res1, res2 = self.get_jacobian_testerror(curmap) 
        self.assertLess(relerr, 1e-8)


if __name__ == '__main__':
    unittest.main()

