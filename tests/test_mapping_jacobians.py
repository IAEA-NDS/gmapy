import unittest
import pathlib
import numpy as np
import pandas as pd

from gmapy.legacy.database_reading import read_gma_database
from gmapy.legacy.conversion_utils import (sanitize_datablock, sanitize_prior)
from gmapy.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapy.data_management.uncfuns import create_relunc_vector
from gmapy.mappings.priortools import attach_shape_prior
from gmapy.mappings.helperfuns import numeric_jacobian

from gmapy.mappings.compound_map import CompoundMap, mapclass_with_params
from gmapy.mappings.cross_section_absolute_ratio_map import CrossSectionAbsoluteRatioMap
from gmapy.mappings.cross_section_fission_average_map import CrossSectionFissionAverageMap
from gmapy.mappings.cross_section_map import CrossSectionMap
from gmapy.mappings.cross_section_ratio_map import CrossSectionRatioMap
from gmapy.mappings.cross_section_ratio_shape_map import CrossSectionRatioShapeMap
from gmapy.mappings.cross_section_shape_map import CrossSectionShapeMap
from gmapy.mappings.cross_section_shape_of_ratio_map import CrossSectionShapeOfRatioMap
from gmapy.mappings.cross_section_shape_of_sum_map import CrossSectionShapeOfSumMap
from gmapy.mappings.cross_section_total_map import CrossSectionTotalMap


class TestMappingJacobians(unittest.TestCase):

    # helper functions for the tests
    def get_error(self, res1, res2, atol=1e-7):
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

    def reduce_table(self, curmapclass, datatable):
        refvals = np.full(len(datatable), 10)
        curmap = curmapclass(datatable)
        Smat = curmap.jacobian(refvals).tocoo()
        idcs1 = np.unique(Smat.col)
        idcs2 = np.unique(Smat.row)
        if (len(set(idcs1)) + len(set(idcs2)) !=
                len(set(np.concatenate([idcs1,idcs2])))):
            raise IndexError('idcs1 and idcs2 must be disjoint')

        sel = np.concatenate([idcs1, idcs2])
        # create filtered datatable and recreate index
        curdatatable = datatable.loc[sel].reset_index(drop=True)
        idcs1 = np.arange(len(idcs1))
        idcs2 = np.arange(len(idcs1), len(idcs1)+len(idcs2))
        return curdatatable, idcs1, idcs2

    def get_jacobian_testerror(self, curmapclass):
        datatable, idcs1, idcs2 = self.reduce_table(curmapclass, self._datatable)
        propfun = self.create_propagate_wrapper(curmapclass, datatable,
                                                idcs1, idcs2)
        curmap = curmapclass(datatable)
        np.random.seed(15)
        x = np.full(len(idcs1)+len(idcs2), 0.)
        x[idcs1] = np.random.uniform(1, 5, len(idcs1))
        res2 = curmap.jacobian(x)
        res1 = numeric_jacobian(propfun, x[idcs1], o=4, h1=1e-2, v=2)
        res2 = np.array(res2.todense())
        res2 = res2[np.ix_(idcs2, idcs1)]
        if np.all(res1 == 0) or np.all(res2 == 0):
            raise ValueError('Some elements be different from zero')

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
        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)

        refvals = datatable['PRIOR']
        uncs = np.full(len(refvals), np.nan)
        expsel = datatable['NODE'].str.match('exp_').to_numpy()
        uncs[expsel] = create_relunc_vector(datablocklist)
        compmap = CompoundMap()
        datatable = attach_shape_prior(datatable, compmap, refvals, uncs)

        cls._datatable = datatable
        cls._datablocklist = datablocklist

    @classmethod
    def tearDownClass(cls):
        del(cls._datatable)
        del(cls._datablocklist)

    def test_cross_section_absolute_ratio_map(self):
        curmapclass = CrossSectionAbsoluteRatioMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_fission_average_map(self):
        for legacy_integration in [False, True]:
            datatable = self._datatable.copy()
            curmapclass = mapclass_with_params(
                CrossSectionFissionAverageMap,
                fix_jacobian=True, legacy_integration=legacy_integration,
                atol=1e-5, rtol=1e-5, maxord=16
            )
            fistable = datatable[datatable['NODE'] == 'fis'].copy()
            datatable, idcs1, idcs2 = self.reduce_table(curmapclass, datatable)
            curmap = curmapclass(datatable)
            if legacy_integration:
                datatable = pd.concat([datatable, fistable], ignore_index=True)

            def propfun(x):
                refvals = orig_x
                refvals[idcs1] = x
                return curmap.propagate(refvals)[idcs2]
            np.random.seed(15)
            x = np.random.uniform(1, 5, len(datatable))
            orig_x = x.copy()
            res1 = numeric_jacobian(propfun, x[idcs1], o=4, h1=1e-2, v=2)
            res2 = curmap.jacobian(x)
            res2 = res2.toarray()[np.ix_(idcs2, idcs1)]
            relerr = self.get_error(res1, res2, atol=1e-4)
            msg = (f'Maximum relative error in SACS Jacobian is {relerr}' +
                   f'for legacy_integration={legacy_integration}')
            self.assertTrue(np.all(np.isclose(res1, res2, rtol=1e-4, atol=1e-4)), msg)

    def test_cross_section_map(self):
        curmapclass = CrossSectionMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_ratio_map(self):
        curmapclass = CrossSectionRatioMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_ratio_shape_map(self):
        curmapclass = CrossSectionRatioShapeMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_map(self):
        curmapclass = CrossSectionShapeMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_of_ratio_map(self):
        curmapclass = CrossSectionShapeOfRatioMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_shape_of_sum_map(self):
        curmapclass = CrossSectionShapeOfSumMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    def test_cross_section_total_map(self):
        curmapclass = CrossSectionTotalMap
        relerr, res1, res2 = self.get_jacobian_testerror(curmapclass)
        self.assertLess(relerr, 1e-8)

    # permutation tests
    def test_permutation_invariance_of_cross_section_fission_average_map(self):
        for legacy_integration in [False, True]:
            datatable = self._datatable.copy()
            curmapclass = mapclass_with_params(
                CrossSectionFissionAverageMap,
                fix_jacobian=True, legacy_integration=legacy_integration,
                atol=1e-5, rtol=1e-5, maxord=10
            )
            fistable = datatable[datatable['NODE'] == 'fis'].copy()
            datatable, idcs1, idcs2 = self.reduce_table(curmapclass, datatable)
            if legacy_integration:
                datatable = pd.concat([datatable, fistable], ignore_index=True)
            permdt = datatable.reindex(np.random.permutation(datatable.index))
            curmap = curmapclass(datatable)
            perm_curmap = curmapclass(permdt)
            np.random.seed(15)
            x = np.random.uniform(1, 5, len(datatable))
            # we preserve the values of the fission spectrum
            x[len(datatable):] = datatable.loc[len(datatable):, 'PRIOR']
            res1 = curmap.jacobian(x)
            res2 = perm_curmap.jacobian(x)
            self.assertTrue(
                np.allclose(res1.toarray(), res2.toarray()),
                msg=f'jacobian not invariant under datatable permutation for '
                    f'legacy_integration: {legacy_integration}'
            )


if __name__ == '__main__':
    unittest.main()
