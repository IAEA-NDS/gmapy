import unittest
import pathlib
import pandas as pd
import numpy as np
from scipy.sparse import block_diag, diags
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.data_management.uncfuns import create_experimental_covmat
from gmapi.mappings.priortools import attach_shape_prior, remove_dummy_datasets
from gmapi.inference import gls_update, lm_update
from gmapi.data_management.database_IO import read_legacy_gma_database
from gmapi.mappings.compound_map import CompoundMap
from gmapi.gmap import run_gmap_simplified
from gmapi.data_management.uncfuns import create_relunc_vector



class TestLevenbergMarquardtUpdate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        db_dic = read_legacy_gma_database(dbpath)
        prior_list = db_dic['prior_list']
        datablock_list = db_dic['datablock_list']
        remove_dummy_datasets(datablock_list)

        priortable = create_prior_table(prior_list)
        priorcov = diags(np.square(priortable['UNC']), format='csc')

        exptable = create_experiment_table(datablock_list)
        expcov = create_experimental_covmat(datablock_list)

        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)

        # the following block to prepare all the quantities
        # to call attach_shape_prior
        expsel = datatable['NODE'].str.match('exp_').to_numpy()
        refvals = datatable['PRIOR']
        reluncs = np.full(len(refvals), np.nan)
        reluncs[expsel] = create_relunc_vector(datablock_list)
        compmap = CompoundMap()
        datatable = attach_shape_prior(datatable, compmap, refvals, reluncs)

        shapecov = diags(np.full(len(datatable)-len(priortable)-len(exptable), np.inf), format='csc')
        totcov = block_diag([priorcov, expcov, shapecov], format='csc')
        cls._dbpath = dbpath
        cls._datatable = datatable
        cls._totcov = totcov

    def test_gls_lm_equivalence(self):
        datatable = self._datatable
        totcov = self._totcov
        compmap = CompoundMap()
        res1 = gls_update(compmap, datatable, totcov, retcov=False)
        res2 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-16, maxiter=1, print_status=True)
        self.assertTrue(np.all(np.isclose(res1['upd_vals'], res2['upd_vals'],
            atol=1e-8, rtol=1e-8)))

    def test_iterative_gls_lm_equivalence_with_ppp(self):
        dbpath = self._dbpath
        datatable = self._datatable
        totcov = self._totcov
        compmap = CompoundMap()
        # setting lmb to such a small value renders the
        # LM update steps equivalent to the GLS update
        res1 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-50, maxiter=3, print_status=True, correct_ppp=True)
        # due to different convention of counting we must set
        # num_iter=2 to have in total 3 iterations
        res2 = run_gmap_simplified(dbfile=dbpath, dbtype='legacy',
                num_iter=2, correct_ppp=True, remove_dummy=True)
        resvals1 = res1['upd_vals']
        tbl = res2['table']
        sel = (tbl.NODE.str.match('xsid_') | tbl.NODE.str.match('norm_'))
        resvals2 = res2['table'].loc[sel, 'POST'].to_numpy()
        self.assertTrue(np.allclose(resvals1, resvals2))

    def test_iterative_gls_lm_equivalence_without_ppp(self):
        dbpath = self._dbpath
        datatable = self._datatable
        totcov = self._totcov
        compmap = CompoundMap()
        # setting lmb to such a small value renders the
        # LM update steps equivalent to the GLS update
        res1 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-50, maxiter=3, print_status=True, correct_ppp=False)
        # due to different convention of counting we must set
        # num_iter=2 to have in total 3 iterations
        res2 = run_gmap_simplified(dbfile=dbpath, dbtype='legacy',
                num_iter=2, correct_ppp=False, remove_dummy=True)
        resvals1 = res1['upd_vals']
        tbl = res2['table']
        sel = (tbl.NODE.str.match('xsid_') | tbl.NODE.str.match('norm_'))
        resvals2 = res2['table'].loc[sel, 'POST'].to_numpy()
        self.assertTrue(np.allclose(resvals1, resvals2))

    def test_lm_unique_convergence(self):
        datatable = self._datatable
        totcov = self._totcov
        compmap = CompoundMap()
        res1 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-8, maxiter=10, print_status=True)
        res2 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-4, maxiter=10, print_status=True)
        self.assertTrue(np.all(np.isclose(res1['upd_vals'], res2['upd_vals'],
            atol=1e-8, rtol=1e-8)))

    def test_lm_prior_influence(self):
        datatable = self._datatable
        totcov = self._totcov
        prior_sel = datatable.NODE.str.match('xsid_|norm_').to_numpy()
        priorvals = datatable.PRIOR[prior_sel].to_numpy()
        totcov2 = totcov.copy()
        totcov2_diag = totcov2.diagonal()
        totcov2_diag[prior_sel] = 1e-8
        totcov2.setdiag(0.)
        totcov2 = totcov2 + diags(totcov2_diag)
        compmap = CompoundMap()
        res1 = lm_update(compmap, datatable, totcov, retcov=False,
                lmb=1e-8, maxiter=10, print_status=True)
        res2 = lm_update(compmap, datatable, totcov2, retcov=False,
                lmb=1e-8, maxiter=10, print_status=True)

        diff1 = np.mean(np.abs(res1['upd_vals'] - priorvals))
        diff2 = np.mean(np.abs(res2['upd_vals'] - priorvals))
        # if we have a prior with finite uncertainties compared
        # to one with infinite uncertainties will yield a
        # posterior closer to the prior
        self.assertTrue(diff1 > diff2)
        self.assertTrue(diff1 > 0.04)
        self.assertTrue(diff2 < 1e-4)



if __name__ == '__main__':
    unittest.main()
