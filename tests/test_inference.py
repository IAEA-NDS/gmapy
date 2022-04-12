import unittest
import pathlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from gmapi.legacy.database_reading import read_gma_database
from gmapi.legacy.conversion_utils import (sanitize_datablock, sanitize_prior)
from gmapi.data_management.uncfuns import (create_relunc_vector,
        create_experimental_covmat)
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.mappings.priortools import attach_shape_prior
from gmapi.inference import gls_update
from gmapi.mappings.compound_map import CompoundMap


class TestInference(unittest.TestCase):

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

    def test_propagation_permutation_invariance(self):
        datatable = self._datatable.copy()
        np.random.seed(25)
        perm = np.random.permutation(len(datatable))
        perm_datatable = datatable.loc[perm].copy()
        comp_map = CompoundMap()
        refvals = datatable['PRIOR'].to_numpy()
        preds1 = comp_map.propagate(datatable, refvals)
        preds2 = comp_map.propagate(perm_datatable, refvals)
        self.assertTrue(np.all(preds1 == preds2))

    def test_jacobian_permutation_invariance(self):
        datatable = self._datatable.copy()
        np.random.seed(27)
        perm = np.random.permutation(len(datatable))
        perm_datatable = datatable.loc[perm].copy()
        comp_map = CompoundMap()
        refvals = datatable['PRIOR'].to_numpy()
        S1 = comp_map.jacobian(datatable, refvals, ret_mat=True)
        S2 = comp_map.jacobian(perm_datatable, refvals, ret_mat=True)
        self.assertTrue(np.all(S1.todense() == S2.todense()))

    def test_inference_permutation_invariance(self):
        datatable = self._datatable.copy()
        datablocklist = self._datablocklist
        compmap = CompoundMap()
        # prepare experimental data and uncertainties
        uncs = np.full(len(datatable), np.nan)
        expsel = datatable['NODE'].str.match('exp_').to_numpy()
        uncs[expsel] = create_relunc_vector(datablocklist)
        expdata = datatable['DATA'].to_numpy()
        # construct the sparse csr covariance matrix
        uncs_red = uncs[expsel]
        expdata_red = expdata[expsel]
        idcs = datatable.index[expsel].to_numpy()
        tmp = create_experimental_covmat(datablocklist, expdata_red, uncs_red)
        tmp = coo_matrix(tmp)
        expcovmat = csr_matrix((tmp.data, (idcs[tmp.row], idcs[tmp.col])),
                               shape = (len(datatable), len(datatable)),
                               dtype=float)
        del(tmp)

        np.random.seed(27)
        perm = np.random.permutation(len(datatable))
        perm_datatable = datatable.loc[perm].copy()
        upd_res1 = gls_update(compmap, datatable, expcovmat, retcov=True)
        upd_res2 = gls_update(compmap, perm_datatable, expcovmat, retcov=True)

        upd_vals_same = np.all(upd_res1['upd_vals'] == upd_res2['upd_vals'])
        upd_covmat_same = np.all(upd_res1['upd_covmat'] == upd_res2['upd_covmat'])
        self.assertTrue(upd_vals_same)
        self.assertTrue(upd_covmat_same)


if __name__ == '__main__':
    unittest.main()

