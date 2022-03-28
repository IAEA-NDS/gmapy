import unittest
import pathlib
import numpy as np

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

    def test_propagation_permutation_invariance(self):
        priortable = self._priortable
        exptable = self._exptable
        np.random.seed(25)
        perm1 = np.random.permutation(len(priortable))
        perm2 = np.random.permutation(len(exptable))
        perm_priortable = priortable.loc[perm1].copy()
        perm_exptable = exptable.loc[perm2].copy()
        comp_map = CompoundMap()
        refvals = priortable['PRIOR'].to_numpy()
        preds1 = comp_map.propagate(priortable, exptable, refvals) 
        preds2 = comp_map.propagate(perm_priortable, perm_exptable, refvals) 
        self.assertTrue(np.all(preds1 == preds2))

    def test_jacobian_permutation_invariance(self):
        priortable = self._priortable
        exptable = self._exptable
        np.random.seed(27)
        perm1 = np.random.permutation(len(priortable))
        perm2 = np.random.permutation(len(exptable))
        perm_priortable = priortable.loc[perm1].copy()
        perm_exptable = exptable.loc[perm2].copy()
        comp_map = CompoundMap()
        refvals = priortable['PRIOR'].to_numpy()
        S1 = comp_map.jacobian(priortable, exptable, refvals, ret_mat=True)
        S2 = comp_map.jacobian(perm_priortable, perm_exptable,
                               refvals, ret_mat=True)
        self.assertTrue(np.all(S1.todense() == S2.todense()))

    def test_inference_permutation_invariance(self):
        compmap = CompoundMap()
        priortable = self._priortable
        exptable = self._exptable
        datablocklist = self._datablocklist
        expdata = exptable['DATA']
        uncs = create_relunc_vector(datablocklist) 
        expcovmat = create_experimental_covmat(datablocklist, expdata, uncs)
        np.random.seed(27)
        perm1 = np.random.permutation(len(priortable))
        perm2 = np.random.permutation(len(exptable))
        perm_priortable = priortable.loc[perm1].copy()
        perm_exptable = exptable.loc[perm2].copy()
        upd_res1 = gls_update(priortable, compmap, exptable, expcovmat, retcov=True)
        upd_res2 = gls_update(perm_priortable, compmap, perm_exptable,
                                  expcovmat, retcov=True)
        upd_vals_same = np.all(upd_res1['upd_vals'] == upd_res2['upd_vals'])
        upd_covmat_same = np.all(upd_res1['upd_covmat'] == upd_res2['upd_covmat'])
        self.assertTrue(upd_vals_same)
        self.assertTrue(upd_covmat_same)


if __name__ == '__main__':
    unittest.main()

