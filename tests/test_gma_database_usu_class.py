import unittest
import pathlib
import pandas as pd
from gmapi.gma_database_usu_class import GMADatabaseUSU
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky



class TestGMADatabaseUSU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._gmadb = GMADatabaseUSU(dbpath)

    def test_set_usu_components_does_not_alter_sth_else(self):
        gmadb = deepcopy(self._gmadb)
        orig_dt = gmadb.get_datatable()
        orig_cov = gmadb.get_covmat()
        gmadb.set_usu_components(['REAC'])
        new_dt = gmadb.get_datatable()
        new_cov = gmadb.get_covmat()
        self.assertTrue(orig_dt.equals(new_dt.loc[:len(orig_dt)-1]))
        red_new_cov = new_cov[:,:len(orig_dt)][:len(orig_dt),:]
        self.assertTrue((orig_cov != red_new_cov).nnz == 0.)

    def test_logdet_computation_in_two_ways(self):
        gmadb = deepcopy(self._gmadb)
        gmadb.set_usu_components(['REAC'])
        datatable = gmadb.get_datatable()
        covmat = gmadb.get_covmat()
        mapping = gmadb.get_mapping()
        refvals = datatable['PRIOR'].to_numpy()
        # set the USU covmat element to something finite
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        diagvals = np.linspace(1, 50, num=len(usu_idcs))
        covmat[usu_idcs, usu_idcs] = diagvals
        # this function applies the log determinant lemma
        res1 = mapping.logdet(datatable, refvals, covmat)
        # without determinant lemma
        Susu = mapping.jacobian(datatable, refvals, ret_mat=True, only_usu=True)
        exp_idcs = datatable.index[datatable.NODE.str.match('exp_')]
        alt_covmat = covmat + Susu @ covmat @ Susu.T
        alt_covmat = alt_covmat[exp_idcs,:][:,exp_idcs]
        alt_covmat_fact = cholesky(alt_covmat.tocsc())
        res2 = alt_covmat_fact.logdet()
        self.assertTrue(np.allclose(res1, res2))

    def test_chisquare_computation_in_two_ways(self):
        gmadb = deepcopy(self._gmadb)
        gmadb.set_usu_components(['REAC'])
        datatable = gmadb.get_datatable()
        covmat = gmadb.get_covmat()
        mapping = gmadb.get_mapping()
        refvals = datatable['PRIOR'].to_numpy()
        # set the USU covmat element to something finite
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        diagvals = np.linspace(1, 50, num=len(usu_idcs))
        covmat[usu_idcs, usu_idcs] = diagvals
        # this function applies the Woodbury identity
        expvals = datatable.DATA.to_numpy()
        res1 = mapping.chisquare(datatable, refvals, expvals, covmat)
        # direct computation
        Susu = mapping.jacobian(datatable, refvals, ret_mat=True, only_usu=True)
        exp_idcs = datatable.index[datatable.NODE.str.match('exp_')]
        alt_covmat = covmat + Susu @ covmat @ Susu.T
        alt_covmat = alt_covmat[exp_idcs,:][:,exp_idcs]
        alt_covmat_fact = cholesky(alt_covmat.tocsc())
        propcss = mapping.propagate(datatable, refvals, only_usu=False)
        d = (expvals-propcss)[exp_idcs]
        res2 = d.T @ alt_covmat_fact(d)
        self.assertTrue(np.allclose(res1, res2))


if __name__ == '__main__':
    unittest.main()

