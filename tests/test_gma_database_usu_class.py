import unittest
import pathlib
import pandas as pd
from gmapi.gma_database_usu_class import GMADatabaseUSU
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky
from gmapi.mappings.helperfuns import numeric_jacobian

from gmapi.optimization_auxfuns import create_posterior_funs


# NOTE: This test case tests at present tests mostly the functionality
#       of the USUErrorMap class regarding the calculation of
#       the log likelihood and gradients thereof.

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

    def test_grad_logdet_computation(self):
        gmadb = deepcopy(self._gmadb)
        gmadb.set_usu_components(['REAC'])
        datatable = gmadb.get_datatable()
        mapping = gmadb.get_mapping()
        refvals = datatable['PRIOR'].to_numpy()
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        usu_uncs = np.linspace(1, 50, num=len(usu_idcs))
        # we only calculate some elements of the gradient
        # to keep the computation time reasonable for a test
        optim_idcs = np.array([3, 9, 27])
        # specify the USU uncertainty values
        counter = 0
        def logdet_wrap(x):
            nonlocal counter
            nonlocal usu_uncs
            counter += 1
            print('call number: ' + str(counter))
            cur_usu_uncs = usu_uncs.copy()
            cur_usu_uncs[optim_idcs] = x
            datatable.loc[usu_idcs, 'UNC'] = cur_usu_uncs
            gmadb.set_datatable(datatable)
            covmat = gmadb.get_covmat()
            res = mapping.logdet(datatable, refvals, covmat)
            return np.ravel(res)
        # call logdet_wrap once to initialize USU uncertainties
        logdet_wrap(usu_uncs[optim_idcs])
        covmat = gmadb.get_covmat()
        res1 = mapping.grad_logdet(datatable, refvals, covmat)
        # now do it by numerical differentiation
        res2 = numeric_jacobian(logdet_wrap, usu_uncs[optim_idcs])
        red_res1 = res1[optim_idcs]
        self.assertTrue(np.allclose(red_res1, res2))

    def test_grad_chisquare_computation(self):
        gmadb = deepcopy(self._gmadb)
        gmadb.set_usu_components(['REAC'])
        datatable = gmadb.get_datatable()
        mapping = gmadb.get_mapping()
        refvals = datatable['PRIOR'].to_numpy()
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        usu_uncs = np.linspace(1, 50, num=len(usu_idcs))
        expvals = datatable.DATA.to_numpy()
        # we only calculate some elements of the gradient
        # to keep the computation time reasonable for a test
        optim_idcs = np.array([3, 9, 27])
        # specify the USU uncertainty values
        counter = 0
        def chisquare_wrap(x):
            nonlocal counter
            nonlocal usu_uncs
            nonlocal expvals
            counter += 1
            print('call number: ' + str(counter))
            cur_usu_uncs = usu_uncs.copy()
            cur_usu_uncs[optim_idcs] = x
            datatable.loc[usu_idcs, 'UNC'] = cur_usu_uncs
            gmadb.set_datatable(datatable)
            covmat = gmadb.get_covmat()
            res = mapping.chisquare(datatable, refvals, expvals, covmat)
            return np.ravel(res)
        # call chisquare_wrap once to initialize USU uncertainties
        chisquare_wrap(usu_uncs[optim_idcs])
        covmat = gmadb.get_covmat()
        res1 = mapping.grad_chisquare(datatable, refvals, expvals, covmat)
        # now do it by numerical differentiation
        res2 = numeric_jacobian(chisquare_wrap, usu_uncs[optim_idcs])
        red_res1 = res1[optim_idcs]
        self.assertTrue(np.allclose(red_res1, res2))

    def test_grad_loglikelihood_computation(self):
        gmadb = deepcopy(self._gmadb)
        gmadb.set_usu_components(['REAC'])
        datatable = gmadb.get_datatable()
        mapping = gmadb.get_mapping()
        refvals = datatable['PRIOR'].to_numpy()
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        usu_uncs = np.linspace(1, 50, num=len(usu_idcs))
        expvals = datatable.DATA.to_numpy()
        # we only calculate some elements of the gradient
        # to keep the computation time reasonable for a test
        optim_idcs = np.array([3, 9, 27])
        # specify the USU uncertainty values
        counter = 0
        def loglikelihood_wrap(x):
            nonlocal counter
            nonlocal usu_uncs
            nonlocal expvals
            counter += 1
            print('call number: ' + str(counter))
            cur_usu_uncs = usu_uncs.copy()
            cur_usu_uncs[optim_idcs] = x
            datatable.loc[usu_idcs, 'UNC'] = cur_usu_uncs
            gmadb.set_datatable(datatable)
            covmat = gmadb.get_covmat()
            res = mapping.loglikelihood(datatable, refvals, expvals, covmat)
            return np.ravel(res)
        # call loglikelihood_wrap once to initialize USU uncertainties
        loglikelihood_wrap(usu_uncs[optim_idcs])
        covmat = gmadb.get_covmat()
        res1 = mapping.grad_loglikelihood(datatable, refvals, expvals, covmat)
        # now do it by numerical differentiation
        res2 = numeric_jacobian(loglikelihood_wrap, usu_uncs[optim_idcs])
        red_res1 = res1[optim_idcs]
        self.assertTrue(np.allclose(red_res1, res2))

    def test_evaluation_with_usu_converges(self):
        gmadb = deepcopy(self._gmadb)
        dt = gmadb.get_datatable()
        # add the usu group feature
        expdt = dt[dt.NODE.str.match('exp_', na=False)]
        energy_group = expdt.apply(lambda x: 'EN1' if x['ENERGY'] < 5 else 'EN2', axis=1)
        ds_group = expdt.apply(lambda x: x['NODE'].split('_')[1], axis=1)
        groupassoc = energy_group + '_' + ds_group
        dt['usu_feat'] = groupassoc
        gmadb.set_datatable(dt)
        gmadb.set_usu_components(['usu_feat'], ['feat'])
        dt = gmadb.get_datatable()
        # add the coupling of uncertainties
        usudt = dt[dt.NODE.str.match('usu_', na=False)]
        energy_group = usudt.apply(lambda x: x['usu_feat'].split('_')[0], axis=1)
        dt['usu_coupling'] = energy_group
        gmadb.set_datatable(dt)
        gmadb.set_usu_couplings('usu_coupling')
        #gmadb.determine_usu_uncertainties()
        gmadb.evaluate(print_status=True, adjust_usu=True, rtol=1e-6, atol=1e-6)
        self.assertTrue(gmadb._cache['converged'])

    # NOTE: At present this test is at present not really related
    # to the GMADatabaseUSU class but is here for convenience
    # of a programmer that does not want to move a small
    # tiny single test to somewhere else.
    def test_logposterior_and_gradient_coherence(self):
        gmadb = deepcopy(self._gmadb)
        dt = gmadb.get_datatable()
        postobj = create_posterior_funs(gmadb.get_mapping(),
                gmadb.get_datatable(), gmadb.get_covmat(), fnscale=-1)
        def logpost_wrap(x):
            cur_refvals = refvals.copy()
            cur_refvals[red_adj_idcs] = x
            res = postobj['logposterior'](cur_refvals[postobj['adj_idcs']])
            return np.ravel(res)

        red_adj_idcs = postobj['adj_idcs'][:3]
        refvals = dt.PRIOR.to_numpy()
        res = postobj['logposterior'](refvals[postobj['adj_idcs']])
        grad = postobj['grad_logposterior'](refvals[postobj['adj_idcs']])
        numgrad = numeric_jacobian(logpost_wrap, refvals[red_adj_idcs])
        self.assertTrue(np.allclose(numgrad, grad[red_adj_idcs]))


if __name__ == '__main__':
    unittest.main()

