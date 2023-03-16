import unittest
import numpy as np
import pathlib
from scipy.sparse import csc_matrix
import scipy.sparse as sps
from scipy.stats import norm, uniform, shapiro, ks_2samp
from gmapy.mcmc_inference import (
    symmetric_mh_algo,
    Posterior,
    gmap_mh_inference
)
from gmapy.mappings.compound_map import CompoundMap
from gmapy.inference import lm_update
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.priortools import prepare_prior_and_exptable
from gmapy.mappings.helperfuns import numeric_jacobian


class TestMCMCInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        gmadb = GMADatabase(dbpath)
        cls.__datatable = gmadb.get_datatable()
        cls.__covmat = gmadb.get_covmat()

    @classmethod
    def tearDownClass(cls):
        del cls.__datatable
        del cls.__covmat

    class MockMapping:
        def __init__(self, *dim):
            self.__S = csc_matrix(np.random.rand(*dim))

        def propagate(self, x):
            return self.__S @ x

        def jacobian(self, x=None):
            return self.__S

    def create_mock_data(self):
        mockmap = TestMCMCInference.MockMapping(5, 3)
        p = np.array([[1,2,3,4,5], [4,5.5,6,1,-13],
                      [6.1,1,8,15,6], [8,-13, 4, 12, -1],
                      [20, 15, 0, 8, 4]], dtype=float)
        expcov = csc_matrix(np.array(p @ p.T))
        expvals = np.array([5, 11, 6, 9, 4], dtype=float).reshape(-1,1)
        q = np.array([[10, 2, 4], [3, 7, -1], [2, 1.5, 7.9]], dtype=float)
        priorcov = csc_matrix(q.T @ q)
        priorvals = np.array([-1, 2, -4], dtype=float).reshape(-1,1)
        return priorvals, priorcov, mockmap, expvals, expcov

    def create_proposal_gauss(self, scale):
        def proposal(x):
            return np.array(x + norm.rvs(loc=0, scale=scale))
        return proposal

    def create_proposal_unif(self, hw):
        def proposal(x):
            return np.array(x + uniform.rvs(loc=-hw, scale=2*hw))
        return proposal

    def create_log_probdens(self, loc, scale):
        def log_probdens(x):
            return norm.logpdf(x, loc=loc, scale=scale)
        return log_probdens

    def test_symmetric_mh_algo_with_unif_proposal_on_normal_target(self):
        np.random.seed(510999)
        proposal = self.create_proposal_unif(1.5)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        mh_res = symmetric_mh_algo(startvals, log_probdens, proposal, 11000)
        samples = mh_res['samples']
        # throw away burn in samples
        samples = samples[:, 1000:].flatten()
        # shapiro test seems to be very sensitive to
        # auto-correlation so thin the chain
        samples = samples[::10]
        testres = shapiro(samples)
        self.assertTrue(testres.pvalue >= 0.05)

    def test_symmetric_mh_algo_with_unif_and_norm_proposal(self):
        np.random.seed(938272)
        proposal1 = self.create_proposal_unif(1.5)
        proposal2 = self.create_proposal_gauss(0.7)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        mh_res1 = symmetric_mh_algo(startvals, log_probdens, proposal1, 11000)
        mh_res2 = symmetric_mh_algo(startvals, log_probdens, proposal2, 11000)
        samples1 = mh_res1['samples']
        samples2 = mh_res2['samples']
        samples1 = samples1[:, 1000::10].flatten()
        samples2 = samples2[:, 1000::10].flatten()
        testres = ks_2samp(samples1, samples2)
        self.assertTrue(testres.pvalue >= 0.05)

    def test_evaluation_of_posterior_logpdf(self):
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        # reference evaluation
        xvec = priorvals + np.random.normal(size=priorvals.shape)
        d1 = xvec - priorvals
        d2 = expvals - mockmap.propagate(xvec)
        invpriorcov = sps.linalg.inv(priorcov)
        invexpcov = sps.linalg.inv(expcov)
        refres = (-0.5) * (d2.T @ invexpcov @ d2 + (d1.T @ invpriorcov @ d1)) 
        # test result
        testres = post.logpdf(xvec)
        self.assertTrue(np.isclose(testres, refres))

    def test_gradient_of_posterior_logpdf(self):
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.array([1, 2, 3]).reshape(-1, 1)
        testres = post.grad_logpdf(xref)
        refres = numeric_jacobian(post.logpdf, np.squeeze(xref))
        self.assertTrue(np.allclose(testres, refres.T))

    def test_gradient_of_posterior_logpdf_with_zero_priorunc(self):
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[1, :] = 0.
        priorcov[:, 1] = 0.
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals.copy() + np.array([1, 0, 3]).reshape(-1, 1)
        testres = post.grad_logpdf(xref)
        refres = numeric_jacobian(post.logpdf, np.squeeze(xref))
        self.assertTrue(np.allclose(testres, refres.T))

    def test_covmat_of_proposal_distribution(self):
        np.random.seed(299792)
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.random.normal(size=priorvals.size)
        scale_fact = 0.1
        propfun = post.generate_proposal_fun(xref, scale=scale_fact)
        invpriorcov = sps.linalg.inv(priorcov)
        invexpcov = sps.linalg.inv(expcov)
        S = mockmap.jacobian()
        invpostcov = S.T @ invexpcov @ S + invpriorcov
        postcov = sps.linalg.inv(invpostcov.tocsc()).toarray()
        samples = propfun(np.zeros((S.shape[1], 10000)))
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(samplecov/(scale_fact**2), postcov, atol=0.5))

    def test_covmat_of_proposal_distribution_width_zero_prior_uncertainty(self):
        np.random.seed(299794)
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[:,1] = 0.
        priorcov[1,:] = 0.
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.random.normal(size=priorvals.size)
        xref[1] = priorvals[1]
        scale_fact = 1. 
        propfun = post.generate_proposal_fun(xref, scale=scale_fact)
        # reference calculation
        postcov = post.approximate_covmat(xref)
        samples = propfun(np.zeros((priorvals.size, 10000), dtype=float))
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(postcov.toarray(), samplecov,
                        atol=0.5, rtol=0.2))

    def test_covmat_of_propposal_distribution_using_gma_database(self):
        np.random.seed(299794)
        dt = self.__datatable
        covmat = self.__covmat
        priordt, expdt, _, _ = prepare_prior_and_exptable(
            dt, True, reset_index=False
        )
        prior_idcs = priordt.index
        exp_idcs = expdt.index
        m = CompoundMap(dt, reduce=True)
        priorvals = dt.loc[prior_idcs, 'PRIOR'].to_numpy(copy=True)
        expvals = dt.loc[exp_idcs, 'DATA'].to_numpy(copy=True)
        priorcov = covmat[np.ix_(prior_idcs, prior_idcs)]
        expcov = covmat[np.ix_(exp_idcs, exp_idcs)]
        post = Posterior(priorvals, priorcov, m, expvals, expcov)
        propfun = post.generate_proposal_fun(priorvals, scale=1.)
        postcov = post.approximate_covmat(priorvals).toarray()
        
        num_samples = 10000
        samples = np.zeros((priorvals.size, num_samples), dtype=float) 
        for i in range(num_samples):
            samples[:,i:i+1] = propfun(priorvals)
        samplecov = np.cov(samples)
        print(samplecov)
        print(postcov)
        self.assertTrue(np.allclose(samplecov, postcov,
                                    atol=0.5, rtol=0.5))

    def test_effect_of_scale_argument_on_proposal(self):
        np.random.seed(1660)
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.random.normal(size=priorvals.size)
        propfun = post.generate_proposal_fun(xref, scale=1e-6)
        propvals = propfun(xref)
        self.assertTrue(np.allclose(propvals, xref, atol=1e-4))

    def test_implementation_of_adjustable_option(self):
        np.random.seed(1662)
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[1,:] = 0
        priorcov[:,1] = 0
        post = Posterior(
            priorvals, priorcov, mockmap, expvals, expcov
        )
        xref = np.full(priorvals.shape, 0.) 
        propfun = post.generate_proposal_fun(xref)
        samples = propfun(np.zeros((xref.shape[0], 10000)))
        postcov = post.approximate_covmat(xref).toarray()
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(postcov, samplecov, atol=1.))
        self.assertTrue(np.all(samplecov[postcov == 0] == 0))

    def test_post_logpdf_method(self):
        dt = self.__datatable
        covmat = self.__covmat
        priordt, expdt, _, _ = prepare_prior_and_exptable(
            dt, True, reset_index=False
        )
        prior_idcs = priordt.index
        exp_idcs = expdt.index
        m = CompoundMap(dt, reduce=True)
        priorvals = dt.loc[prior_idcs, 'PRIOR'].to_numpy(copy=True)
        expvals = dt.loc[exp_idcs, 'DATA'].to_numpy(copy=True)
        priorcov = covmat[np.ix_(prior_idcs, prior_idcs)]
        expcov = covmat[np.ix_(exp_idcs, exp_idcs)]
        post = Posterior(priorvals, priorcov, m, expvals, expcov)
        testres = post.logpdf(priorvals)
        invexpcov = sps.linalg.inv(expcov)
        propx = m.propagate(priorvals)
        d = expvals - propx
        # we don't need to add prior contribution
        # because uncertainties are infinite in GMA database
        refres = float((-0.5) * d.T @ invexpcov @ d)
        self.assertTrue(np.isclose(testres, refres))

    def test_gmap_mh_inference(self):
        np.random.seed(123)
        datatable = self.__datatable
        covmat = self.__covmat
        mh_res = gmap_mh_inference(datatable, covmat, 200, 0.05, num_burn=100)
        # TODO: something better to test than acceptance rate
        #       that does not take too long?
        self.assertTrue(mh_res['accept_rate'] > 0.2)


if __name__ == '__main__':
    unittest.main()
