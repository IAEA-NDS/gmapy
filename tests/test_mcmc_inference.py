import unittest
import numpy as np
import os
import pathlib
from scipy.sparse import csc_matrix
import scipy.sparse as sps
from scipy.stats import norm, uniform, shapiro, ks_2samp
from gmapy.mcmc_inference import (
    mh_algo,
    symmetric_mh_algo,
    parallel_symmetric_mh_algo,
    gmap_mh_inference,
    _get_chain_files
)
from gmapy.posterior import Posterior
from gmapy.mappings.compound_map import CompoundMap
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.helperfuns import numeric_jacobian
import tempfile


class TestMCMCInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        gmadb = GMADatabase(dbpath)
        cls.__gmadb = gmadb
        cls.__datatable = gmadb.get_datatable()
        cls.__covmat = gmadb.get_covmat()

    @classmethod
    def tearDownClass(cls):
        del cls.__gmadb
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
        def proposal(x, random_state=None):
            return np.array(x + norm.rvs(
                loc=0, scale=scale, random_state=random_state)
            )
        return proposal

    def create_proposal_unif(self, hw):
        def proposal(x, random_state=None):
            return np.array(x + uniform.rvs(
                loc=-hw, scale=2*hw, random_state=random_state)
            )
        return proposal

    def create_log_probdens(self, loc, scale):
        def log_probdens(x):
            return norm.logpdf(x, loc=loc, scale=scale)
        return log_probdens

    def test_symmetric_mh_algo_with_unif_proposal_on_normal_target(self):
        proposal = self.create_proposal_unif(1.5)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res = symmetric_mh_algo(
                startvals, log_probdens, proposal, 11000,
                save_dir=tmpdir, seed=510999
            )
        samples = mh_res['samples']
        # throw away burn in samples
        samples = samples[:, 1000:].flatten()
        # shapiro test seems to be very sensitive to
        # auto-correlation so thin the chain
        samples = samples[::10]
        testres = shapiro(samples)
        self.assertTrue(testres.pvalue >= 0.05)

    def test_symmetric_mh_algo_with_unif_and_norm_proposal(self):
        proposal1 = self.create_proposal_unif(1.5)
        proposal2 = self.create_proposal_gauss(0.7)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res1 = symmetric_mh_algo(
                startvals, log_probdens, proposal1, 11000, save_dir=tmpdir,
                seed=938272
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res2 = symmetric_mh_algo(
                startvals, log_probdens, proposal2, 11000, save_dir=tmpdir,
                seed=2232
            )
        samples1 = mh_res1['samples']
        samples2 = mh_res2['samples']
        samples1 = samples1[:, 1000::10].flatten()
        samples2 = samples2[:, 1000::10].flatten()
        testres = ks_2samp(samples1, samples2)
        self.assertTrue(testres.pvalue >= 0.05)

    def test_single_parallel_symmetric_mh_algo_equivalence(self):
        proposal = self.create_proposal_gauss(0.7)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res1a = symmetric_mh_algo(
                startvals, log_probdens, proposal, 1000, save_dir=tmpdir,
                seed=123, save_batchsize=137
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res1b = symmetric_mh_algo(
                startvals, log_probdens, proposal, 1000, save_dir=tmpdir,
                seed=456, save_batchsize=229
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res2 = parallel_symmetric_mh_algo(
                2, startvals, log_probdens, proposal, 1000,
                save_dir=tmpdir, seeds=[123, 456],
                save_batchsize=197
            )
        samples1a = mh_res1a['samples']
        samples1b = mh_res1b['samples']
        samples2a = mh_res2[0]['samples']
        samples2b = mh_res2[1]['samples']
        self.assertTrue(np.all(samples1a == samples2a))
        self.assertTrue(np.all(samples1b == samples2b))

    def test_resume_capability_works_correctly(self):
        proposal = self.create_proposal_gauss(0.7)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res1 = symmetric_mh_algo(
                startvals, log_probdens, proposal, 1000, save_dir=tmpdir,
                seed=123, save_batchsize=137
            )
            chain_files = _get_chain_files('mh_res_', '.pkl', tmpdir)
            for frem in chain_files[-3:]:
                rempath = os.path.join(tmpdir, frem)
                os.remove(rempath)
            mh_res2 = symmetric_mh_algo(
                startvals, log_probdens, proposal, 1000, save_dir=tmpdir,
                seed=123, save_batchsize=127, attempt_resume=True
            )
        samples1 = mh_res1['samples']
        samples2 = mh_res2['samples']
        self.assertTrue(np.all(samples1 == samples2))
        logprob_hist1 = mh_res1['logprob_hist']
        logprob_hist2 = mh_res2['logprob_hist']
        self.assertTrue(np.all(logprob_hist1 == logprob_hist2))
        accept_rate1 = mh_res1['accept_rate']
        accept_rate2 = mh_res2['accept_rate']
        self.assertTrue(np.isclose(accept_rate1, accept_rate2))

    def test_batchsize_does_not_matter(self):
        proposal = self.create_proposal_gauss(0.7)
        log_probdens = self.create_log_probdens(1, 1.5)
        startvals = np.array([0.])
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res1 = mh_algo(
                startvals, log_probdens, proposal, 500, save_dir=tmpdir,
                save_batchsize=211, seed=549
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            mh_res2 = mh_algo(
                startvals, log_probdens, proposal, 500, save_dir=tmpdir,
                save_batchsize=147, seed=549
            )
        samples1 = mh_res1['samples']
        samples2 = mh_res2['samples']
        self.assertTrue(np.all(samples1 == samples2))

    def test_gradient_of_posterior_logpdf(self):
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.array([1, 2, 3]).reshape(-1, 1)
        post.set_squeeze(False)
        testres = post.grad_logpdf(xref)
        refres = numeric_jacobian(post.logpdf, np.squeeze(xref))
        self.assertTrue(np.allclose(testres, refres.T))

    def test_gradient_of_posterior_logpdf_with_zero_priorunc(self):
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[1, :] = 0.
        priorcov[:, 1] = 0.
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        post.set_squeeze(False)
        xref = priorvals.copy() + np.array([1, 0, 3]).reshape(-1, 1)
        testres = post.grad_logpdf(xref)
        refres = numeric_jacobian(post.logpdf, np.squeeze(xref))
        self.assertTrue(np.allclose(testres, refres.T))

    def test_covmat_of_proposal_distribution(self):
        rg = np.random.Generator(np.random.PCG64(3989))
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
        samples = propfun(np.zeros((S.shape[1], 10000)), random_state=rg)
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(samplecov/(scale_fact**2), postcov, atol=0.5))

    def test_covmat_of_proposal_distribution_with_zero_prior_uncertainty(self):
        rg = np.random.Generator(np.random.PCG64(3990))
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[:,1] = 0.
        priorcov[1,:] = 0.
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.random.normal(size=priorvals.shape)
        xref[1] = priorvals[1]
        scale_fact = 1.
        propfun = post.generate_proposal_fun(xref, scale=scale_fact)
        # reference calculation
        postcov = post.approximate_postcov(xref)
        samples = propfun(np.zeros((priorvals.size, 10000), dtype=float),
                          random_state=rg)
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(postcov.toarray(), samplecov,
                        atol=0.5, rtol=0.2))

    def test_covmat_of_proposal_distribution_using_gma_database(self):
        rg = np.random.Generator(np.random.PCG64(3990))
        # retrieve relevant data structures
        gmadb = self.__gmadb
        dt = gmadb.get_datatable()
        m = CompoundMap(dt, reduce=True)
        priorvals = gmadb.get_priorvals()
        expvals = gmadb.get_expvals()
        priorcov = gmadb.get_priorcov()
        expcov = gmadb.get_expcov()
        # prepare MH prerequisites
        post = Posterior(priorvals, priorcov, m, expvals, expcov)
        propfun = post.generate_proposal_fun(priorvals, scale=1.)
        postcov = post.approximate_postcov(priorvals).toarray()
        # do the sampling
        num_samples = 10000
        samples = np.zeros((priorvals.size, num_samples), dtype=float)
        for i in range(num_samples):
            samples[:,i:i+1] = propfun(priorvals, random_state=rg)
        samplecov = np.cov(samples)
        print(samplecov)
        print(postcov)
        self.assertTrue(np.allclose(samplecov, postcov,
                                    atol=0.5, rtol=0.5))

    def test_reproducibility_of_posterior_class_proposal(self):
        # retrieve relevant data structures
        gmadb = self.__gmadb
        dt = gmadb.get_datatable()
        m = CompoundMap(dt, reduce=True)
        priorvals = gmadb.get_priorvals()
        expvals = gmadb.get_expvals()
        priorcov = gmadb.get_priorcov()
        expcov = gmadb.get_expcov()
        # prepare MH prerequisites
        post = Posterior(priorvals, priorcov, m, expvals, expcov)
        propfun = post.generate_proposal_fun(priorvals, scale=1.)
        rg = np.random.Generator(np.random.PCG64(3990))
        num_samples = 20
        samples1 = np.zeros((priorvals.size, num_samples), dtype=float)
        for i in range(num_samples):
            samples1[:,i:i+1] = propfun(priorvals, random_state=rg)
        rg = np.random.Generator(np.random.PCG64(3990))
        samples2 = np.zeros((priorvals.size, num_samples), dtype=float)
        for i in range(num_samples):
            samples2[:,i:i+1] = propfun(priorvals, random_state=rg)
        self.assertTrue(np.all(samples1 == samples2))

    def test_effect_of_scale_argument_on_proposal(self):
        rg = np.random.Generator(np.random.PCG64(1990))
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        post = Posterior(priorvals, priorcov, mockmap, expvals, expcov)
        xref = priorvals + np.random.normal(size=priorvals.size)
        propfun = post.generate_proposal_fun(xref, scale=1e-6)
        propvals = propfun(xref, random_state=rg)
        self.assertTrue(np.allclose(propvals, xref, atol=1e-4))

    def test_implementation_of_adjustable_option(self):
        rg = np.random.Generator(np.random.PCG64(1892))
        priorvals, priorcov, mockmap, expvals, expcov = \
            self.create_mock_data()
        priorcov[1,:] = 0
        priorcov[:,1] = 0
        post = Posterior(
            priorvals, priorcov, mockmap, expvals, expcov
        )
        xref = np.full(priorvals.shape, 0.)
        propfun = post.generate_proposal_fun(xref)
        samples = propfun(np.zeros((xref.shape[0], 10000)),
                          random_state=rg)
        postcov = post.approximate_postcov(xref).toarray()
        samplecov = np.cov(samples)
        self.assertTrue(np.allclose(postcov, samplecov, atol=1.))
        self.assertTrue(np.all(samplecov[postcov == 0] == 0))

    @unittest.skip
    def test_gmap_mh_inference(self):
        rg = np.random.Generator(np.random.PCG64(1330))
        datatable = self.__datatable
        covmat = self.__covmat
        mh_res = gmap_mh_inference(datatable, covmat, 200, 0.05, num_burn=100)
        # TODO: something better to test than acceptance rate
        #       that does not take too long?
        self.assertTrue(mh_res[0]['accept_rate'] > 0.2)


if __name__ == '__main__':
    unittest.main()
