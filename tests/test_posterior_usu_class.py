import unittest
from gmapy.posterior_usu import PosteriorUSU
from gmapy.mappings.helperfuns import numeric_jacobian
import numpy as np
import scipy.sparse as sps
from scipy.stats import multivariate_normal


class TestPosteriorUSUClass(unittest.TestCase):

    def create_mock_quantities(self):
        saved_state = np.random.get_state()
        np.random.seed(44)
        priorvals = np.random.rand(10) + 5
        tmp = np.random.rand(10, 10)
        priorcov = tmp.T @ tmp
        priorcov = sps.csr_matrix(priorcov)
        expvals = np.random.rand(20) + 10
        tmp = np.random.rand(20, 20)
        expcov = tmp.T @ tmp
        expcov = sps.csr_matrix(expcov)
        S = np.random.rand(len(expvals), len(priorvals))
        S = sps.csr_matrix(S)

        class MockMap:

            def propagate(self, x):
                return S @ np.square(x)

            def jacobian(self, x):
                return sps.csr_matrix(S.toarray() * 2 * x.reshape(1, -1))

        mock_map = MockMap()
        np.random.set_state(saved_state)
        return priorvals, priorcov, mock_map, expvals, expcov

    class MockMapLinear:

        def __init__(self, yref, S, xref):
            self.yref = yref.reshape(-1, 1)
            self.S = sps.csr_matrix(S)
            self.xref = xref.reshape(-1, 1)

        def propagate(self, x):
            x = x.reshape(-1, 1)
            return self.yref + self.S @ (x - self.xref)

        def jacobian(self, x):
            return self.S

    def compute_reference_logpdf(
        self, priorvals, priorcov, mapping, expvals, expcov, x
    ):
        x = x.reshape(-1, 1)
        priorvals = priorvals.reshape(-1, 1)
        expvals = expvals.reshape(-1, 1)
        if sps.issparse(priorcov):
            priorcov = priorcov.toarray()
        if sps.issparse(expcov):
            expcov = expcov.toarray()
        preds = mapping.propagate(x.flatten()).reshape(-1, 1)
        d1 = x - priorvals
        d2 = expvals - preds
        inv_priorcov = np.linalg.inv(priorcov)
        inv_expcov = np.linalg.inv(expcov)
        logdet_priorcov = np.linalg.slogdet(priorcov)[1]
        logdet_expcov = np.linalg.slogdet(expcov)[1]
        chisqr_prior = d1.T @ inv_priorcov @ d1
        chisqr_exp = d2.T @ inv_expcov @ d2
        N1 = len(d1)
        N2 = len(d2)
        log_normconst1 = (np.pi*N1 + logdet_priorcov)
        log_normconst2 = (np.pi*N2 + logdet_expcov)
        prior_like = -0.5 * (chisqr_prior + log_normconst1)
        exp_like = -0.5 * (chisqr_exp + log_normconst2)
        ret = prior_like + exp_like
        return ret.flatten()

    def test_correct_computation_of_logpdf(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_group_A = 4.0
        unc_group_B = 2.5
        priorcov = priorcov.toarray()
        priorcov[:, unc_idcs] = 0.
        priorcov[unc_idcs, :] = 0.
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)
        testx = np.random.rand(len(priorvals))
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))
        # and another time...
        unc_group_A = 23.0
        unc_group_B = 29.0
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)
        testx = np.random.rand(len(priorvals))
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_correct_computation_of_logpdf_with_ppp(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_group_A = 4.0
        unc_group_B = 2.5
        priorcov = priorcov.toarray()
        priorcov[:, unc_idcs] = 0.
        priorcov[unc_idcs, :] = 0.
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)

        testx = np.random.rand(len(priorvals))
        preds = mock_map.propagate(testx)
        scl = preds.reshape(-1, 1) / expvals.reshape(-1, 1)
        sclmat = scl.reshape(-1, 1) * scl.reshape(1, -1)
        expcov = expcov.toarray() * sclmat
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx,
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_param_proposal_function(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_dict = {'grp_A': 0.3, 'grp_B': 0.02}
        num_unc = len(unc_dict)
        xref = postdist.stack_params_and_uncs(priorvals, unc_dict)
        xarr = np.hstack([xref]*1000000)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0)
        res = propfun(xarr)
        ref_cov = np.linalg.inv(invcov.toarray())
        test_cov = np.cov(res[:-len(unc_dict), :])
        self.assertTrue(np.allclose(test_cov, ref_cov, rtol=1e-4, atol=1e-4))
        ref_mean = xref[:-num_unc]
        test_mean = np.mean(res[:-num_unc], axis=1, keepdims=True)
        self.assertTrue(np.allclose(ref_mean, test_mean, rtol=1e-4))

    def test_param_logpdf_function(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_dict = {'grp_A': 0.3, 'grp_B': 0.02}
        num_unc = len(unc_dict)
        xref = postdist.stack_params_and_uncs(priorvals, unc_dict)
        xarr = np.hstack([xref]*100)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0)
        ref_cov = np.linalg.inv(invcov.toarray())
        res = propfun(xarr)
        logpdf_vec = prop_logpdf(xarr, res)
        mvn = multivariate_normal(mean=priorvals, cov=ref_cov)
        ref_logpdf_vec = mvn.logpdf(res[:-num_unc].T)
        self.assertTrue(np.allclose(logpdf_vec, ref_logpdf_vec))


if __name__ == '__main__':
    unittest.main()
