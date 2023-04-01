import unittest
from gmapy.mcmc_inference import Posterior
import numpy as np
import scipy.sparse as sps


class TestPosteriorClass(unittest.TestCase):

    def create_mock_quantities(self):
        saved_state = np.random.get_state()
        np.random.seed(42)
        priorvals = np.random.rand(10)
        tmp = np.random.rand(10, 10)
        priorcov = tmp.T @ tmp
        priorcov = sps.csr_matrix(priorcov)
        expvals = np.random.rand(20)
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
            self.yref = yref
            self.S = S
            self.xref = xref

        def propagate(self, x):
            return self.yref + self.S @ (x - self.xref)

        def jacobian(self, x):
            return self.S

    def compute_reference_logpdf(
        self, priorvals, priorcov, mapping, expvals, expcov, x
    ):
        if sps.issparse(priorcov):
            priorcov = priorcov.toarray()
        if sps.issparse(expcov):
            expcov = expcov.toarray()
        preds = mapping.propagate(x)
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
        return ret

    def test_correct_computation_of_logpdf(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(priorvals, priorcov, mock_map, expvals, expcov)

        testx = np.random.rand(len(priorvals))
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx
        )
        test_logpdf = postdist.logpdf(testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_correct_computation_of_logpdf_with_ppp(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True
        )
        testx = np.random.rand(len(priorvals))
        preds = mock_map.propagate(testx)
        scl = preds.reshape(-1, 1) / expvals.reshape(-1, 1)
        sclmat = scl.reshape(-1, 1) * scl.reshape(1, -1)
        expcov = expcov.toarray() * sclmat
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx,
        )
        test_logpdf = postdist.logpdf(testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_correct_computation_of_approximate_logpdf(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(priorvals, priorcov, mock_map, expvals, expcov)

        refx = np.random.rand(len(priorvals))
        testx = np.random.rand(len(priorvals))
        test_logpdf = postdist.approximate_logpdf(refx, testx)
        yref = mock_map.propagate(refx)
        S = mock_map.jacobian(refx)
        mock_map_lin = self.MockMapLinear(yref, S, refx)
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map_lin, expvals, expcov, testx
        )
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_correct_computation_of_approximate_logpdf_with_ppp(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True
        )
        refx = np.random.rand(len(priorvals))
        testx = np.random.rand(len(priorvals))
        test_logpdf = postdist.approximate_logpdf(refx, testx)
        yref = mock_map.propagate(refx)
        S = mock_map.jacobian(refx)
        mock_map_lin = self.MockMapLinear(yref, S, refx)
        ypred = mock_map_lin.propagate(testx)
        scl = ypred.reshape(-1, 1) / expvals.reshape(-1, 1)
        sclmat = scl.reshape(-1, 1) * scl.reshape(1, -1)
        expcov = expcov.toarray() * sclmat
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map_lin, expvals, expcov, testx
        )
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))


if __name__ == '__main__':
    unittest.main()
