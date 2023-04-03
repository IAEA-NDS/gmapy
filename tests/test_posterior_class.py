import unittest
from gmapy.mcmc_inference import Posterior
from gmapy.mappings.helperfuns import numeric_jacobian
import numpy as np
import scipy.sparse as sps


class TestPosteriorClass(unittest.TestCase):

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

    def test_correct_computation_of_grad_logpdf(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            squeeze=False
        )
        testx = np.random.rand(len(priorvals))
        ref_grad = np.squeeze(numeric_jacobian(postdist.logpdf, testx))
        test_grad = np.squeeze(postdist.grad_logpdf(testx))
        self.assertTrue(np.allclose(test_grad, ref_grad))

    def test_correct_computation_of_grad_logpdf_with_ppp(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True, squeeze=False
        )
        expvals = mock_map.propagate(priorvals)
        expvals += np.random.rand(len(expvals)) / 10
        testx = priorvals.copy()
        ref_grad = np.squeeze(numeric_jacobian(postdist.logpdf, testx))
        test_grad = np.squeeze(postdist.grad_logpdf(testx))
        self.assertTrue(np.allclose(test_grad, ref_grad))

    def test_correct_computation_of_grad_logpdf_with_ppp_and_mask(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        src_mask = {
            'idcs': np.array([1, 4, 7]),
            'vals': priorvals[np.array([1, 4, 7])]
        }
        tar_mask = {
            'idcs': np.array([1, 12, 17]),
            'vals': expvals[[1, 12, 17]]
        }
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True, squeeze=False,
            source_mask=src_mask, target_mask=tar_mask
        )
        expvals = mock_map.propagate(priorvals)
        expvals += np.random.rand(len(expvals)) / 10
        testx = priorvals.copy()
        ref_grad = np.squeeze(numeric_jacobian(postdist.logpdf, testx))
        test_grad = np.squeeze(postdist.grad_logpdf(testx))
        self.assertTrue(np.allclose(test_grad, ref_grad))

    def test_correct_computation_jacobian_of_difference(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        src_mask = {
            'idcs': np.array([1, 4, 7]),
            'vals': priorvals[np.array([1, 4, 7])]
        }
        tar_mask = {
            'idcs': np.array([1, 12, 17]),
            'vals': expvals[[1, 12, 17]]
        }
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True, squeeze=False,
            source_mask=src_mask, target_mask=tar_mask
        )

        def compute_difference(x):
            propx = postdist._get_propx(x)
            propx2 = postdist._get_propx2(x)
            d = postdist._get_d2(propx, propx2)
            return d.flatten()

        testx = priorvals.copy()
        ref_grad = np.squeeze(numeric_jacobian(compute_difference, testx))

        S = mock_map.jacobian(testx)
        propx = postdist._get_propx(testx)
        propx2 = postdist._get_propx2(testx)
        test_grad = postdist._exp_pred_diff_jacobian(
            testx, S, propx, propx2
        ).toarray()
        self.assertTrue(np.allclose(test_grad, ref_grad))

    def test_correct_computation_jacobian_of_logdet(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        src_mask = {
            'idcs': np.array([1, 4, 7]),
            'vals': priorvals[np.array([1, 4, 7])]
        }
        tar_mask = {
            'idcs': np.array([1, 12, 17]),
            'vals': expvals[[1, 12, 17]]
        }
        postdist = Posterior(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True, squeeze=False,
            source_mask=src_mask, target_mask=tar_mask
        )

        def compute_logdet(x):
            propx2 = postdist._get_propx2(x)
            mycov = expcov.toarray()
            scl = propx2.flatten() / expvals.flatten()
            mycov *= scl.reshape(-1, 1) * scl.reshape(1, -1)
            ret = np.array([np.linalg.slogdet(mycov)[1]])
            return ret

        testx = priorvals.copy()
        ref_grad = np.squeeze(numeric_jacobian(compute_logdet, testx))

        S = mock_map.jacobian(testx)
        propx2 = postdist._get_propx2(testx)
        test_grad = postdist._prop_logdet_derivative(
            testx, S, propx2
        )
        self.assertTrue(np.allclose(test_grad, ref_grad))

    def test_analytic_chisquare_derivative_with_ppp(self):
        np.random.seed(88)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()

        def my_chisquare(x):
            d = (expvals - x) * expvals / x
            d = d.reshape(-1, 1)
            res = d.T @ np.linalg.inv(expcov.toarray()) @ d
            return res.flatten()

        def my_grad_chisquare(x):
            d = (expvals - x) * expvals / x
            d = d.reshape(-1, 1)
            outer_jac = (np.square(expvals / x)).reshape(-1, 1)
            return -2 * outer_jac * (np.linalg.inv(expcov.toarray()) @ d)

        propvals = expvals + np.random.rand(len(expvals))
        my_grad = my_grad_chisquare(propvals).flatten()
        your_grad = numeric_jacobian(my_chisquare, propvals).flatten()
        self.assertTrue(np.allclose(my_grad, your_grad))

    def test_analytic_logdet_derivative_with_ppp(self):
        np.random.seed(88)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()

        def my_logdet(x):
            scl = x / expvals
            sclmat = scl.reshape(-1, 1) * scl.reshape(1, -1)
            scl_expcov = sclmat * expcov.toarray()
            logdet = np.linalg.slogdet(scl_expcov)[1]
            return np.array([logdet])

        def my_grad_logdet(x):
            return 2 / x

        propvals = expvals + np.random.rand(len(expvals))
        my_grad = my_grad_logdet(propvals).flatten()
        your_grad = numeric_jacobian(my_logdet, propvals).flatten()
        self.assertTrue(np.allclose(my_grad, your_grad))


if __name__ == '__main__':
    unittest.main()
