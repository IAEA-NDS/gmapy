import unittest
import numpy as np
import tensorflow as tf
from gmapy.tf_uq.custom_distributions import (
    DistributionWrapper,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct,
    MultivariateNormal,
    MultivariateNormalLikelihoodWithCovParams,
    LogBarrier
)
from gmapy.tf_uq.inference import determine_MAP_estimate_precond_lbfgs


class _AsymmetricHessianStub:
    # pins the scatter orientation in DistributionForParameterSubset;
    # a real Hessian is symmetric and could not detect a transposed
    # scatter
    def log_prob(self, x):
        return tf.reduce_sum(x)

    def log_prob_hessian(self, x):
        return tf.constant([[1., 2.], [3., 4.]], dtype=tf.float64)


class TestDistributionUtilityClasses(unittest.TestCase):

    def test_parameter_subset_accepts_default_idcs(self):
        dist = DistributionForParameterSubset(None, 3)
        x = tf.constant([1., 2., 3.], dtype=tf.float64)
        self.assertEqual(float(dist.log_prob(x)), 0.)
        hess = dist.log_prob_hessian(x).numpy()
        self.assertTrue(np.all(hess == 0.))

    def test_parameter_subset_hessian_scatter_orientation(self):
        dist = DistributionForParameterSubset(
            _AsymmetricHessianStub(), 4, idcs=[1, 3]
        )
        x = tf.zeros(4, dtype=tf.float64)
        hess = dist.log_prob_hessian(x).numpy()
        expected = np.zeros((4, 4))
        expected[np.ix_([1, 3], [1, 3])] = [[1., 2.], [3., 4.]]
        self.assertTrue(np.array_equal(hess, expected))

    def test_wrapper_hessian_accepts_numpy_input(self):
        dist = DistributionWrapper(lambda x: -0.5 * tf.reduce_sum(x**4))
        hess = dist.log_prob_hessian(np.array([1., 2.])).numpy()
        expected = np.diag([-6. * 1.**2, -6. * 2.**2])
        self.assertTrue(np.allclose(hess, expected))

    def test_empty_distribution_product_raises(self):
        with self.assertRaises(ValueError):
            UnnormalizedDistributionProduct([])

    def test_single_no_ppp_index(self):
        def propfun(x):
            return 2. * x

        def jacfun(x):
            return tf.sparse.from_dense(
                2. * tf.eye(3, dtype=tf.float64)
            )

        def like_cov_fun(u):
            return tf.linalg.LinearOperatorDiag(
                tf.ones(3, dtype=tf.float64)
            )

        like_data = np.array([1., 5., 3.])
        dist = MultivariateNormalLikelihoodWithCovParams(
            3, 0, propfun, jacfun, like_data, like_cov_fun,
            relative=True, approximate_hessian=True, no_ppp_idcs=[1]
        )
        pars = tf.constant([1., 1., 1.], dtype=tf.float64)
        covop = dist._like_cov_fun(pars, tf.zeros(0, dtype=tf.float64))
        # scaling vector is the model prediction (2, 2, 2) except for
        # the no-ppp row, which is scaled by the data value 5
        diag = np.diag(covop.to_dense().numpy())
        self.assertTrue(np.allclose(diag, [4., 25., 4.]))


class TestLogBarrier(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(19)
        self._qmat = rng.uniform(0.1, 1., size=(5, 4))
        self._barrier = LogBarrier(self._qmat, strength=1e-2)
        self._x = rng.uniform(0.5, 1.5, size=4)

    def test_value_and_feasibility(self):
        barrier = self._barrier
        expected = 1e-2 * np.sum(np.log(self._qmat @ self._x))
        self.assertAlmostEqual(
            float(barrier.log_prob(self._x)), expected, places=12
        )
        self.assertTrue(barrier.is_feasible(self._x))
        x_bad = self._x.copy()
        x_bad[0] = -5.
        self.assertFalse(barrier.is_feasible(x_bad))
        self.assertFalse(np.isfinite(float(barrier.log_prob(x_bad))))

    def test_gradient_and_hessian_vs_finite_differences(self):
        barrier = self._barrier
        x = self._x
        grad = barrier.log_prob_gradient(x).numpy()
        hess = barrier.log_prob_hessian(x).numpy()
        eps = 1e-7
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            gfd = (float(barrier.log_prob(xp))
                   - float(barrier.log_prob(xm))) / (2. * eps)
            self.assertAlmostEqual(grad[k], gfd, places=6)
            hfd = (barrier.log_prob_gradient(xp).numpy()
                   - barrier.log_prob_gradient(xm).numpy()) / (2. * eps)
            self.assertTrue(np.allclose(hess[:, k], hfd, rtol=1e-6))

    def test_batch_matches_single_evaluations(self):
        rng = np.random.default_rng(23)
        xmat = rng.uniform(0.5, 1.5, size=(6, 4))
        batch = self._barrier.log_prob_batch(
            tf.constant(xmat, dtype=tf.float64)
        ).numpy()
        singles = np.array(
            [float(self._barrier.log_prob(xmat[k])) for k in range(6)]
        )
        self.assertTrue(np.allclose(batch, singles, rtol=1e-12))

    def test_constrained_optimization_stays_feasible(self):
        n = 6
        loc = np.array([2., 1., 0.5, -1., -3., 0.2])
        prior_scale = tf.linalg.LinearOperatorDiag(
            tf.constant(np.full(n, 1.), dtype=tf.float64)
        )
        mvn = MultivariateNormal(loc, prior_scale)
        barrier = LogBarrier(np.eye(n), strength=1e-3)
        post = UnnormalizedDistributionProduct([mvn, barrier])
        x0 = tf.constant(np.full(n, 1.), dtype=tf.float64)
        res = determine_MAP_estimate_precond_lbfgs(
            x0, post.neg_log_prob_and_gradient, post.neg_log_prob_hessian,
            nugget=1e-10, ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        xopt = np.array(res.position)
        self.assertTrue(np.all(xopt > 0.))
        # unconstrained components sit at their prior locations, the
        # constrained ones just inside the boundary
        self.assertTrue(np.allclose(xopt[:3], loc[:3], atol=1e-2))
        self.assertTrue(np.all(xopt[3:5] < 1e-2))


if __name__ == '__main__':
    unittest.main()
