import unittest
import numpy as np
import tensorflow as tf
from gmapy.tf_uq.custom_distributions import (
    DistributionWrapper,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct,
    MultivariateNormalLikelihoodWithCovParams
)


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


if __name__ == '__main__':
    unittest.main()
