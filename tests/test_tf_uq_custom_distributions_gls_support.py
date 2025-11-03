import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from gmapy.tf_uq.inference import (
    iterative_gls_estimate,
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormalLikelihood,
    MultivariateNormalLikelihoodWithCovParams,
    ChiSquarePseudoDist,
)


class TestTfUQCustomDistributions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._priordt = pd.DataFrame({
            'NODE': ['xsid_1']*2,
            'REAC': ['MT:1-R1:1']*2,
            'ENERGY': [0.0, 10.0]
        })
        cls._expdt = pd.DataFrame({
            'NODE': ['exp_1']*2,
            'REAC': ['MT:1-R1:1']*2,
            'ENERGY': [5.0, 6.0] ,
        })
        cls._compmap = CompoundMapTF([cls._priordt, cls._expdt], reduce=True)
        cls._propfun = cls._compmap.propagate
        cls._jacfun = cls._compmap.jacobian
        cls._like_data = tf.constant([3.0, 4.0], dtype=tf.float64)
        cls._like_scale = tf.linalg.LinearOperatorDiag(
            tf.constant([1.0, 2.0], dtype=tf.float64), is_positive_definite=True
        )
        cls._x = tf.constant([10.0, 100.0], dtype=tf.float64)

    def test_multivariate_normal_likelihood_gls_support(self):
        like = MultivariateNormalLikelihood(
            len(self._priordt), self._propfun, self._jacfun,
            self._like_data, self._like_scale
        )
        tfr = tf.reduce_all
        self.assertTrue(
            tfr(self._propfun(self._x) == like.get_model_prediction(self._x))
        )
        self.assertTrue(
            tfr(tf.sparse.to_dense(self._jacfun(self._x)) == like.get_model_jacobian(self._x))
        )
        covmat = self._like_scale.matmul(self._like_scale.to_dense(), adjoint=True)
        self.assertTrue(
            tfr(covmat == like.get_covariance_linop(self._x).to_dense())

        )

    def test_iterative_gls_with_multivariate_normal_likelihood(self):
        like = MultivariateNormalLikelihood(
            len(self._priordt), self._propfun, self._jacfun,
            self._like_data, self._like_scale
        )
        startvals = self._x + 1.0
        optres = iterative_gls_estimate(
            startvals, like.get_model_prediction, like.get_model_jacobian, like.get_data_vector(), like.get_covariance_linop, rel_damp_unc=1e8, ret_optres=True
        )
        tfr = tf.reduce_all
        self.assertTrue(tfr(tf.abs(optres.position - tf.constant([-2.0, 8.0], dtype=tf.float64)) < 1e-10))

    def test_iterative_gls_with_relative_multivariate_normal_likelihood(self):
        like = MultivariateNormalLikelihood(
            len(self._priordt), self._propfun, self._jacfun,
            self._like_data, self._like_scale, relative=True, approximate_hessian=True
        )
        startvals = self._x + 4.0
        optres = iterative_gls_estimate(
            startvals, like.get_model_prediction, like.get_model_jacobian, like.get_data_vector(), like.get_covariance_linop, rel_damp_unc=1e8, ret_optres=True
        )
        tfr = tf.reduce_all
        self.assertTrue(tfr(tf.abs(optres.position - tf.constant([-2.0, 8.0], dtype=tf.float64)) < 1e-10))

    def test_relative_multivariate_normal_likelihood_gls_support(self):
        like = MultivariateNormalLikelihood(
            len(self._priordt), self._propfun, self._jacfun,
            self._like_data, self._like_scale, relative=True, approximate_hessian=True
        )
        tfr = tf.reduce_all
        self.assertTrue(
            tfr(self._propfun(self._x) == like.get_model_prediction(self._x))
        )
        self.assertTrue(
            tfr(tf.sparse.to_dense(self._jacfun(self._x)) == like.get_model_jacobian(self._x))
        )
        covmat = self._like_scale.matmul(self._like_scale.to_dense(), adjoint=True).numpy()
        propvals = self._propfun(self._x).numpy()
        covmat *= propvals.reshape(1,-1) * propvals.reshape(-1, 1)
        self.assertTrue(
            tfr(covmat == like.get_covariance_linop(self._x).to_dense())
        )

    def test_chisquare_likelihood_gls_support(self):
        relative = (True, False)
        for rel in relative:
            like = MultivariateNormalLikelihood(
                len(self._priordt), self._propfun, self._jacfun,
                self._like_data, self._like_scale, relative=rel, approximate_hessian=True
            )
            like2 = ChiSquarePseudoDist(
                len(self._priordt), self._propfun, self._jacfun,
                self._like_data, self._like_scale, relative=rel, approximate_hessian=True
            )
            tfr = tf.reduce_all
            x = self._x
            self.assertTrue(
                tfr(like.get_model_prediction(x) == like2.get_model_prediction(x))
            )
            self.assertTrue(
                tfr(like.get_model_jacobian(x) == like2.get_model_jacobian(x))
            )
            self.assertTrue(
                tfr(
                    like.get_covariance_linop(x).to_dense()
                    == like2.get_covariance_linop(x).to_dense()
                )
            )

    def test_multivariate_normal_likelihood_with_cov_params_gls_support(self):
        def generate_like_cov_op_fun(propfun, like_scale):
            def like_cov_op_fun(x):
                diag_op = tf.linalg.LinearOperatorDiag
                scale_op = diag_op(propfun(x), is_positive_definite=True)
                return tf.linalg.LinearOperatorComposition(
                    [scale_op, like_scale.adjoint(), like_scale, scale_op],
                    is_positive_definite=True
                )
            return like_cov_op_fun
        x = self._x
        x2 = tf.concat([self._x]*2, axis=0)
        like_cov_fun = generate_like_cov_op_fun(self._propfun, self._like_scale)
        like = MultivariateNormalLikelihood(
            len(self._priordt), self._propfun, self._jacfun,
            self._like_data, self._like_scale, relative=True, approximate_hessian=True
        )
        like2 = MultivariateNormalLikelihoodWithCovParams(
            len(self._priordt), len(self._priordt), self._propfun, self._jacfun,
            self._like_data, like_cov_fun, relative=False
        )
        tfr = tf.reduce_all
        self.assertTrue(
            tfr(like.get_model_prediction(x) == like2.get_model_prediction(x2))
        )
        self.assertTrue(
            tfr(like.get_model_jacobian(x) == like2.get_model_jacobian(x2))
        )
        self.assertTrue(
            tfr(
                like.get_covariance_linop(x).to_dense()
                == like2.get_covariance_linop(x2).to_dense()
            )
        )
