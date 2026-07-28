import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormalLikelihood,
    MultivariateNormalLikelihoodWithCovParams,
    ChiSquarePseudoDist,
    ChiSquarePseudoDistWithCovParams
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestExactHessianAllDistributions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data-2017-07-26.gma').resolve().as_posix()
        rawdb = read_gma_database(dbpath)
        priortable = create_prior_table(rawdb['prior_list'])
        exptable = create_experiment_table(rawdb['datablock_list'])
        priortable = attach_shape_prior((priortable, exptable))
        vecmap = VectorizedCompoundMap((priortable, exptable), reduce=True)
        priorvals = priortable['PRIOR'].to_numpy() + 1e-5
        num_params = len(priorvals)
        legacy_src = np.unique(np.concatenate([
            np.concatenate([np.asarray(s) for s in src_idcs_list])
            for curmap in vecmap._legacy_maps
            for src_idcs_list in curmap._src_idcs_list
        ]))
        others = np.setdiff1d(np.arange(num_params), legacy_src)
        rng = np.random.default_rng(47)
        free_idcs = np.sort(np.concatenate([
            rng.choice(legacy_src, size=10, replace=False),
            rng.choice(others, size=40, replace=False)
        ]))
        fixed_idcs = np.setdiff1d(np.arange(num_params), free_idcs)
        cls._restrimap = RestrictedMap(
            num_params, vecmap.propagate, vecmap.jacobian,
            fixed_params=priorvals[fixed_idcs],
            fixed_params_idcs=fixed_idcs,
            whessfun=vecmap.weighted_row_hessian
        )
        cls._like_data = exptable['DATA'].to_numpy()
        cls._x_free = priorvals[free_idcs]
        cls._num_free = len(free_idcs)
        num_exp = len(cls._like_data)
        cls._abs_scale = tf.linalg.LinearOperatorDiag(
            tf.constant(0.1 * np.abs(cls._like_data) + 0.1, dtype=tf.float64)
        )
        cls._rel_scale = tf.linalg.LinearOperatorDiag(
            tf.constant(np.full(num_exp, 0.05), dtype=tf.float64)
        )
        num_covpars = 3
        base_op = tf.linalg.LinearOperatorDiag(
            tf.constant(np.full(num_exp, 0.05**2), dtype=tf.float64)
        )
        smat = tf.constant(
            rng.normal(scale=1e-2, size=(num_exp, num_covpars)),
            dtype=tf.float64
        )

        def like_cov_fun(u):
            return tf.linalg.LinearOperatorLowRankUpdate(
                base_op, smat, tf.square(u) + 1e-4,
                is_self_adjoint=True, is_positive_definite=True,
                is_diag_update_positive=True
            )

        cls._like_cov_fun = staticmethod(like_cov_fun)
        cls._num_covpars = num_covpars
        cls._covpars = np.array([0.1, 0.2, 0.15])

    def _make_likelihood(self, cls_, relative):
        return cls_(
            self._num_free, self._restrimap.propagate,
            self._restrimap.jacobian, self._like_data,
            self._rel_scale if relative else self._abs_scale,
            relative=relative, approximate_hessian=False,
            whessfun=self._restrimap.weighted_hessian
        )

    def _make_covpar_likelihood(self, cls_, relative):
        return cls_(
            self._num_free, self._num_covpars, self._restrimap.propagate,
            self._restrimap.jacobian, self._like_data, self._like_cov_fun,
            relative=relative, approximate_hessian=False,
            whessfun=self._restrimap.weighted_hessian
        )

    def _check_hessian_vs_finite_differences(self, dist, x):
        hess = dist.log_prob_hessian(x).numpy()
        eps = 1e-6
        hess_fd = np.zeros_like(hess)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            gp = dist.log_prob_gradient(xp).numpy()
            gm = dist.log_prob_gradient(xm).numpy()
            hess_fd[:, k] = (gp - gm) / (2. * eps)
        scale = np.max(np.abs(hess_fd))
        self.assertTrue(
            np.allclose(hess, hess_fd, rtol=1e-4, atol=1e-6 * scale),
            msg=f'{type(dist).__name__}: max abs diff '
                f'{np.max(np.abs(hess - hess_fd))} at scale {scale}'
        )

    def test_mvn_likelihood_relative(self):
        dist = self._make_likelihood(MultivariateNormalLikelihood, True)
        self._check_hessian_vs_finite_differences(dist, self._x_free)

    def test_chisquare_dist_nonrelative(self):
        dist = self._make_likelihood(ChiSquarePseudoDist, False)
        self._check_hessian_vs_finite_differences(dist, self._x_free)

    def test_chisquare_dist_relative(self):
        dist = self._make_likelihood(ChiSquarePseudoDist, True)
        self._check_hessian_vs_finite_differences(dist, self._x_free)

    def test_chisquare_covpar_dist_nonrelative(self):
        dist = self._make_covpar_likelihood(
            ChiSquarePseudoDistWithCovParams, False
        )
        x = np.concatenate([self._x_free, self._covpars])
        self._check_hessian_vs_finite_differences(dist, x)

    def test_chisquare_covpar_dist_relative(self):
        dist = self._make_covpar_likelihood(
            ChiSquarePseudoDistWithCovParams, True
        )
        x = np.concatenate([self._x_free, self._covpars])
        self._check_hessian_vs_finite_differences(dist, x)

    def test_relative_exact_without_whessfun_raises(self):
        with self.assertRaises(NotImplementedError):
            MultivariateNormalLikelihood(
                self._num_free, self._restrimap.propagate,
                self._restrimap.jacobian, self._like_data,
                self._rel_scale, relative=True, approximate_hessian=False
            )


if __name__ == '__main__':
    unittest.main()
