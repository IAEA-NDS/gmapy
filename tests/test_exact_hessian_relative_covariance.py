import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormalLikelihoodWithCovParams
)
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestExactHessianRelativeCovariance(unittest.TestCase):

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
        # params feeding the legacy (SACS) datasets stay fixed: their
        # second-order contributions are not covered by the
        # weighted-hessian contraction
        legacy_src = np.unique(np.concatenate([
            np.concatenate([np.asarray(s) for s in src_idcs_list])
            for curmap in vecmap._legacy_maps
            for src_idcs_list in curmap._src_idcs_list
        ]))
        candidates = np.setdiff1d(np.arange(num_params), legacy_src)
        rng = np.random.default_rng(43)
        free_idcs = np.sort(rng.choice(candidates, size=50, replace=False))
        fixed_idcs = np.setdiff1d(np.arange(num_params), free_idcs)
        restrimap = RestrictedMap(
            num_params, vecmap.propagate, vecmap.jacobian,
            fixed_params=priorvals[fixed_idcs],
            fixed_params_idcs=fixed_idcs,
            whessfun=vecmap.weighted_row_hessian
        )
        like_data = exptable['DATA'].to_numpy()
        num_exp = len(like_data)
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

        cls._likelihood = MultivariateNormalLikelihoodWithCovParams(
            len(free_idcs), num_covpars, restrimap.propagate,
            restrimap.jacobian, like_data, like_cov_fun,
            relative=True, approximate_hessian=False,
            whessfun=restrimap.weighted_hessian
        )
        cls._likelihood_approx = MultivariateNormalLikelihoodWithCovParams(
            len(free_idcs), num_covpars, restrimap.propagate,
            restrimap.jacobian, like_data, like_cov_fun,
            relative=True, approximate_hessian=True
        )
        covpars = np.array([0.1, 0.2, 0.15])
        cls._x = np.concatenate([priorvals[free_idcs], covpars])

    def test_exact_hessian_matches_finite_differences(self):
        lik = self._likelihood
        x = self._x
        hess = lik.log_prob_hessian(x).numpy()
        self.assertTrue(np.allclose(hess, hess.T, rtol=1e-12, atol=1e-12))
        eps = 1e-6
        hess_fd = np.zeros_like(hess)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            gp = lik.log_prob_gradient(xp).numpy()
            gm = lik.log_prob_gradient(xm).numpy()
            hess_fd[:, k] = (gp - gm) / (2. * eps)
        scale = np.max(np.abs(hess_fd))
        self.assertTrue(
            np.allclose(hess, hess_fd, rtol=1e-4, atol=1e-6 * scale),
            msg=f'max abs diff {np.max(np.abs(hess - hess_fd))} '
                f'at scale {scale}'
        )

    def test_exact_hessian_differs_from_approximate(self):
        hess = self._likelihood.log_prob_hessian(self._x).numpy()
        hess_approx = self._likelihood_approx.log_prob_hessian(
            self._x
        ).numpy()
        npars = self._likelihood._num_params
        self.assertFalse(
            np.allclose(hess[:npars, :npars], hess_approx[:npars, :npars])
        )


if __name__ == '__main__':
    unittest.main()
