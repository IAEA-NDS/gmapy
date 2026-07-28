import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.tf_uq.custom_distributions import MultivariateNormalLikelihood
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


class TestExactHessianFastModelPart(unittest.TestCase):

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
        # keep parameters feeding the legacy (SACS) datasets fixed:
        # their second-order contributions are not covered by
        # weighted_row_hessian, so this makes the reference loop and
        # the contraction path compute the same quantity
        legacy_src = np.unique(np.concatenate([
            np.concatenate([np.asarray(s) for s in src_idcs_list])
            for curmap in vecmap._legacy_maps
            for src_idcs_list in curmap._src_idcs_list
        ]))
        candidates = np.setdiff1d(np.arange(num_params), legacy_src)
        rng = np.random.default_rng(41)
        free_idcs = np.sort(rng.choice(candidates, size=50, replace=False))
        fixed_idcs = np.setdiff1d(np.arange(num_params), free_idcs)
        restrimap = RestrictedMap(
            num_params, vecmap.propagate, vecmap.jacobian,
            fixed_params=priorvals[fixed_idcs],
            fixed_params_idcs=fixed_idcs,
            whessfun=vecmap.weighted_row_hessian
        )
        like_data = exptable['DATA'].to_numpy()
        like_scale = tf.linalg.LinearOperatorDiag(
            0.1 * np.abs(like_data) + 0.1
        )
        args = (len(free_idcs), restrimap.propagate, restrimap.jacobian,
                like_data, like_scale)
        cls._lik_fast = MultivariateNormalLikelihood(
            *args, approximate_hessian=False,
            whessfun=restrimap.weighted_hessian
        )
        cls._lik_loop = MultivariateNormalLikelihood(
            *args, approximate_hessian=False
        )
        cls._x = priorvals[free_idcs]

    def test_fast_model_part_matches_parameter_loop(self):
        hess_fast = self._lik_fast.log_prob_hessian(self._x).numpy()
        hess_loop = self._lik_loop.log_prob_hessian(self._x).numpy()
        scale = np.max(np.abs(hess_loop))
        self.assertTrue(
            np.allclose(hess_fast, hess_loop, rtol=1e-8, atol=1e-10 * scale),
            msg=f'max abs diff {np.max(np.abs(hess_fast - hess_loop))} '
                f'at scale {scale}'
        )
        # ensure the model part is actually exercised
        hess_gls = self._lik_fast._log_prob_hessian_gls_part(self._x).numpy()
        self.assertFalse(np.allclose(hess_fast, hess_gls))

    def test_hessian_matches_finite_differences(self):
        x = self._x
        hess = self._lik_fast.log_prob_hessian(x).numpy()
        eps = 1e-6
        hess_fd = np.zeros_like(hess)
        for k in range(len(x)):
            xp = x.copy(); xp[k] += eps
            xm = x.copy(); xm[k] -= eps
            gp = self._lik_fast.log_prob_gradient(xp).numpy()
            gm = self._lik_fast.log_prob_gradient(xm).numpy()
            hess_fd[:, k] = (gp - gm) / (2. * eps)
        scale = np.max(np.abs(hess_fd))
        self.assertTrue(
            np.allclose(hess, hess_fd, rtol=1e-4, atol=1e-6 * scale),
            msg=f'max abs diff {np.max(np.abs(hess - hess_fd))} '
                f'at scale {scale}'
        )


if __name__ == '__main__':
    unittest.main()
