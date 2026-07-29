import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.tf_uq.covariance_models import (
    GenericCovarianceModel,
    LowRankCovarianceModel
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormalLikelihoodWithCovParams
)
from gmapy.mappings.tf.vectorized_compound_map_tf import (
    VectorizedCompoundMap
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table
)
from gmapy.mappings.priortools import attach_shape_prior


def _make_synthetic_models(m=40, k=7, p=3, seed=13):
    rng = np.random.default_rng(seed)
    base_op = tf.linalg.LinearOperatorDiag(
        tf.constant(rng.uniform(0.5, 1.5, m), dtype=tf.float64)
    )
    smat = tf.constant(rng.normal(scale=0.3, size=(m, k)), dtype=tf.float64)
    ids = tf.constant(rng.integers(0, p, size=k), dtype=tf.int32)

    def diag_fun(u):
        return tf.square(tf.gather(u, ids)) + 1e-4

    def like_cov_fun(u):
        return tf.linalg.LinearOperatorLowRankUpdate(
            base_op, smat, diag_fun(u),
            is_self_adjoint=True, is_positive_definite=True,
            is_diag_update_positive=True
        )

    generic = GenericCovarianceModel(like_cov_fun)
    lowrank = LowRankCovarianceModel(base_op, smat, diag_fun)
    return generic, lowrank, rng


class TestCovarianceModelsSynthetic(unittest.TestCase):

    def test_operator_agreement(self):
        generic, lowrank, rng = _make_synthetic_models()
        u = tf.constant([0.3, 0.5, 0.2], dtype=tf.float64)
        c1 = generic.operator(u).to_dense().numpy()
        c2 = lowrank.operator(u).to_dense().numpy()
        self.assertTrue(np.allclose(c1, c2, rtol=1e-14))

    def test_covpar_blocks_agreement(self):
        generic, lowrank, rng = _make_synthetic_models()
        u = tf.constant([0.3, 0.5, 0.2], dtype=tf.float64)
        z = tf.constant(rng.normal(size=40), dtype=tf.float64)
        kmat = tf.constant(rng.normal(size=(40, 5)), dtype=tf.float64)
        for include_logdet in (True, False):
            cross1, covpar1 = generic.covpar_blocks(
                u, kmat, z, include_logdet=include_logdet
            )
            cross2, covpar2 = lowrank.covpar_blocks(
                u, kmat, z, include_logdet=include_logdet
            )
            self.assertTrue(
                np.allclose(cross1.numpy(), cross2.numpy(),
                            rtol=1e-9, atol=1e-12),
                msg=f'cross block, include_logdet={include_logdet}'
            )
            self.assertTrue(
                np.allclose(covpar1.numpy(), covpar2.numpy(),
                            rtol=1e-9, atol=1e-12),
                msg=f'covpar block, include_logdet={include_logdet}'
            )
            self.assertTrue(np.allclose(
                covpar2.numpy(), covpar2.numpy().T, rtol=1e-12, atol=1e-14
            ))

    def test_batch_chisqr_and_logdet_agreement(self):
        generic, lowrank, rng = _make_synthetic_models()
        u_batch = tf.constant(
            rng.uniform(0.1, 0.5, size=(5, 3)), dtype=tf.float64
        )
        z_batch = tf.constant(rng.normal(size=(5, 40)), dtype=tf.float64)
        c1, l1 = generic.batch_chisqr_and_logdet(u_batch, z_batch)
        c2, l2 = lowrank.batch_chisqr_and_logdet(u_batch, z_batch)
        self.assertTrue(np.allclose(c1.numpy(), c2.numpy(), rtol=1e-9))
        self.assertTrue(np.allclose(l1.numpy(), l2.numpy(), rtol=1e-9))


class TestLikelihoodWithLowRankModel(unittest.TestCase):

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
        rng = np.random.default_rng(59)
        free_idcs = np.sort(rng.choice(num_params, size=50, replace=False))
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

        def diag_fun(u):
            return tf.square(u) + 1e-4

        def like_cov_fun(u):
            return tf.linalg.LinearOperatorLowRankUpdate(
                base_op, smat, diag_fun(u),
                is_self_adjoint=True, is_positive_definite=True,
                is_diag_update_positive=True
            )

        lowrank_model = LowRankCovarianceModel(base_op, smat, diag_fun)
        cls._liks = {}
        for tag, covarg, approx in (
            ('generic_exact', like_cov_fun, False),
            ('lowrank_exact', lowrank_model, False),
            ('generic_approx', like_cov_fun, True),
            ('lowrank_approx', lowrank_model, True),
        ):
            cls._liks[tag] = MultivariateNormalLikelihoodWithCovParams(
                len(free_idcs), num_covpars, restrimap.propagate,
                restrimap.jacobian, like_data, covarg,
                relative=True, approximate_hessian=approx,
                whessfun=restrimap.weighted_hessian
            )
        covpars = np.array([0.1, 0.2, 0.15])
        cls._x = np.concatenate([priorvals[free_idcs], covpars])

    def _compare(self, tag1, tag2):
        lik1 = self._liks[tag1]
        lik2 = self._liks[tag2]
        lp1 = float(lik1.log_prob(self._x))
        lp2 = float(lik2.log_prob(self._x))
        self.assertAlmostEqual(lp1, lp2, places=8)
        h1 = lik1.log_prob_hessian(self._x).numpy()
        h2 = lik2.log_prob_hessian(self._x).numpy()
        scale = np.max(np.abs(h1))
        self.assertTrue(
            np.allclose(h1, h2, rtol=1e-8, atol=1e-10 * scale),
            msg=f'{tag1} vs {tag2}: max abs diff '
                f'{np.max(np.abs(h1 - h2))} at scale {scale}'
        )

    def test_log_prob_batch_matches_single_evaluations(self):
        rng = np.random.default_rng(3)
        xmat = np.tile(self._x, (4, 1))
        xmat += rng.normal(scale=1e-3, size=xmat.shape)
        for tag in ('lowrank_exact', 'generic_exact'):
            lik = self._liks[tag]
            batch = lik.log_prob_batch(
                tf.constant(xmat, dtype=tf.float64)
            ).numpy()
            singles = np.array(
                [float(lik.log_prob(xmat[k])) for k in range(4)]
            )
            self.assertTrue(
                np.allclose(batch, singles, rtol=1e-10),
                msg=f'{tag}: max diff {np.max(np.abs(batch - singles))}'
            )

    def test_lowrank_matches_generic_exact_hessian(self):
        self._compare('generic_exact', 'lowrank_exact')

    def test_lowrank_matches_generic_approximate_hessian(self):
        self._compare('generic_approx', 'lowrank_approx')


if __name__ == '__main__':
    unittest.main()
