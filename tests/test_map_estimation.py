import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.tf_uq.inference import (
    determine_MAP_estimate,
    determine_MAP_estimate_newton,
    determine_MAP_estimate_trust_region,
    determine_MAP_estimate_lbfgs,
    determine_MAP_estimate_precond_lbfgs
)
from gmapy.tf_uq.covariance_models import LowRankCovarianceModel
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihoodWithCovParams,
    UnnormalizedDistributionProduct
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


class TestNewtonMapEstimateQuadratic(unittest.TestCase):

    def test_quadratic_problem_converges_immediately(self):
        rng = np.random.default_rng(7)
        n = 20
        amat = rng.normal(size=(n, n))
        prec = tf.constant(amat @ amat.T + n * np.eye(n), dtype=tf.float64)
        loc = tf.constant(rng.normal(size=n), dtype=tf.float64)

        def nlpg(x):
            d = tf.reshape(x - loc, (-1, 1))
            g = tf.matmul(prec, d)
            return 0.5 * tf.reduce_sum(d * g), tf.reshape(g, (-1,))

        def nlph(x):
            return prec

        x0 = tf.constant(rng.normal(size=n), dtype=tf.float64)
        res = determine_MAP_estimate_newton(
            x0, nlpg, nlph, nugget=1e-10, ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        self.assertLessEqual(int(res.num_iterations), 2)
        self.assertTrue(np.allclose(
            np.array(res.position), np.array(loc), atol=1e-10
        ))
        res = determine_MAP_estimate_trust_region(
            x0, nlpg, nlph, ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        self.assertLessEqual(int(res.num_iterations), 10)
        self.assertTrue(np.allclose(
            np.array(res.position), np.array(loc), atol=1e-8
        ))
        res = determine_MAP_estimate_lbfgs(
            x0, nlpg, nlph, nugget=1e-10, ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        self.assertTrue(np.allclose(
            np.array(res.position), np.array(loc), atol=1e-6
        ))
        res = determine_MAP_estimate_precond_lbfgs(
            x0, nlpg, nlph, nugget=1e-10, ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        self.assertLessEqual(int(res.num_iterations), 5)
        self.assertTrue(np.allclose(
            np.array(res.position), np.array(loc), atol=1e-6
        ))


class TestNewtonVsBfgsOnDatabase(unittest.TestCase):

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
        rng = np.random.default_rng(61)
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

        covmodel = LowRankCovarianceModel(base_op, smat, diag_fun)
        likelihood = MultivariateNormalLikelihoodWithCovParams(
            len(free_idcs), num_covpars, restrimap.propagate,
            restrimap.jacobian, like_data, covmodel,
            relative=True, approximate_hessian=False,
            whessfun=restrimap.weighted_hessian
        )
        cls._x0 = np.concatenate(
            [priorvals[free_idcs], np.full(num_covpars, 0.3)]
        )
        # a prior regularizes the multimodality of the bare
        # likelihood with relative covariance (as in production)
        prior_scale = tf.linalg.LinearOperatorDiag(
            tf.constant(0.1 * np.abs(cls._x0) + 0.05, dtype=tf.float64)
        )
        prior = MultivariateNormal(cls._x0, prior_scale)
        cls._likelihood = likelihood
        cls._post = UnnormalizedDistributionProduct([prior, likelihood])

    def test_newton_and_bfgs_find_same_optimum(self):
        post = self._post
        nlpg = tf.function(post.neg_log_prob_and_gradient)
        res_newton = determine_MAP_estimate_newton(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        res_bfgs = determine_MAP_estimate(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        self.assertTrue(bool(res_newton.converged))
        self.assertTrue(bool(res_bfgs.converged))
        fn = float(res_newton.objective_value)
        fb = float(res_bfgs.objective_value.numpy())
        self.assertAlmostEqual(fn, fb, delta=1e-8 * abs(fb))
        # along weakly constrained posterior directions the position
        # agreement is limited by the achievable gradient floor
        pn = np.array(res_newton.position)
        pb = np.array(res_bfgs.position)
        self.assertTrue(
            np.allclose(pn, pb, rtol=1e-4, atol=2e-3),
            msg=f'max abs diff {np.max(np.abs(pn - pb))}'
        )

    def test_preconditioned_lbfgs_finds_same_optimum(self):
        post = self._post
        nlpg = tf.function(post.neg_log_prob_and_gradient)
        res_lbfgs = determine_MAP_estimate_lbfgs(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        res_bfgs = determine_MAP_estimate(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        self.assertTrue(bool(res_lbfgs.converged))
        fl = float(res_lbfgs.objective_value)
        fb = float(res_bfgs.objective_value.numpy())
        self.assertAlmostEqual(fl, fb, delta=1e-7 * abs(fb))
        pl = np.array(res_lbfgs.position)
        pb = np.array(res_bfgs.position)
        self.assertTrue(
            np.allclose(pl, pb, rtol=1e-4, atol=2e-3),
            msg=f'max abs diff {np.max(np.abs(pl - pb))}'
        )

    def test_persistent_precond_lbfgs_finds_same_optimum(self):
        post = self._post
        nlpg = tf.function(post.neg_log_prob_and_gradient)
        res_pl = determine_MAP_estimate_precond_lbfgs(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        res_bfgs = determine_MAP_estimate(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        self.assertTrue(bool(res_pl.converged))
        fl = float(res_pl.objective_value)
        fb = float(res_bfgs.objective_value.numpy())
        self.assertAlmostEqual(fl, fb, delta=1e-7 * abs(fb))
        pl = np.array(res_pl.position)
        pb = np.array(res_bfgs.position)
        self.assertTrue(
            np.allclose(pl, pb, rtol=1e-4, atol=2e-3),
            msg=f'max abs diff {np.max(np.abs(pl - pb))}'
        )

    def test_precond_lbfgs_bold_clipping_escapes_saddle(self):
        # double-well objective with indefinite Hessian at the
        # starting point: the bold (positive) clipping must still
        # converge to one of the minima
        def nlpg(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                f = (0.25 * x[0]**4 - 0.5 * x[0]**2
                     + 0.5 * tf.reduce_sum(x[1:]**2))
            return f, tape.gradient(f, x)

        def nlph(x):
            hess = tf.linalg.diag(tf.concat(
                [[3. * x[0]**2 - 1.], tf.ones(4, dtype=tf.float64)],
                axis=0
            ))
            return hess

        x0 = tf.constant([0.01, 1., -1., 2., 0.5], dtype=tf.float64)
        res = determine_MAP_estimate_precond_lbfgs(
            x0, nlpg, nlph, nugget=1e-8, saddle_free=False,
            ret_optres=True
        )
        self.assertTrue(bool(res.converged))
        self.assertAlmostEqual(abs(float(res.position[0])), 1., places=6)
        self.assertAlmostEqual(float(res.objective_value), -0.25, places=8)

    def test_precond_lbfgs_with_batched_line_search(self):
        post = self._post
        nlpg = tf.function(post.neg_log_prob_and_gradient)
        batch_nlp = tf.function(post.neg_log_prob_batch)
        res_batch = determine_MAP_estimate_precond_lbfgs(
            self._x0, nlpg, post.neg_log_prob_hessian,
            batch_neg_log_prob=batch_nlp, ret_optres=True
        )
        res_seq = determine_MAP_estimate_precond_lbfgs(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        self.assertTrue(bool(res_batch.converged))
        fb = float(res_batch.objective_value)
        fs = float(res_seq.objective_value)
        self.assertAlmostEqual(fb, fs, delta=1e-7 * abs(fs))
        pb = np.array(res_batch.position)
        ps = np.array(res_seq.position)
        self.assertTrue(
            np.allclose(pb, ps, rtol=1e-4, atol=2e-3),
            msg=f'max abs diff {np.max(np.abs(pb - ps))}'
        )

    def test_trust_region_finds_same_optimum(self):
        post = self._post
        likelihood = self._likelihood
        nlpg = tf.function(post.neg_log_prob_and_gradient)

        def gn_hessian(x):
            likelihood._approximate_hessian = True
            try:
                return post.neg_log_prob_hessian(x)
            finally:
                likelihood._approximate_hessian = False

        res_tr = determine_MAP_estimate_trust_region(
            self._x0, nlpg, post.neg_log_prob_hessian,
            neg_log_prob_gn_hessian=gn_hessian, ret_optres=True
        )
        res_bfgs = determine_MAP_estimate(
            self._x0, nlpg, post.neg_log_prob_hessian, ret_optres=True
        )
        self.assertTrue(bool(res_tr.converged))
        ft = float(res_tr.objective_value)
        fb = float(res_bfgs.objective_value.numpy())
        self.assertAlmostEqual(ft, fb, delta=1e-8 * abs(fb))
        pt = np.array(res_tr.position)
        pb = np.array(res_bfgs.position)
        self.assertTrue(
            np.allclose(pt, pb, rtol=1e-4, atol=2e-3),
            msg=f'max abs diff {np.max(np.abs(pt - pb))}'
        )


if __name__ == '__main__':
    unittest.main()
