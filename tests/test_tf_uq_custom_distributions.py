import unittest
import pathlib
import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table,
)
from gmapy.mappings.priortools import (
    attach_shape_prior,
    initialize_shape_prior,
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct,
    MultivariateNormalLikelihoodWithCovParams
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.helperfuns.numeric_jacobian import numeric_jacobian
from gmapy.tf_uq.custom_linear_operators import MyLinearOperatorLowRankUpdate


class TestTfUQCustomDistributions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db_path = (pathlib.Path(__file__).parent / 'testdata' /
                'data_and_sacs.json').resolve().as_posix()
        db = read_gma_database(db_path)
        priortable = create_prior_table(db['prior_list'])
        priorcov = create_prior_covmat(db['prior_list'])
        exptable = create_experiment_table(db['datablock_list'])
        expcov = create_experimental_covmat(db['datablock_list'])
        exptable['UNC'] = np.sqrt(expcov.diagonal())
        expvals = exptable.DATA.to_numpy()
        # initialize the normalization errors
        priortable, priorcov = attach_shape_prior(
            (priortable, exptable), covmat=priorcov
        )
        compmap_tf = CompoundMapTF((priortable, exptable), reduce=True)
        initialize_shape_prior((priortable, exptable), compmap_tf)
        priorvals = priortable.PRIOR.to_numpy()
        # save everything for the tests
        cls._compmap_tf = compmap_tf
        cls._priortable = priortable
        cls._priorvals = priorvals
        cls._priorcov = priorcov
        cls._exptable = exptable
        cls._expvals = expvals
        cls._expcov = expcov
        cls._datablock_list = db['datablock_list']
        cls._prior_list = db['prior_list']

    # auxiliary functions
    def generate_fun_i_j(self, fun, full_x, test_i, test_j):
        def loc_fun(x):
            cur_full_x = full_x.copy()
            cur_full_x[test_i] = x[0]
            cur_full_x[test_j] = x[1]
            res = fun(cur_full_x).numpy().reshape((-1,))
            return res
        return loc_fun

    def generate_jacfun_i_j(self, fun, full_x, test_i, test_j):
        def jac_fun(x):
            return numeric_jacobian(myfun, x, o=4, h1=1e-2).reshape((-1,))
        myfun = self.generate_fun_i_j(fun, full_x, test_i, test_j)
        return jac_fun

    def hessfun(self, fun, x, test_i, test_j):
        curjacfun = self.generate_jacfun_i_j(fun, x, test_i, test_j)
        red_x = np.array([x[test_i], x[test_j]])
        res = numeric_jacobian(curjacfun, red_x, o=4, h1=1e-2)
        return res

    # tests
    def test_likelihood_with_equivalent_linear_operators(self):
        compmap_tf = self._compmap_tf
        priorvals = self._priorvals
        expvals = self._expvals
        expcov = self._expcov
        propfun = compmap_tf.propagate
        jacfun = compmap_tf.jacobian
        expchol = tf.linalg.cholesky(expcov.toarray())
        expchol_linop = tf.linalg.LinearOperatorLowerTriangular(expchol)
        expcov_blocks, _ = create_datablock_covmat_list(self._datablock_list)
        expchol_blocks = [
            tf.linalg.cholesky(b.toarray()) for b in expcov_blocks
        ]
        expchol_linops = [
            tf.linalg.LinearOperatorLowerTriangular(b) for b in expchol_blocks
        ]
        better_expchol_linop = tf.linalg.LinearOperatorBlockDiag(expchol_linops)
        mvn_like = MultivariateNormalLikelihood(
            len(priorvals), propfun, jacfun, expvals, expchol_linop
        )
        better_mvn_like = MultivariateNormalLikelihood(
            len(priorvals), propfun, jacfun, expvals, better_expchol_linop
        )
        res1 = mvn_like.log_prob(priorvals)
        res2 = better_mvn_like.log_prob(priorvals)
        self.assertTrue(np.allclose(res1, res2))

    def test_likelihood_with_restricted_mapping(self):
        compmap_tf = self._compmap_tf
        priorvals = self._priorvals
        priorcov = self._priorcov
        expcov = self._expcov
        propfun = compmap_tf.propagate
        jacfun = compmap_tf.jacobian
        expvals = self._expvals
        expchol = tf.linalg.cholesky(expcov.toarray())
        expchol_linop = tf.linalg.LinearOperatorLowerTriangular(expchol)
        is_adj = priorcov.diagonal() != 0.
        mvn_like = MultivariateNormalLikelihood(
            len(priorvals), propfun, jacfun, expvals, expchol_linop
        )
        restricted_map = RestrictedMap(
            len(priorvals), propfun, jacfun,
            priorvals[~is_adj], np.where(~is_adj)[0]
        )
        mvn_like_fix = MultivariateNormalLikelihood(
            np.sum(is_adj), restricted_map.propagate, restricted_map.jacobian,
            expvals, expchol_linop
        )
        res1 = mvn_like.log_prob(priorvals)
        res2 = mvn_like_fix.log_prob(priorvals[is_adj])
        self.assertTrue(np.allclose(res1, res2))

    def test_hessian_of_posterior(self):
        compmap_tf = self._compmap_tf
        priorvals = self._priorvals
        priorcov = self._priorcov.copy()
        priorcov[0, 0] = 20.
        priorcov[2, 2] = 10.
        priorcov[0, 2] = 1.
        priorcov[2, 0] = 1.
        expvals = self._expvals
        expcov = self._expcov
        expchol = tf.linalg.cholesky(expcov.toarray())
        expchol_linop = tf.linalg.LinearOperatorLowerTriangular(expchol)
        is_adj = priorcov.diagonal() != 0.
        is_adj_and_finite = is_adj & np.isfinite(priorcov.diagonal())
        priorchol_linop = tf.linalg.LinearOperatorDiag(
            np.sqrt(priorcov.diagonal()[is_adj_and_finite])
        )
        restricted_map = RestrictedMap(
            len(priorvals), compmap_tf.propagate, compmap_tf.jacobian,
            priorvals[~is_adj], np.where(~is_adj)[0]
        )
        propfun = tf.function(restricted_map.propagate)
        jacfun = tf.function(restricted_map.jacobian)
        mvn_prior = MultivariateNormal(
            priorvals[is_adj_and_finite], priorchol_linop
        )
        mvn_prior_const = DistributionForParameterSubset(
            mvn_prior, np.sum(is_adj), np.where(is_adj_and_finite)
        )
        mvn_like_fix = MultivariateNormalLikelihood(
            np.sum(is_adj), propfun, jacfun,
            expvals, expchol_linop
        )
        post = UnnormalizedDistributionProduct([mvn_prior_const, mvn_like_fix])
        el_idx1 = 0
        el_idx2 = 2
        el_idcs = np.array([el_idx1, el_idx2])
        numeric_hess_mat = self.hessfun(
            post.log_prob, priorvals[is_adj], el_idx1, el_idx2
        )
        analytic_hess = post.log_prob_hessian(priorvals[is_adj])
        res1 = numeric_hess_mat
        res2 = analytic_hess.numpy()[np.ix_(el_idcs, el_idcs)]
        self.assertTrue(np.allclose(res1, res2))

    # auxiliary functions
    def create_Smat(self, exptable):
        dataset_ids = exptable.NODE.unique()
        row_idcs = []
        col_idcs = []
        for idx, ds_id in enumerate(dataset_ids):
            curdt = exptable[exptable.NODE == ds_id]
            cur_row_idcs = curdt.index.to_numpy()
            cur_col_idcs = np.full_like(cur_row_idcs, idx)
            row_idcs.extend(cur_row_idcs)
            col_idcs.extend(cur_col_idcs)
        ones_vec = np.ones_like(row_idcs)
        Smat = coo_matrix((ones_vec, (row_idcs, col_idcs)), dtype=np.float64)
        return Smat.toarray()

    def create_like_cov_fun(self, expcov_linop, Smat):
        def like_cov_fun(u):
            covop = MyLinearOperatorLowRankUpdate(
                expcov_linop, Smat, u
            )
            return covop
        return like_cov_fun

    def test_hessian_of_likelihood_with_covpars(self):
        compmap = self._compmap_tf
        propfun = tf.function(compmap.propagate)
        jacfun = tf.function(compmap.jacobian)
        expvals = self._expvals
        priorvals = self._priorvals
        expcov_list, _ = create_datablock_covmat_list(self._datablock_list)
        expchol_list = [
            tf.constant(np.linalg.cholesky(p.toarray()), dtype=tf.float64)
            for p in expcov_list
        ]
        expchol_linop_list = [
            tf.linalg.LinearOperatorLowerTriangular(p) for p in expchol_list
        ]
        expcov_linop_list = [
            tf.linalg.LinearOperatorComposition([p, p.adjoint()])
            for p in expchol_linop_list
        ]
        expcov_linop = tf.linalg.LinearOperatorBlockDiag(expcov_linop_list)
        Smat = self.create_Smat(self._exptable)
        like_cov_fun = self.create_like_cov_fun(expcov_linop, Smat)
        num_params = len(priorvals)
        num_covpars = Smat.shape[1]
        testdist = MultivariateNormalLikelihoodWithCovParams(
            num_params, num_covpars, propfun, jacfun, expvals, like_cov_fun,
            approximate_hessian=False
        )
        testpars = testdist.combine_pars(priorvals, np.full(num_covpars, 0.3))
        xpars, covpars = testdist.split_pars(testpars)
        # test the parameter part of the Hessian
        logprobfun = tf.function(testdist.log_prob)
        print('test parameter block of the Hessian matrix...')
        print('evaluate Hessian analytically...')
        ana_hess = testdist.log_prob_hessian(testpars)
        print('evaluate Hessian numerically...')
        num_hess_pars = self.hessfun(logprobfun, testpars.numpy(), 0, 7)
        red_ana_hess = ana_hess.numpy()[np.ix_([0,7], [0,7])]
        self.assertTrue(np.allclose(num_hess_pars, red_ana_hess))
        # test some elements in the covpars part
        print('test covparams block of the Hessian matrix...')
        num_hess_covpars = self.hessfun(logprobfun, testpars.numpy(), 1329, 1330)
        red_ana_hess = ana_hess.numpy()[np.ix_([1329,1330], [1329,1330])]
        self.assertTrue(np.allclose(num_hess_covpars, red_ana_hess))
        # test some off-diagonal elements (between covpars part and param part)
        print('test off-diagonal block of the Hessian matrix...')
        num_hess_offdiag = self.hessfun(logprobfun, testpars.numpy(), 1685, 1070)
        red_ana_hess = ana_hess.numpy()[np.ix_([1685,1070], [1685,1070])]
        self.assertTrue(np.allclose(num_hess_offdiag, red_ana_hess, rtol=2e-5))
        # test if log_prob is correctly evaluated by comparing to reference implementation
        propvals = propfun(testpars)
        covop = like_cov_fun(covpars)
        covop_chol = tf.linalg.cholesky(covop.to_dense())
        covchol_linop = tf.linalg.LinearOperatorLowerTriangular(covop_chol)
        tfd = tfp.distributions
        pdf = tfd.MultivariateNormalLinearOperator(
            loc=propvals, scale=covchol_linop
        )
        refres = pdf.log_prob(expvals)
        res = testdist.log_prob(testpars)
        self.assertTrue(np.allclose(res.numpy(), refres.numpy()))


if __name__ == '__main__':
    unittest.main()
