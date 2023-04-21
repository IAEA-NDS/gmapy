import unittest
from gmapy.posterior_usu import PosteriorUSU
from gmapy.mcmc_inference import mh_algo
from gmapy.inference import lm_update
import numpy as np
import scipy.sparse as sps
from scipy.stats import multivariate_normal


class TestPosteriorUSUClass(unittest.TestCase):

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
            self.yref = yref.reshape(-1, 1)
            self.S = sps.csr_matrix(S)
            self.xref = xref.reshape(-1, 1)

        def propagate(self, x):
            x = x.reshape(-1, 1)
            return self.yref + self.S @ (x - self.xref)

        def jacobian(self, x):
            return self.S

    def compute_reference_logpdf(
        self, priorvals, priorcov, mapping, expvals, expcov, x
    ):
        x = x.reshape(-1, 1)
        priorvals = priorvals.reshape(-1, 1)
        expvals = expvals.reshape(-1, 1)
        if sps.issparse(priorcov):
            priorcov = priorcov.toarray()
        if sps.issparse(expcov):
            expcov = expcov.toarray()
        preds = mapping.propagate(x.flatten()).reshape(-1, 1)
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
        return ret.flatten()

    def test_correct_computation_of_logpdf(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_group_A = 4.0
        unc_group_B = 2.5
        priorcov = priorcov.toarray()
        priorcov[:, unc_idcs] = 0.
        priorcov[unc_idcs, :] = 0.
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)
        testx = np.random.rand(len(priorvals))
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))
        # and another time...
        unc_group_A = 23.0
        unc_group_B = 29.0
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)
        testx = np.random.rand(len(priorvals))
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_correct_computation_of_logpdf_with_ppp(self):
        np.random.seed(47)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_group_A = 4.0
        unc_group_B = 2.5
        priorcov = priorcov.toarray()
        priorcov[:, unc_idcs] = 0.
        priorcov[unc_idcs, :] = 0.
        priorcov[[3, 7], [3, 7]] = unc_group_A * unc_group_A
        priorcov[[5, 9], [5, 9]] = unc_group_B * unc_group_B
        priorcov = sps.csr_matrix(priorcov)

        testx = np.random.rand(len(priorvals))
        preds = mock_map.propagate(testx)
        scl = preds.reshape(-1, 1) / expvals.reshape(-1, 1)
        sclmat = scl.reshape(-1, 1) * scl.reshape(1, -1)
        expcov = expcov.toarray() * sclmat
        ref_logpdf = self.compute_reference_logpdf(
            priorvals, priorcov, mock_map, expvals, expcov, testx,
        )
        ext_testx = postdist.stack_params_and_uncs(
            testx, {'grp_A': unc_group_A, 'grp_B': unc_group_B}
        )
        test_logpdf = postdist.logpdf(ext_testx)
        self.assertTrue(np.isclose(test_logpdf, ref_logpdf))

    def test_param_proposal_function(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_dict = {'grp_A': 0.3, 'grp_B': 0.02}
        num_unc = len(unc_dict)
        xref = postdist.stack_params_and_uncs(priorvals, unc_dict)
        xarr = np.hstack([xref]*1000000)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0)
        res = propfun(xarr)
        ref_cov = np.linalg.inv(invcov.toarray())
        test_cov = np.cov(res[:-len(unc_dict), :])
        self.assertTrue(np.allclose(test_cov, ref_cov, rtol=1e-4, atol=1e-4))
        ref_mean = xref[:-num_unc]
        test_mean = np.mean(res[:-num_unc], axis=1, keepdims=True)
        self.assertTrue(np.allclose(ref_mean, test_mean, rtol=1e-4))

    def test_param_logpdf_function(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_dict = {'grp_A': 0.3, 'grp_B': 0.02}
        num_unc = len(unc_dict)
        xref = postdist.stack_params_and_uncs(priorvals, unc_dict)
        xarr = np.hstack([xref]*100)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0)
        ref_cov = np.linalg.inv(invcov.toarray())
        res = propfun(xarr)
        logpdf_vec, inv_logpdf_vec = prop_logpdf(xarr, res)
        mvn = multivariate_normal(mean=priorvals, cov=ref_cov)
        ref_logpdf_vec = mvn.logpdf(res[:-num_unc].T)
        self.assertTrue(np.allclose(logpdf_vec, ref_logpdf_vec))

    def test_unc_proposal_logpdf(self):
        np.random.seed(49)
        priorvals, priorcov, mock_map, expvals, expcov = \
            self.create_mock_quantities()
        unc_idcs = [3, 5, 7, 9]
        unc_group_assoc = ['grp_A', 'grp_B', 'grp_A', 'grp_B']
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map, expvals, expcov,
            relative_exp_errors=True,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        unc_dict = {'grp_A': 0.3, 'grp_B': 0.02}
        param_ref = priorvals + np.random.rand(*priorvals.shape)
        xref = postdist.stack_params_and_uncs(param_ref, unc_dict)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=1, squeeze=False)
        smpl = []
        logpdfvec1 = []
        for i in range(1000):
            cursmpl = propfun(xref)
            smpl.append(cursmpl)
            cur_logpdfvec1, _ = prop_logpdf(cursmpl, cursmpl)
            logpdfvec1.append(cur_logpdfvec1)
        smpl = np.concatenate(smpl, axis=1)
        logpdfvec1 = np.concatenate(logpdfvec1)
        logpdfvec2 = postdist.logpdf(smpl)
        diffs = logpdfvec1 - logpdfvec2
        self.assertTrue(np.isclose(min(diffs), max(diffs)))

    def test_posterior_sampling_1(self):
        # see (1) if sampling from posterior distribution
        # leads to estimate of uncertainty compatible with
        # the data generation process
        n_param = 100
        n_exp = 110
        np.random.seed(49)
        # unc groups
        unc_idcs = np.arange(1, n_param)
        unc_group_assoc = np.full(len(unc_idcs), 'grp_A')
        # prior
        priorvals = 5 + 2 * np.random.rand(n_param, 1)
        priorcov = sps.diags([500] * priorvals.shape[0])
        # system quantities
        yref = 5 + 4 * np.random.rand(n_exp, 1)
        S = sps.csr_matrix(np.random.rand(n_exp, n_param))
        mock_map_linear = self.MockMapLinear(yref, S, priorvals)
        # true values
        real_unc = 20
        truevals = priorvals.copy()
        truevals[unc_idcs, :] += np.random.normal(
            scale=real_unc, size=(len(unc_idcs), 1)
        )
        # and associated experimental values
        expvals = mock_map_linear.propagate(truevals)
        expcov = sps.diags([10.] * len(expvals))
        # define posterior
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map_linear, expvals, expcov,
            relative_exp_errors=False,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        # define mapping
        unc_dict = {'grp_A': real_unc}
        param_ref = truevals.copy()
        xref = postdist.stack_params_and_uncs(param_ref, unc_dict)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0.5, scale=0.1, squeeze=False)
        mh_res = mh_algo(xref, postdist.logpdf, propfun, num_samples=2000,
                         thin_step=100, log_transition_pdf=prop_logpdf,
                         num_burn=100)
        smpl = mh_res['samples']
        uncvec = postdist.extract_uncvec(smpl)
        assert uncvec.shape[0] == 1
        m = np.mean(smpl, axis=1, keepdims=True)
        s = np.std(smpl, axis=1, keepdims=True)
        normdiff = np.abs(m - xref) / (s+1e-10)
        self.assertTrue(np.all(np.max(np.abs(normdiff)) < 2))
        self.assertTrue(mh_res['accept_rate'] > 0.6)
        # check that the mixture sampling is done appropriately
        # by choosing a really bad proposal in the MH step so that
        # accepts only happen in the Gibbs step (sampling the uncertainty)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0.2, scale=1000, squeeze=False)
        mh_res2 = mh_algo(xref, postdist.logpdf, propfun, num_samples=100,
                          thin_step=100, log_transition_pdf=prop_logpdf,
                          num_burn=0)
        self.assertTrue(np.isclose(mh_res2['accept_rate'], 0.2, atol=1e-2))

    def test_posterior_sampling_2(self):
        # see if sampling from posterior distribution
        # with no usu errors present drives the usu uncertainty
        # towards zero
        groups = ['grp_A', 'grp_B', 'grp_C']
        n_groups = len(groups)
        n_normal_params = 10
        n_usu_params = n_groups * 30
        n_param = 10 + n_usu_params
        n_exp = 110
        np.random.seed(49)
        # unc groups
        unc_idcs = np.arange(n_normal_params, n_param)
        unc_group_assoc = np.full(
            len(unc_idcs), groups * int(n_usu_params / n_groups))
        # prior
        priorvals = np.full((n_param, 1), 0.)
        priorvals[:n_normal_params, :] = 10 + 4*np.random.rand(n_normal_params, 1)
        priorcov = np.diag([10000] * priorvals.shape[0])
        # add non-adjustable parameter as well
        nonadj_idx = 5
        assert nonadj_idx < n_normal_params
        priorcov[nonadj_idx, nonadj_idx] = 0.
        priorcov = sps.csr_matrix(priorcov)
        # system quantities
        yref = 5 + 4 * np.random.rand(n_exp, 1)
        S = sps.csr_matrix(np.random.rand(n_exp, n_param))
        mock_map_linear = self.MockMapLinear(yref, S, priorvals)
        # true values
        real_unc = 1e-10
        truevals = priorvals.copy()
        truevals[:n_normal_params, :] += np.random.normal(
            scale=15, size=(n_normal_params, 1)
        )
        truevals[unc_idcs, :] += np.random.normal(
            scale=real_unc, size=(len(unc_idcs), 1)
        )
        truevals[nonadj_idx, :] = priorvals[nonadj_idx, :]
        # and associated experimental values
        expvals = mock_map_linear.propagate(truevals)
        expcov = sps.diags([10.] * len(expvals))
        # define posterior
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map_linear, expvals, expcov,
            relative_exp_errors=False,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        # define mapping
        unc_dict = {k: real_unc for k in groups}
        param_ref = truevals.copy()
        xref = postdist.stack_params_and_uncs(param_ref, unc_dict)
        propfun, prop_logpdf, invcov = \
            postdist.generate_proposal_fun(xref, rho=0.5, scale=0.1, squeeze=False)
        mh_res = mh_algo(xref, postdist.logpdf, propfun, num_samples=2000,
                         thin_step=100, log_transition_pdf=prop_logpdf,
                         num_burn=100)
        smpl = mh_res['samples']
        m = np.mean(smpl, axis=1, keepdims=True)
        s = np.std(smpl, axis=1, keepdims=True)
        normdiff = np.abs(m - xref) / (s+1e-10)
        self.assertTrue(np.all(np.max(normdiff[:-n_groups, :]) < 2))
        self.assertTrue(np.all(np.max(normdiff[-n_groups:, :]) < 3.5))
        self.assertTrue(mh_res['accept_rate'] > 0.6)
        self.assertTrue(np.all(smpl[nonadj_idx, :] == smpl[nonadj_idx, 0]))

    def test_unc_adjustment_in_approximate_postmode_and_logpdf(self):
        # approximate_postmode with huge lmb parameter
        # should lead only to an update of the uncertainty
        # parameters that should
        groups = ['grp_A', 'grp_B', 'grp_C']
        n_groups = len(groups)
        n_normal_params = 10
        n_usu_params = n_groups * 30
        n_param = 10 + n_usu_params
        n_exp = 110
        np.random.seed(56)
        # unc groups
        unc_idcs = np.arange(n_normal_params, n_param)
        unc_group_assoc = np.full(
            len(unc_idcs), groups * int(n_usu_params / n_groups))
        # prior
        priorvals = np.full((n_param, 1), 0.)
        priorvals[:n_normal_params, :] = 10 + 4*np.random.rand(n_normal_params, 1)
        priorcov = np.diag([10000] * priorvals.shape[0])
        # add non-adjustable parameter as well
        nonadj_idx = 5
        assert nonadj_idx < n_normal_params
        priorcov[nonadj_idx, nonadj_idx] = 0.
        priorcov = sps.csr_matrix(priorcov)
        # system quantities
        yref = 5 + 4 * np.random.rand(n_exp, 1)
        S = sps.csr_matrix(np.random.rand(n_exp, n_param))
        mock_map_linear = self.MockMapLinear(yref, S, priorvals)
        # true values
        real_unc = 70
        truevals = priorvals.copy()
        truevals[:n_normal_params, :] += np.random.normal(
            scale=15, size=(n_normal_params, 1)
        )
        truevals[unc_idcs, :] += np.random.normal(
            scale=real_unc, size=(len(unc_idcs), 1)
        )
        truevals[nonadj_idx, :] = priorvals[nonadj_idx, :]
        # and associated experimental values
        expvals = mock_map_linear.propagate(truevals)
        expcov = sps.diags([10.] * len(expvals))
        # define posterior
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map_linear, expvals, expcov,
            relative_exp_errors=False,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        # define mapping
        unc_dict = {k: real_unc/10 for k in groups}
        param_ref = truevals.copy()
        xref = postdist.stack_params_and_uncs(param_ref, unc_dict)
        new_x = postdist.approximate_postmode(xref, lmb=1e16)
        new_params = postdist.extract_params(new_x)
        self.assertTrue(np.allclose(new_params, truevals))
        self.assertTrue(new_params[nonadj_idx] == truevals[nonadj_idx])
        # check if we find really the mode of the
        # posterior distribution conditioned on parameters
        new_uncvec = postdist.extract_uncvec(new_x)
        for i in range(len(new_uncvec)):
            tmp_up = new_uncvec.copy()
            tmp_down = new_uncvec.copy()
            tmp_up[i] += 0.01
            tmp_down[i] -= 0.01
            full_tmp_up = np.vstack([new_params, tmp_up])
            full_tmp_down = np.vstack([new_params, tmp_down])
            logprob = postdist.logpdf(new_x)
            logprob_up = postdist.logpdf(full_tmp_up)
            logprob_down = postdist.logpdf(full_tmp_down)
            self.assertTrue(logprob_up < logprob)
            self.assertTrue(logprob_down < logprob)
        # check the approximate_logpdf functionality
        logpdf_ref = postdist.logpdf(xref)
        logpdf_exact = postdist.logpdf(new_x)
        logpdf_approx = postdist.approximate_logpdf(xref, new_x)
        self.assertTrue(np.isclose(logpdf_exact, logpdf_approx))
        self.assertFalse(np.isclose(logpdf_ref, logpdf_exact))

    def test_check_lm_update_convergence(self):
        groups = ['grp_A', 'grp_B', 'grp_C']
        n_groups = len(groups)
        n_normal_params = 10
        n_usu_params = n_groups * 30
        n_param = 10 + n_usu_params
        n_exp = 110
        np.random.seed(56)
        # unc groups
        unc_idcs = np.arange(n_normal_params, n_param)
        unc_group_assoc = np.full(
            len(unc_idcs), groups * int(n_usu_params / n_groups))
        # prior
        priorvals = np.full((n_param, 1), 0.)
        priorvals[:n_normal_params, :] = 10 + 4*np.random.rand(n_normal_params, 1)
        priorcov = sps.diags([10000] * priorvals.shape[0])
        # system quantities
        yref = 5 + 4 * np.random.rand(n_exp, 1)
        S = sps.csr_matrix(np.random.rand(n_exp, n_param))
        mock_map_linear = self.MockMapLinear(yref, S, priorvals)
        # true values
        real_unc = 70
        truevals = priorvals.copy()
        truevals[:n_normal_params, :] += np.random.normal(
            scale=15, size=(n_normal_params, 1)
        )
        truevals[unc_idcs, :] += np.random.normal(
            scale=real_unc, size=(len(unc_idcs), 1)
        )
        # and associated experimental values
        expvals = mock_map_linear.propagate(truevals)
        expcov = sps.diags([1e-8] * len(expvals))
        # define posterior
        postdist = PosteriorUSU(
            priorvals, priorcov, mock_map_linear, expvals, expcov,
            relative_exp_errors=False,
            unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
        )
        # define mapping
        unc_dict = {k: real_unc/100 for k in groups}
        param_ref = priorvals.copy()
        xref = postdist.stack_params_and_uncs(param_ref, unc_dict)
        # check the convergence of the lm_update function
        lm_res = lm_update(postdist, xref, print_status=True)
        test_vals = lm_res['upd_vals']
        test_params = postdist.extract_params(test_vals)
        self.assertTrue(np.allclose(
            test_params.flatten(), truevals.flatten(), atol=1e-8, rtol=1e-6
        ))
        # check if we find really the mode of the
        # posterior distribution conditioned on parameters
        new_uncvec = postdist.extract_uncvec(test_vals)
        for i in range(len(new_uncvec)):
            tmp_up = new_uncvec.copy()
            tmp_down = new_uncvec.copy()
            tmp_up[i] += 0.01
            tmp_down[i] -= 0.01
            full_tmp_up = np.concatenate([test_params, tmp_up])
            full_tmp_down = np.concatenate([test_params, tmp_down])
            logprob = postdist.logpdf(test_vals)
            logprob_up = postdist.logpdf(full_tmp_up)
            logprob_down = postdist.logpdf(full_tmp_down)
            self.assertTrue(logprob_up < logprob)
            self.assertTrue(logprob_down < logprob)


if __name__ == '__main__':
    unittest.main()
