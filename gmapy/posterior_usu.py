############################################################
#
# Author(s):       Georg Schnabel
# Email:           g.schnabel@iaea.org
# Creation date:   2023/04/14
# Last modified:   2023/04/14
# License:         MIT
# Copyright (c) 2023 International Atomic Energy Agency (IAEA)
#
############################################################
import numpy as np
from sksparse.cholmod import cholesky
import scipy.sparse as sps
from scipy.sparse import csr_matrix, coo_matrix, identity
from .posterior import Posterior
from scipy.stats import invgamma


class PosteriorUSU(Posterior):

    def __init__(self, priorvals, priorcov, mapping, expvals, expcov,
                 squeeze=True, relative_exp_errors=False,
                 source_mask=None, target_mask=None,
                 unc_idcs=None, unc_group_assoc=None):

        if len(unc_idcs) != len(unc_group_assoc):
            raise TypeError('adj_unc_idcs and unc_group_assoc ' +
                            'must be of the same length')
        priorcov = self._prepare_priorcov(priorcov, unc_idcs)
        self._unc_group_dict = \
            self._prepare_unc_group_dict(unc_idcs, unc_group_assoc)
        self._groups = np.array(tuple(self._unc_group_dict.keys()))
        super().__init__(
            priorvals, priorcov, mapping, expvals, expcov,
            squeeze=squeeze, relative_exp_errors=relative_exp_errors,
            source_mask=source_mask, target_mask=target_mask
        )
        # NOTE: we need to take self._priorcov in the following as
        #       the constructor of Posterior rearranges the indices
        #       in the COO Matrix instance
        self._unc_group_dict2 = self._determine_idcs_in_coo_matrix(
            self._priorcov, self._unc_group_dict
        )

    def _prepare_priorcov(self, priorcov, unc_idcs):
        priorcov = coo_matrix(priorcov)
        rowsel = ~np.isin(priorcov.row, unc_idcs)
        colsel = ~np.isin(priorcov.col, unc_idcs)
        sel = np.logical_and(rowsel, colsel)
        red_rows = priorcov.row[sel]
        red_cols = priorcov.col[sel]
        red_data = priorcov.data[sel]
        comb_rows = np.concatenate([red_rows, unc_idcs])
        comb_cols = np.concatenate([red_cols, unc_idcs])
        comb_data = np.concatenate([red_data, np.full(len(unc_idcs), 1.0)])
        return coo_matrix(
            (comb_data, (comb_rows, comb_cols)),
            shape=priorcov.shape, dtype=float, copy=True
        )

    def _prepare_unc_group_dict(self, unc_idcs, unc_group_assoc):
        unc_group_assoc = np.array(unc_group_assoc)
        unc_idcs = np.array(unc_idcs)
        groups = np.unique(unc_group_assoc)
        group_dict = {
            group: unc_idcs[unc_group_assoc == group] for group in groups
        }
        return group_dict

    def _determine_idcs_in_coo_matrix(self, mat, unc_group_dict):
        idcs_dict = {k: list(v) for k, v in unc_group_dict.items()}
        for group in tuple(idcs_dict):
            idcs = idcs_dict[group]
            row_sel = np.isin(mat.row, idcs)
            col_sel = np.isin(mat.col, idcs)
            if not np.all(row_sel == col_sel):
                raise IndexError('covariance matrix block associated ' +
                                 'with adjustable uncertainties is not' +
                                 'a diagonal matrix')
            idcs_dict[group] = np.where(row_sel)[0]
        return idcs_dict

    def _update_priorcov_if_necessary(self, uncvec):
        found_diff = False
        diag = self._priorcov.diagonal()
        for i, group in enumerate(self._groups):
            idcs = self._unc_group_dict[group]
            squared_unc = uncvec[i]*uncvec[i]
            if np.any(diag[idcs] != squared_unc):
                found_diff = True
                break
        if found_diff:
            for i, group in enumerate(self._groups):
                idcs = self._unc_group_dict2[group]
                squared_unc = uncvec[i]*uncvec[i]
                self._priorcov.data[idcs] = squared_unc
            self._priorfact.cholesky_inplace(
                self._priorcov.tocsc(copy=True)
            )

    def _determine_alpha_beta(self, x):
        res = []
        for i, group in enumerate(self._groups):
            idcs = self._unc_group_dict[group]
            usu_errors = x[idcs, :]
            d = usu_errors - self._priorvals[idcs, :]
            beta = 0.5 * np.sum(np.square(d), axis=0)
            alpha = 0.5 * (usu_errors.shape[0] - 1.)
            res.append((group, alpha, beta))
        return tuple(res)

    def extract_params(self, x):
        fidx = len(x) - len(self._groups)
        if len(x.shape) == 1:
            return x[0:fidx].copy()
        elif len(x.shape) == 2:
            return x[0:fidx, :].copy()

    def extract_uncvec(self, x):
        sidx = len(x) - len(self._groups)
        return x[sidx:].copy()

    def stack_params_and_uncs(self, params, uncs):
        if len(params.shape) == 1:
            params = params.reshape(-1, 1)
        vec_list = [params]
        for group in self._groups:
            vec_list.append(np.array([uncs[group]]).reshape(-1, 1))
        res = np.concatenate(vec_list, axis=0)
        return res

    def unstack_params_and_uncs(self, x):
        params = self.extract_params(x)
        uncvec = self.extract_uncvec(x)
        uncs = {group: uncvec[i] for i, group in enumerate(self._groups)}
        return params, uncs

    # interface for important quantities for Bayesian inference

    def logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        uncvec = self.extract_uncvec(x)
        logpdfs = []
        for i in range(x.shape[1]):
            self._update_priorcov_if_necessary(uncvec[:, i])
            params = self.extract_params(x[:, i])
            logpdfs.append(self._logpdf(params))
        return np.array(logpdfs)

    def approximate_logpdf(self, xref, x):
        raise NotImplementedError('approximate_logpdf method not implemented')
        return self._logpdf(x, xref=xref)

    def grad_logpdf(self, x):
        raise NotImplementedError('grad_logpdf method not implemented')
        x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != 1:
            raise ValueError('x must be a vector and not a matrix')
        adj = self._adj
        nonadj = self._nonadj
        # gradient of prior contribution
        pf = self._priorfact
        d1 = x[adj] - self._priorvals[adj]
        z1r = pf(d1)
        z1 = np.zeros(self._priorvals.shape, dtype=float)
        z1[adj, :] = (-z1r)
        # gradient of likelihood contribution
        m = self._mapping
        ef = self._expfact
        propx = m.propagate(x.flatten()).reshape(-1, 1)
        S = m.jacobian(x.flatten())
        if not self._relative_exp_errors:
            d2 = self._expvals - propx
            z2 = S.T @ ef(d2)
            if self._debug_only_likelihood_chisquare:
                z2[nonadj, :] = 0.
                return -2*z2.flatten()
        else:
            propx2 = self._get_propx2(x)
            d2 = self._get_d2(propx, propx2)
            inv_expcov_times_d2 = ef(d2)
            d2deriv = self._exp_pred_diff_jacobian(S, propx, propx2)
            z2 = ((-1) * (inv_expcov_times_d2.T @ d2deriv))
            if self._debug_only_likelihood_chisquare:
                z2[:, nonadj] = 0.
                return -2*z2.flatten()
            if self._debug_only_likelihood_logdet:
                return self._likelihood_logdet_jacobian(S, propx2).flatten()
            z2 -= 0.5 * self._likelihood_logdet_jacobian(S, propx2)
            z2 = z2.T

        z2[nonadj, :] = 0.
        res = z1 + z2
        if self._apply_squeeze:
            res = np.squeeze(res)
        return res

    def approximate_postmode(self, xref, lmb=0.):
        raise NotImplementedError('approximate_postmode not implemented')
        priorvals = self._priorvals
        # calculate the inverse posterior covariance matrix
        xref = xref.copy()
        xref[self._nonadj] = priorvals.flatten()[self._nonadj]
        inv_post_cov = self._approximate_invpostcov(xref)
        dampmat = lmb * identity(inv_post_cov.shape[0],
                                 dtype=float, format='csr')
        inv_post_cov += dampmat
        # calculate the gradient of the difference in
        # the experimental chisquare value
        zvec = self.grad_logpdf(xref)
        zvec = zvec[self._adj]
        postvals = xref.reshape(-1, 1)[self._adj]
        postvals += sps.linalg.spsolve(inv_post_cov, zvec).reshape(-1, 1)
        xref[self._adj] = postvals.flatten()
        return xref

    def approximate_postcov(self, xref):
        uncvec = self.extract_uncvec(xref)
        self._update_priorcov_if_necessary(uncvec)
        params = self.extract_params(xref)
        return super().approximate_postcov(params)

    def generate_proposal_fun(self, xref, scale=1., rho=0.5, squeeze=False):

        def unc_proposal(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            uncvec = np.empty((len(self._groups), x.shape[1]), dtype=float)
            alpha_betas = self._determine_alpha_beta(x)
            for i, (group, alpha, beta) in enumerate(alpha_betas):
                rv = invgamma.rvs(alpha, size=x.shape[1])
                uncvec[i, :] = np.sqrt(rv * beta)
            propx = x.copy()
            propx[-len(self._groups):, :] = uncvec
            return propx

        def param_proposal(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            xr = self.extract_params(x)
            adj = self._adj
            numadj = self._numadj
            rvec = np.random.normal(size=(numadj, xr.shape[1]))
            d = fact.apply_Pt(fact.solve_Lt(rvec, use_LDLt_decomposition=False))
            rres = xr.copy()
            rres[adj, :] += d*scale
            res = np.empty(x.shape, dtype=float)
            res[:-len(self._groups), :] = rres
            res[-len(self._groups):, :] = self.extract_uncvec(x)
            return res

        def proposal(x):
            z = np.random.rand()
            if z > rho:
                res = param_proposal(x)
            else:
                res = unc_proposal(x)
            if squeeze:
                res = np.squeeze(res)
            return res

        def param_proposal_logpdf(x, propx):
            uncs = self.extract_uncvec(x)
            prop_uncs = self.extract_uncvec(propx)
            if np.any(uncs != prop_uncs):
                return -np.inf
            xr = self.extract_params(x)
            propxr = self.extract_params(propx)
            if np.any(xr[self._nonadj] != propxr[self._nonadj]):
                return -np.inf
            d = (propxr[self._adj] - xr[self._adj]) / scale
            dp = fact.apply_P(d)
            z = fact.L().T @ dp
            # chisqr = np.sum(d * (invcov @ d), axis=0)
            chisqr = np.sum(np.square(z), axis=0)
            logdet = fact_logdet
            # -logdet because inverse posterior covariance matrix
            res = -0.5 * (chisqr - logdet + np.log(2*np.pi)*self._numadj)
            return res

        def unc_proposal_logpdf(x, propx):
            params = self.extract_params(x)
            prop_params = self.extract_params(propx)
            if np.any(params != prop_params):
                return -np.inf
            uncvec = self.extract_uncvec(propx)
            alpha_betas = self._determine_alpha_beta(x)
            log_prob = 0.
            for i, (group, alpha, beta) in enumerate(alpha_betas):
                u = uncvec[i]
                z = u*u / beta
                log_prob += invgamma.logpdf(z, a=alpha) + np.log(2*u / beta)
            return log_prob

        def proposal_logpdf(x, propx):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            if len(propx.shape) == 1:
                propx = propx.reshape(-1, 1)
            log_p1 = param_proposal_logpdf(x, propx)
            log_p2 = unc_proposal_logpdf(x, propx)
            contrib1 = log_1mrho + log_p1
            contrib2 = log_rho + log_p2
            m = np.maximum(contrib1, contrib2)
            r = np.log(np.exp(contrib1-m) + np.exp(contrib2 - m)) + m
            if squeeze and len(r) == 1:
                r = r[0]
            return r

        if rho < 0. or rho > 1:
            raise ValueError('violation of constraint 0 <= rho <= 1')
        elif rho == 0.:
            log_rho = -np.inf
            log_1mrho = 0.
        elif rho == 1.:
            log_rho = 0.
            log_1mrho = -np.inf
        else:
            log_rho = np.log(rho)
            log_1mrho = np.log(1-rho)

        uncref = self.extract_uncvec(xref)
        self._update_priorcov_if_necessary(uncref)
        xref = self.extract_params(xref)
        S = self._mapping.jacobian(xref).tocsc()[:, self._adj]
        pf = self._priorfact
        ef = self._expfact
        invcov = (S.T @ ef(S.tocsc()) + pf.inv()).tocsc()
        fact = cholesky(invcov)
        fact_logdet = fact.logdet()
        # TODO: calculate the determinant and the chisquare value
        #       because it will serve as the normalization constant
        del S
        return proposal, proposal_logpdf, invcov
