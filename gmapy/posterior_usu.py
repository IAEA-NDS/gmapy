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
from scipy.sparse import coo_matrix
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
        # create a version of unc_group_dict for the covariance matrix
        # that may be of smaller dimension due to non-adjustable parameters
        if (unc_idcs is not None and
                np.any((priorcov.diagonal() == 0)[unc_idcs])):
            raise IndexError('unc_idcs refer to non-adjustable parameters')
        adjustable_idcs = np.arange(len(priorvals))[priorcov.diagonal() > 0]
        self._adjunc_group_dict = {
            k: np.searchsorted(adjustable_idcs, v, side='left')
            for k, v in self._unc_group_dict.items()
        }
        # NOTE: we need to take self._priorcov in the following as
        #       the constructor of Posterior rearranges the indices
        #       in the COO Matrix instance
        self._adjunc_group_dict2 = self._determine_idcs_in_coo_matrix(
            self._priorcov, self._adjunc_group_dict
        )
        # caching some vars to accelerate
        self.__cache = {}

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
        found_diff = 'uncvec' not in self.__cache
        if not found_diff:
            found_diff = np.any(self.__cache['uncvec'] != uncvec)
        if found_diff:
            self.__cache['uncvec'] = uncvec.copy()
            for i, group in enumerate(self._groups):
                idcs = self._adjunc_group_dict2[group]
                squared_unc = uncvec[i]*uncvec[i]
                self._priorcov.data[idcs] = squared_unc
            self._priorfact.cholesky_inplace(
                self._priorcov.tocsc(copy=True)
            )
        return found_diff

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
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(xref.shape) == 1:
            xref = xref.reshape(-1, 1)
        uncvec_ref = self.extract_uncvec(xref)
        uncarr = self.extract_uncvec(x)
        params_ref = self.extract_params(xref)
        params = self.extract_params(x)
        logpdfs = np.empty(x.shape[1], dtype=float)
        for i in range(x.shape[1]):
            self._update_priorcov_if_necessary(uncvec_ref[:, i])
            logpdfs[i] = self._logpdf(params[:, i], xref=params_ref)
        # post-hoc adjustment to account for uncertainty maximization
        alpha_betas_ref = self._determine_alpha_beta(xref)
        refval = 0
        for i, (group, alpha, beta) in enumerate(alpha_betas_ref):
            num = alpha*2 + 1
            u = uncvec_ref[i]
            refval -= beta / (u*u) + num*np.log(u)
        alpha_betas = self._determine_alpha_beta(x)
        tarvals = np.zeros((1, xref.shape[1]), dtype=float)
        for i, (group, alpha, beta) in enumerate(alpha_betas):
            num = alpha*2 + 1
            us = uncarr[i, :]
            tarvals -= beta / (us*us) + num*np.log(us)
        logpdf_adj = tarvals - refval
        tmp = self._logpdf(params, xref=params_ref)
        return (tmp + logpdf_adj).reshape(-1)

    def grad_logpdf(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        uncvec = self.extract_uncvec(x)
        gradarr = np.full(x.shape, 0., dtype=float)
        for i in range(x.shape[1]):
            self._update_priorcov_if_necessary(uncvec[:, i])
            params = self.extract_params(x[:, i])
            tmp = super().grad_logpdf(params)
            gradarr[:params.shape[0], i] = tmp
        if self._apply_squeeze:
            gradarr = np.squeeze(gradarr)
        return gradarr

    def approximate_postmode(self, xref, lmb=0.):
        if len(xref.shape) == 1:
            xref = xref.reshape(-1, 1)
        parvec = self.extract_params(xref)
        new_parvec = super().approximate_postmode(parvec, lmb=lmb)
        alpha_betas = self._determine_alpha_beta(new_parvec)
        new_uncvec = np.empty((len(alpha_betas), xref.shape[1]), dtype=float)
        for i, (group, alpha, beta) in enumerate(alpha_betas):
            # determine uncertainty corresponding to mode
            # of inverse gamma distribution
            curunc = np.sqrt(beta / (alpha + 0.5))
            new_uncvec[i, :] = curunc
        return np.concatenate([new_parvec, new_uncvec], axis=0)

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
            nonlocal fact
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            xr = self.extract_params(x)
            adj = self._adj
            numadj = self._numadj
            rvec = np.random.normal(size=(numadj, xr.shape[1]))
            _update_propcov(self.extract_uncvec(x))
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
            nonlocal fact, fact_logdet
            uncs = self.extract_uncvec(x)
            prop_uncs = self.extract_uncvec(propx)
            if np.any(uncs != prop_uncs):
                return -np.inf
            xr = self.extract_params(x)
            propxr = self.extract_params(propx)
            if np.any(xr[self._nonadj] != propxr[self._nonadj]):
                return -np.inf
            _update_propcov(uncs)
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
            # calculate inverse transition propx -> x
            inv_log_p1 = log_p1
            inv_log_p2 = unc_proposal_logpdf(propx, x)
            inv_contrib1 = log_1mrho + inv_log_p1
            inv_contrib2 = log_rho + inv_log_p2
            inv_m = np.maximum(inv_contrib1, inv_contrib2)
            inv_r = np.log(
                np.exp(inv_contrib1 - inv_m) +
                np.exp(inv_contrib2 - inv_m)
            ) + inv_m
            if squeeze and len(r) == 1:
                return r[0], inv_r[0]
            return r, inv_r

        def _update_propcov(uncvec):
            nonlocal ST_invexpcov_S, invcov
            nonlocal fact, fact_logdet
            updated = self._update_priorcov_if_necessary(uncvec)
            if updated:
                invcov = ST_invexpcov_S + self._priorfact.inv().tocsc()
                fact.cholesky_inplace(invcov)
                fact_logdet = fact.logdet()

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
        # begin of vars used in param_proposal_logpdf and param_proposal
        ST_invexpcov_S = (S.T @ self._expfact(S.tocsc())).tocsc()
        invcov = ST_invexpcov_S + self._priorfact.inv()
        fact = cholesky(invcov)
        fact_logdet = fact.logdet()
        # end of vars
        del S
        return proposal, proposal_logpdf, invcov
