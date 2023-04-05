############################################################
#
# Author(s):       Georg Schnabel
# Email:           g.schnabel@iaea.org
# Creation date:   2023/04/05
# Last modified:   2023/04/05
# License:         MIT
# Copyright (c) 2023 International Atomic Energy Agency (IAEA)
#
############################################################
import numpy as np
from sksparse.cholmod import cholesky
import scipy.sparse as sps
from scipy.sparse import csr_matrix, coo_matrix, identity
from .mappings.priortools import apply_mask


class Posterior:

    def __init__(self, priorvals, priorcov, mapping, expvals, expcov,
                 squeeze=True, relative_exp_errors=False,
                 source_mask=None, target_mask=None):
        self.__priorvals = priorvals.reshape(-1, 1)
        adjustable = priorcov.diagonal() != 0.
        priorcov = priorcov.tocsr()[adjustable,:].tocsc()[:,adjustable]
        self.__size = priorvals.size
        self.__adj = adjustable
        self.__adj_idcs = np.where(adjustable)[0]
        self.__nonadj = np.logical_not(adjustable)
        self.__numadj = np.sum(adjustable)
        self.__priorfact = cholesky(priorcov)
        self.__mapping = mapping
        self.__expvals = expvals.reshape(-1, 1)
        self.__expfact = cholesky(expcov)
        self.__relative_exp_errors = relative_exp_errors
        self.__apply_squeeze = squeeze
        self.__source_mask = source_mask
        self.__target_mask = target_mask
        # for debugging
        self._debug_only_likelihood_logdet = False
        self._debug_only_likelihood_chisquare = False

    def set_squeeze(self, flag):
        self.__apply_squeeze = flag

    def set_relative_exp_errors(self, flag):
        self.__relative_exp_errors = flag

    def get_priorvals(self):
        return self.__priorvals.flatten()

    def update_expcov(self, new_expcov):
        new_expcov = csr_matrix(new_expcov)
        self.__expfact = cholesky(new_expcov)

    def logpdf(self, x):
        return self._logpdf(x)

    def _get_propx(self, x):
        return self.__mapping.propagate(x.flatten()).reshape(-1, 1)

    def _get_propx2(self, x):
        m = self.__mapping
        x2 = x.copy()
        apply_mask(x2, self.__source_mask)
        propx2 = m.propagate(x2.flatten()).reshape(-1, 1)
        apply_mask(propx2,  self.__target_mask)
        return propx2

    def _get_d2(self, propx, propx2):
        return (self.__expvals - propx) * self.__expvals / propx2

    def _exp_pred_diff_jacobian(self, S, propx, propx2):
        outer_jac1 = self.__expvals / propx2
        outer_jac2 = (self.__expvals - propx)
        outer_jac2 *= self.__expvals / np.square(propx2)
        if self.__target_mask is not None:
            outer_jac2[self.__target_mask['idcs']] = 0.
        z2a = S.T.multiply(outer_jac1.T).tocsr()
        z2b = S.T.multiply(outer_jac2.T).tocsr()
        if self.__source_mask is not None:
            z2b[self.__source_mask['idcs']] = 0.
        return -(z2a + z2b).T

    def _likelihood_logdet(self, propx2):
        res = self.__expfact.logdet()
        scl = propx2 / self.__expvals
        res += 2*np.sum(np.log(np.abs(scl)))
        return res

    def _likelihood_logdet_jacobian(self, S, propx2):
        outer_jac_det = (2/propx2).reshape(-1, 1)
        if self.__target_mask is not None:
            outer_jac_det[self.__target_mask['idcs']] = 0.
        res = S.T @ outer_jac_det
        if self.__source_mask is not None:
            res[self.__source_mask['idcs']] = 0.
        return res.T

    def _likelihood_logdet_approximate_hessian(self, S, propx2):
        outer_2nd_deriv = (-2/np.square(propx2)).flatten()
        if self.__target_mask is not None:
            outer_2nd_deriv[self.__target_mask['idcs']] = 0.
        s = len(outer_2nd_deriv)
        U = sps.spdiags(outer_2nd_deriv, diags=0, m=s, n=s)
        if self.__source_mask is not None:
            ridcs = self.__source_mask['idcs']
            mask = np.full(S.shape[1], True)
            mask[ridcs] = False
            kidcs = np.where(mask)[0]
            R = csr_matrix(([1.]*len(kidcs), (kidcs, kidcs)),
                           shape=(S.shape[1], S.shape[1]))
            S = S @ R
        return S.T @ U @ S

    def _likelihood_logdet_taylorapprox(self, x, logdet_ref, xref, propx2, S):
        T1 = self._likelihood_logdet_jacobian(S, propx2)
        T2 = self._likelihood_logdet_approximate_hessian(S, propx2)
        d = x.reshape(-1, 1) - xref.reshape(-1, 1)
        ypred = logdet_ref + T1 @ d + 0.5 * d.T @ T2 @ d
        return ypred

    def grad_logpdf(self, x):
        x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != 1:
            raise ValueError('x must be a vector and not a matrix')
        adj = self.__adj
        nonadj = self.__nonadj
        # gradient of prior contribution
        pf = self.__priorfact
        d1 = x[adj] - self.__priorvals[adj]
        z1r = pf(d1)
        z1 = np.zeros(self.__priorvals.shape, dtype=float)
        z1[adj, :] = (-z1r)
        # gradient of likelihood contribution
        m = self.__mapping
        ef = self.__expfact
        propx = m.propagate(x.flatten()).reshape(-1, 1)
        S = m.jacobian(x.flatten())
        if not self.__relative_exp_errors:
            d2 = self.__expvals - propx
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
        if self.__apply_squeeze:
            res = np.squeeze(res)
        return res

    def generate_proposal_fun(self, xref, scale=1.):
        def proposal(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            adj = self.__adj
            numadj = self.__numadj
            rvec = np.random.normal(size=(numadj, x.shape[1]))
            d = fact.apply_Pt(fact.solve_Lt(rvec, use_LDLt_decomposition=False))
            res = x.copy()
            res[adj] += d*scale
            return res
        S = self.__mapping.jacobian(xref).tocsc()[:, self.__adj]
        pf = self.__priorfact
        ef = self.__expfact
        tmp = (S.T @ ef(S.tocsc()) + pf.inv()).tocsc()
        fact = cholesky(tmp)
        del tmp
        del S
        return proposal

    def approximate_logpdf(self, xref, x):
        return self._logpdf(x, xref=xref)

    def approximate_postmode(self, xref, lmb=0.):
        priorvals = self.__priorvals
        # calculate the inverse posterior covariance matrix
        xref = xref.copy()
        xref[self.__nonadj] = priorvals.flatten()[self.__nonadj]
        inv_post_cov = self._approximate_invpostcov(xref)
        dampmat = lmb * identity(inv_post_cov.shape[0],
                                 dtype=float, format='csr')
        inv_post_cov += dampmat
        # calculate the gradient of the difference in
        # the experimental chisquare value
        zvec = self.grad_logpdf(xref)
        zvec = zvec[self.__adj]
        postvals = xref.reshape(-1, 1)[self.__adj]
        postvals += sps.linalg.spsolve(inv_post_cov, zvec).reshape(-1, 1)
        xref[self.__adj] = postvals.flatten()
        return xref

    def approximate_postcov(self, xref):
        xref = xref.flatten()
        xref[self.__nonadj] = self.__priorvals[self.__nonadj, 0]
        adjidcs = self.__adj_idcs
        size = self.__size
        tmp = coo_matrix(sps.linalg.inv(
            self._approximate_invpostcov(xref)
        ))
        postcov = csr_matrix((tmp.data, (adjidcs[tmp.row], adjidcs[tmp.col])),
                             dtype=float, shape=(size, size))
        return postcov

    def _logpdf(self, x, xref=None):
        x = x.copy()
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        adj = self.__adj
        nonadj = self.__nonadj
        x[nonadj] = self.__priorvals[nonadj]
        # prior contribution
        pf = self.__priorfact
        d1 = x[adj] - self.__priorvals[adj]
        d1_perm = pf.apply_P(d1)
        z1 = pf.solve_L(d1_perm, use_LDLt_decomposition=False)
        # likelihood contribution
        m = self.__mapping
        ef = self.__expfact
        if xref is None:
            propx = np.hstack([m.propagate(x[:,i]).reshape(-1,1)
                              for i in range(x.shape[1])])
            d2 = self.__expvals - propx
            if self.__relative_exp_errors:
                x2 = x.copy()
                apply_mask(x2, self.__source_mask)
                propx2 = np.hstack([m.propagate(x2[:,i]).reshape(-1,1)
                                   for i in range(x2.shape[1])])
                apply_mask(propx2, self.__target_mask)
                scl = propx2 / self.__expvals
                d2 /= scl
        else:
            xref = xref.reshape(-1, 1)
            xref[nonadj, :] = self.__priorvals[nonadj, :]
            S = m.jacobian(xref.flatten())
            if not self.__relative_exp_errors:
                yref = m.propagate(xref.flatten()).reshape(-1, 1)
                propx = yref + S @ (x - xref)
                d2 = self.__expvals - propx
            else:
                propx_ref = m.propagate(xref.flatten()).reshape(-1, 1)
                x2ref = xref.copy()
                apply_mask(x2ref, self.__source_mask)
                propx2_ref = m.propagate(x2ref.flatten()).reshape(-1, 1)
                apply_mask(propx2_ref, self.__target_mask)
                d2ref = self._get_d2(propx_ref, propx2_ref)
                J = self._exp_pred_diff_jacobian(S, propx_ref, propx2_ref)
                x2 = x.copy()
                apply_mask(x2, self.__source_mask)
                d2 = d2ref + J @ (x2 - x2ref)
        d2_perm = ef.apply_P(d2)
        z2 = ef.solve_L(d2_perm, use_LDLt_decomposition=False)
        prior_res = np.sum(np.square(z1), axis=0)
        t = pf.D()
        prior_logdet = np.sum(np.log(t[~np.isposinf(t)]))
        prior_res += prior_logdet + np.pi*len(d1)
        like_res = np.sum(np.square(z2), axis=0) + np.pi*len(d2)

        if self._debug_only_likelihood_chisquare:
            return like_res

        if self.__relative_exp_errors:
            if xref is None:
                like_logdet = self._likelihood_logdet(propx2)
            else:
                logdet_ref = self._likelihood_logdet(propx2_ref)
                like_logdet = self._likelihood_logdet_taylorapprox(
                    x2, logdet_ref, x2ref, propx2_ref, S
                ).reshape(1)
        else:
            like_logdet = ef.logdet()

        if self._debug_only_likelihood_logdet:
            return like_logdet

        like_res += like_logdet
        res = -0.5 * (prior_res + like_res)
        if self.__apply_squeeze:
            res = np.squeeze(res)
        return res

    def _approximate_invpostcov(self, xref):
        xref = xref.flatten()
        pf = self.__priorfact
        ef = self.__expfact
        invpostcov = pf.inv()
        S = self.__mapping.jacobian(xref)
        if not self.__relative_exp_errors:
            J = S.tocsc()[:, self.__adj]
        if self.__relative_exp_errors:
            propx = self._get_propx(xref)
            propx2 = self._get_propx2(xref)
            logdet_hessian = \
                self._likelihood_logdet_approximate_hessian(S, propx)
            logdet_hessian = logdet_hessian[:, self.__adj][self.__adj, :]
            invpostcov += 0.5 * logdet_hessian
            J = self._exp_pred_diff_jacobian(S, propx, propx2)
            J = J.tocsc()[:, self.__adj]
        invpostcov += J.T @ ef(J.tocsc())
        invpostcov = invpostcov.tocsc()
        return invpostcov

