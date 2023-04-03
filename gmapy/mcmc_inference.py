import numpy as np
from sksparse.cholmod import cholesky
import scipy.sparse as sps
from scipy.sparse import csr_matrix, coo_matrix, identity
from statsmodels.tsa.stattools import acf
from .mappings.compound_map import CompoundMap
from .mappings.priortools import (
    apply_mask
)
from .inference import lm_update
from multiprocessing import Process, Pipe
import time


def symmetric_mh_algo(startvals, log_probdens, proposal, num_samples,
                      num_burn=0, thin_step=1, seed=None, print_info=False):
    if seed is not None:
        np.random.seed(seed)
    startvals = startvals.reshape(-1, 1)
    dim = startvals.shape[0]
    num_total = num_samples + num_burn
    samples = np.zeros((dim, num_samples), dtype=float)
    logprob_hist = np.zeros(num_samples, dtype=float)
    start_time = time.time()
    curvals = startvals
    cur_logprob = log_probdens(curvals)
    num_acc = 0
    i = -num_burn
    j = 0
    while i < num_samples:
        j += 1
        candidate = proposal(curvals)
        cand_logprob = log_probdens(candidate)
        log_alpha = cand_logprob - cur_logprob
        log_u = np.log(np.random.uniform())
        if log_u < log_alpha:
            curvals = candidate
            cur_logprob = cand_logprob
            if i >= 0:
                num_acc += 1
        if j >= thin_step:
            if i >= 0:
                samples[:, i:i+1] = curvals
                logprob_hist[i] = cur_logprob
            j = 0
            i += 1
            print(f'Obtained sample number {i}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    accept_rate = num_acc / (num_samples*thin_step)
    result = {
        'samples': samples,
        'accept_rate': accept_rate,
        'logprob_hist': logprob_hist,
        'elapsed_time': elapsed_time,
        'num_burn': num_burn,
        'thin_step': thin_step,
        'seed': seed
    }
    return result


def symmetric_mh_worker(con, mh_args, mh_kwargs):
    mh_res = symmetric_mh_algo(*mh_args, **mh_kwargs)
    con.send(mh_res)
    con.close()


def parallel_symmetric_mh_algo(num_workers, startvals, log_probdens, proposal, num_samples,
                               num_burn=0, thin_step=1, seeds=None, print_info=False):
    if seeds is None:
        seeds = np.random.randint(low=0, high=65535, size=num_workers)
    elif len(seeds) != num_workers:
        raise ValueError('number of seed values must equal num_workers')
    mh_args = (startvals, log_probdens, proposal, num_samples)
    mh_kwargs = {'num_burn': num_burn, 'thin_step': thin_step}
    pipe_parents = []
    pipe_children = []
    procs = []
    for i, seed in zip(range(num_workers), seeds):
        mh_kwargs['seed'] = seed
        pipe_parent, pipe_child = Pipe()
        pipe_parents.append(pipe_parent)
        pipe_children.append(pipe_child)
        proc = Process(target=symmetric_mh_worker,
                       args=(pipe_child, mh_args, mh_kwargs))
        proc.start()
        procs.append(proc)

    mh_res_list = []
    for pipe_parent, proc in zip(pipe_parents, procs):
        mh_res_list.append(pipe_parent.recv())
        proc.join()
    return mh_res_list


def compute_acceptance_rate(samples):
    num_samples = samples.shape[1]
    num_reject = np.sum(np.sum(samples[:, 1:] - samples[:, :-1], axis=0) == 0)
    return 1 - num_reject / num_samples


def compute_effective_sample_size(arr):
    n = len(arr)
    acfvec = acf(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acfvec)):
        sums = sums + (n-k)*acfvec[k]/n

    return n/(1+2*sums)


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

    def _exp_pred_diff_jacobian(self, x, S, propx, propx2):
        if not self.__relative_exp_errors:
            return S
        else:
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

    def _prop_logdet_derivative(self, x, S, propx2):
        outer_jac_det = (2/propx2).reshape(-1, 1)
        if self.__target_mask is not None:
            outer_jac_det[self.__target_mask['idcs']] = 0.
        res = S.T @ outer_jac_det
        if self.__source_mask is not None:
            res[self.__source_mask['idcs']] = 0.
        return res.T

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
        z1[adj] = (-z1r)
        # gradient of likelihood contribution
        m = self.__mapping
        ef = self.__expfact
        propx = m.propagate(x.flatten()).reshape(-1, 1)
        S = m.jacobian(x.flatten())
        if not self.__relative_exp_errors:
            d2 = self.__expvals - propx
            z2 = S.T @ ef(d2)
        else:
            propx2 = self._get_propx2(x)
            d2 = self._get_d2(propx, propx2)
            inv_expcov_times_d2 = ef(d2)
            d2deriv = self._exp_pred_diff_jacobian(x, S, propx, propx2)
            z2 = ((-1) * (inv_expcov_times_d2.T @ d2deriv)).T
            z2 -= 0.5 * self._prop_logdet_derivative(x, S, propx2).T

        z2[nonadj] = 0.
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
        propx2 = self._get_propx2(xref)
        S = self.__mapping.jacobian(xref).tocsc()[:, self.__adj]
        # calculate the inverse posterior covariance matrix
        if not self.__relative_exp_errors:
            S2 = S
        else:
            S2 = S.T.multiply(propx2).tocsr()
        xref = xref.copy()
        xref[self.__nonadj] = priorvals.flatten()[self.__nonadj]
        inv_post_cov = self._approximate_invpostcov(xref, S2)
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
            yref = m.propagate(xref.flatten()).reshape(-1, 1)
            S = m.jacobian(xref.flatten())
            propx = yref + S @ (x - xref)
            d2 = self.__expvals - propx
            if self.__relative_exp_errors:
                x2 = x.copy()
                apply_mask(x2, self.__source_mask)
                propx2 = yref + S @ (x2 - xref)
                scl = propx2 / self.__expvals
                d2 /= scl
        d2_perm = ef.apply_P(d2)
        z2 = ef.solve_L(d2_perm, use_LDLt_decomposition=False)
        prior_res = np.sum(np.square(z1), axis=0)
        t = pf.D()
        prior_logdet = np.sum(np.log(t[~np.isposinf(t)]))
        prior_res += prior_logdet + np.pi*len(d1)
        like_res = np.sum(np.square(z2), axis=0)
        like_res += ef.logdet() + np.pi*len(d2)
        if self.__relative_exp_errors:
            like_res += 2*np.sum(np.log(np.abs(scl)))
        res = -0.5 * (prior_res + like_res)
        if self.__apply_squeeze:
            res = np.squeeze(res)
        return res

    def _approximate_invpostcov(self, xref, S=None):
        xref = xref.flatten()
        pf = self.__priorfact
        ef = self.__expfact
        if S is None:
            S = self.__mapping.jacobian(xref).tocsc()[:, self.__adj]
        invpostcov = (S.T @ ef(S.tocsc()) + pf.inv()).tocsc()
        return invpostcov


def gmap_mh_inference(datatable, covmat, num_samples, prop_scaling,
                      startvals=None, num_burn=0, thin_step=1, int_rtol=1e-4,
                      relative_exp_errors=False, num_workers=1):
    if not np.all(datatable.index == np.sort(datatable.index)):
        raise IndexError('index of datatable must be sorted')
    # prepare the relevant objects, e.g., prior values and prior covariance matrix
    dt = datatable.sort_index()
    exp_sel = dt['NODE'].str.match('exp_', na=False)
    exp_idcs = dt[exp_sel].index
    prior_idcs = dt[~exp_sel].index
    priorvals = dt.loc[prior_idcs, 'PRIOR'].to_numpy()
    priorcov = covmat.tocsr()[prior_idcs,:].tocsc()[:,prior_idcs]
    expvals = dt.loc[exp_idcs, 'DATA'].to_numpy()
    expcov = covmat.tocsr()[exp_idcs,:].tocsc()[:,exp_idcs]
    if np.any(expcov.diagonal() <= 0):
        raise ValueError('observed data must have non-zero uncertainty')
    if startvals is None:
        print('Determine initial values for MCMC chain...')
        mapping = CompoundMap(datatable, rtol=int_rtol, reduce=False)
        lmres = lm_update(mapping, datatable, covmat, print_status=True) 
        startvals = np.empty(len(datatable), dtype=float)
        startvals[prior_idcs] = priorvals
        startvals[lmres['idcs']] = lmres['upd_vals']
        startvals = startvals[prior_idcs]
    else:
        if startvals.size != len(datatable):
            raise IndexError('startvals must be of same length as datatable')
        startvals = startvals[prior_idcs]
    # intialize objects to sample and to obtain values from log posterior pdf
    mapping = CompoundMap(datatable, rtol=int_rtol, reduce=True)
    post = Posterior(priorvals, priorcov, mapping, expvals, expcov,
                     relative_exp_errors=relative_exp_errors)
    propfun = post.generate_proposal_fun(startvals, scale=prop_scaling)

    print('Construct the MCMC chain...')
    mh_args = (startvals, post.logpdf, propfun)
    mh_kwargs = {'num_samples': num_samples, 'num_burn': num_burn,
                 'thin_step': thin_step}
    if num_workers == 1:
        mh_res = symmetric_mh_algo(*mh_args, **mh_kwargs)
        return [mh_res]
    else:
        mh_res_list = parallel_symmetric_mh_algo(num_workers, *mh_args, **mh_kwargs)
        return mh_res_list
