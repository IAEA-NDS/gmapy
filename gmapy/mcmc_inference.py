import numpy as np
from sksparse.cholmod import cholesky
import scipy.sparse as sps 
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from .mappings.compound_map import CompoundMap
from .mappings.priortools import prepare_prior_and_exptable
from .inference import lm_update


def symmetric_mh_algo(startvals, log_probdens, proposal, num_samples,
                      num_burn=0, thin_step=1, print_info=False):
    startvals = startvals.reshape(-1, 1)
    dim = startvals.shape[0]
    num_total = num_samples + num_burn
    samples = np.zeros((dim, num_total), dtype=float)
    logprob_hist = np.zeros(num_total, dtype=float)
    curvals = startvals
    cur_logprob = log_probdens(curvals)
    samples[:, 0:1] = curvals
    logprob_hist[0] = cur_logprob
    for i in range(1, num_total):
        candidate = proposal(curvals)
        cand_logprob = log_probdens(candidate)
        log_alpha = cand_logprob - cur_logprob
        log_u = np.log(np.random.uniform())
        if log_u < log_alpha:
            print('--------> accept')
            curvals = candidate
            cur_logprob = cand_logprob
        else:
            print('reject')
        samples[:, i:i+1] = curvals
        logprob_hist[i] = cur_logprob

    # remove burnin samples and thin the chain
    samples = samples[:, num_burn::thin_step]
    logprob_hist = logprob_hist[num_burn::thin_step]

    accept_rate = compute_acceptance_rate(samples)
    print(f'acceptance rate: {accept_rate}')
    result = {
        'samples': samples,
        'accept_rate': accept_rate,
        'logprob_hist': logprob_hist
    }
    return result


def compute_acceptance_rate(samples):
    num_samples = samples.shape[1]
    num_reject = np.sum(np.sum(samples[:, 1:] - samples[:, :-1], axis=0) == 0)
    return 1 - num_reject / num_samples


class Posterior:

    def __init__(self, priorvals, priorcov, mapping, expvals, expcov):
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
        self.__apply_squeeze = True

    def set_squeeze(self, flag):
        self.__apply_squeeze = flag

    def logpdf(self, x):
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
        propx = np.hstack([m.propagate(x[:,i]).reshape(-1,1)
                          for i in range(x.shape[1])])
        d2 = self.__expvals - propx
        d2_perm = ef.apply_P(d2)
        z2 = ef.solve_L(d2_perm, use_LDLt_decomposition=False)
        prior_res = np.sum(np.square(z1), axis=0)
        like_res = np.sum(np.square(z2), axis=0)
        res = -0.5 * (prior_res + like_res)
        if self.__apply_squeeze:
            res = np.squeeze(res)
        return res

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
        propx = m.propagate(x).reshape(-1, 1)
        d2 = self.__expvals - propx
        S = m.jacobian(x)
        z2 = S.T @ ef(d2)
        z2[nonadj] = 0.
        res = z1 + z2
        if self.__apply_squeeze:
            res = np.squeeze(z1+z2)
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

    def approximate_covmat(self, xref):
        xref = xref.flatten()
        adj = self.__adj
        adjidcs = self.__adj_idcs
        size = self.__size
        S = self.__mapping.jacobian(xref).tocsc()[:, self.__adj]
        pf = self.__priorfact
        ef = self.__expfact
        tmp = coo_matrix(sps.linalg.inv(
            (S.T @ ef(S.tocsc()) + pf.inv()).tocsc())
        )
        postcov = csr_matrix((tmp.data, (adjidcs[tmp.row], adjidcs[tmp.col])),
                             dtype=float, shape=(size, size))
        return postcov


def gmap_mh_inference(datatable, covmat, num_samples, prop_scaling,
                      startvals=None, num_burn=0, thin_step=1):
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
        mapping = CompoundMap(datatable, rtol=1e-4, reduce=False)
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
    mapping = CompoundMap(datatable, rtol=1e-4, reduce=True)
    post = Posterior(priorvals, priorcov, mapping, expvals, expcov)
    propfun = post.generate_proposal_fun(startvals, scale=prop_scaling)

    print('Construct the MCMC chain...')
    mh_res = symmetric_mh_algo(startvals, post.logpdf, propfun,
                               num_samples=num_samples, num_burn=num_burn,
                               thin_step=thin_step, print_info=True)
    return mh_res
