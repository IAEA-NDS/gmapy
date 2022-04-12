import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf



def gls_update(mapping, datatable, expcovmat, retcov=False):
    """Calculate updated values and covariance matrix."""
    # prepare quantities required for update
    priorvals = np.full(len(datatable), 0.)
    priorvals[datatable.index] = datatable['PRIOR']
    refvals = priorvals.copy()

    meas = np.full(len(datatable), 0.)
    meas[datatable.index] = datatable['DATA']

    preds = mapping.propagate(datatable, refvals)
    S = mapping.jacobian(datatable, refvals, ret_mat=True)

    # for the time being mask out the fisdata block
    isfis = np.full(len(datatable), False)
    isfis[datatable.index] = datatable['NODE'] == 'fis'
    not_isfis = np.logical_not(isfis)

    isresp = np.empty(len(datatable), dtype=float)
    isresp[datatable.index] = mapping.is_responsible(datatable)
    not_isresp = np.logical_not(isresp)
    has_zerounc = expcovmat.diagonal() == 0.
    not_has_zerounc = np.logical_not(has_zerounc)
    is_indep = np.logical_and(not_isresp, not_isfis)
    is_indep = np.logical_and(is_indep, has_zerounc)
    is_dep = np.logical_and(isresp, not_has_zerounc)

    # reduce the matrices for the GLS solve
    priorvals = priorvals[is_indep]
    meas = meas[is_dep]
    preds = preds[is_dep]
    S = S[is_dep,:].tocsc()
    S = S[:,is_indep]
    expcovmat = expcovmat[is_dep,:].tocsc()
    expcovmat = expcovmat[:,is_dep]

    # perform the update
    inv_post_cov = S.T @ spsolve(expcovmat, S)
    postvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(expcovmat, meas-preds)))

    post_covmat = None
    if retcov is True:
        # following is equivalent to:
        # post_covmat = np.linalg.inv(inv_post_cov.toarray())
        cholfact, _ = dpotrf(inv_post_cov.toarray(), False, False)
        invres , info = dpotri(cholfact)
        if info != 0:
            raise ValueError('Experimental covariance matrix not positive definite')
        post_covmat = np.triu(invres) + np.triu(invres, k=1).T

    return {'upd_vals': postvals, 'upd_covmat': post_covmat,
            'idcs': np.sort(datatable.index[is_indep])}

