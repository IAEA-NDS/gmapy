import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf



def gls_update(mapping, datatable, covmat, retcov=False):
    """Calculate updated values and covariance matrix."""
    # prepare quantities required for update
    priorvals = np.full(len(datatable), 0.)
    priorvals[datatable.index] = datatable['PRIOR']
    refvals = priorvals.copy()

    meas = np.full(len(datatable), 0.)
    meas[datatable.index] = datatable['DATA']

    preds = mapping.propagate(datatable, refvals)
    S = mapping.jacobian(datatable, refvals, ret_mat=True)

    not_isobs = np.isnan(meas)
    isobs = np.logical_not(not_isobs)
    has_zerounc = covmat.diagonal() == 0.
    has_nonzerounc = np.logical_not(has_zerounc)
    isadj = np.logical_and(has_nonzerounc, not_isobs)
    if np.any(np.logical_and(has_zerounc, isobs)):
        raise ValueError('Observed data must have non-zero uncertainty')

    # reduce the matrices for the GLS solve
    priorvals = priorvals[isadj]
    preds = preds[isobs]
    meas = meas[isobs]
    S = S[isobs,:].tocsc()
    S = S[:,isadj]
    obscovmat = covmat[isobs,:].tocsc()
    obscovmat = obscovmat[:,isobs]

    # prepare the inverse prior covariance matrix
    idmat = identity(np.sum(isadj), dtype='d').tocsc()
    priorcovmat = covmat[isadj,:].tocsc()
    priorcovmat = priorcovmat[:,isadj]
    inv_prior_cov = spsolve(priorcovmat, idmat)

    # perform the update
    inv_post_cov = S.T @ spsolve(obscovmat, S) + inv_prior_cov
    postvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(obscovmat, meas-preds)))

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
            'idcs': np.sort(datatable.index[isadj])}

