import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from sksparse.cholmod import cholesky



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



def lm_update(mapping, datatable, covmat, retcov=False, startvals=None,
        maxiter=10, atol=1e-8, rtol=1e-5):
    # define the prior vector
    priorvals = np.full(len(datatable), 0.)
    priorvals[datatable.index] = datatable['PRIOR']
    # define the expansion vector pref
    refvals = priorvals if startvals is None else startvals
    # define the measurement vector
    meas = np.full(len(datatable), 0.)
    meas[datatable.index] = datatable['DATA']
    # get the prediction and mapping matrix
    preds = mapping.propagate(datatable, refvals)
    S = mapping.jacobian(datatable, refvals, ret_mat=True)
    # NaN (=not a number) values indicate that quantity was not measured
    not_isobs = np.isnan(meas)
    isobs = np.logical_not(not_isobs)
    # quantities with zero uncertainty are fixed,
    # hence will be propagated but not adjusted
    has_zerounc = covmat.diagonal() == 0.
    has_nonzerounc = np.logical_not(has_zerounc)
    # not observed quantities with uncertainty are adjustable
    isadj = np.logical_and(has_nonzerounc, not_isobs)
    # observed data must not have zero uncertainites
    # (a measurement without uncertainty is no measurement)
    if np.any(np.logical_and(has_zerounc, isobs)):
        raise ValueError('observed data must have non-zero uncertainty')

    # reduce the matrices for the LM solve
    priorvals = priorvals[isadj]
    preds = preds[isobs]
    meas = meas[isobs]
    S = S[isobs,:].tocsc()
    S = S[:,isadj]
    # prepare experimental covariance matrix
    obscovmat = covmat[isobs,:].tocsc()
    obscovmat = obscovmat[:,isobs]
    obscovmat_fact = cholesky(obscovmat)
    # prepare parameter prior covariance matrix
    priorcovmat = covmat[isadj,:].tocsc()
    priorcovmat = priorcovmat[:,isadj]
    priorcovmat_fact = cholesky(priorcovmat)
    inv_prior_cov = priorcovmat_fact.inv()
    # factorize matrices for faster computation
    inv_post_cov = S.T @ obscovmat_fact(S) + inv_prior_cov
    postvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(obscovmat, meas-preds)))
    return {'upd_vals': postvals, 'upd_covmat': None,
            'idcs': np.sort(datatable.index[isadj])}

