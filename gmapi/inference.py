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
    # NOTE: the second term in in zvals, which is
    # inv_priorcov * (priorvals-refvals) is omitted because in this
    # GLS update the expansion vector priorvals conincides with refvals
    zvals = S.T @ spsolve(obscovmat, meas-preds)
    postvals = priorvals + spsolve(inv_post_cov, zvals)

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
        maxiter=10, atol=1e-8, rtol=1e-5, lmb=1e-6, print_status=False):
    # define the prior vector
    priorvals = np.full(len(datatable), 0.)
    priorvals[datatable.index] = datatable['PRIOR']
    # define the expansion vector pref
    fullrefvals = priorvals.copy() if startvals is None else startvals.copy()
    # define the measurement vector
    meas = np.full(len(datatable), 0.)
    meas[datatable.index] = datatable['DATA']
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

    # prepare experimental covariance matrix
    obscovmat = covmat[isobs,:].tocsc()
    obscovmat = obscovmat[:,isobs]
    obscovmat_fact = cholesky(obscovmat)
    # prepare parameter prior covariance matrix
    priorcovmat = covmat[isadj,:].tocsc()
    priorcovmat = priorcovmat[:,isadj]
    priorcovmat_fact = cholesky(priorcovmat)
    inv_prior_cov = priorcovmat_fact.inv()
    # prepare identity matrix for update damping
    idmat = identity(np.sum(isadj))

    # these quantities remain constant despite
    # throughout the loops below
    priorvals = priorvals[isadj]
    meas = meas[isobs]

    old_postvals = None
    num_iter = 0
    while num_iter < maxiter:
        # termination condition at end of loop
        num_iter += 1
        # get the predictions and Jacobian matrix
        preds = mapping.propagate(datatable, fullrefvals)
        S = mapping.jacobian(datatable, fullrefvals, ret_mat=True)
        # reduce the matrices for the LM solve
        refvals = fullrefvals[isadj]
        preds = preds[isobs]
        S = S[isobs,:].tocsc()
        S = S[:,isadj]

        # GLS update
        inv_post_cov = S.T @ obscovmat_fact(S) + inv_prior_cov + lmb * idmat
        postvals = refvals + spsolve(inv_post_cov, S.T @ (spsolve(obscovmat, meas-preds)))

        # calculate real prediction and expected prediction
        # according to linearization for posterior parameters
        current_propcss = preds
        expected_propcss = preds + S @ (postvals - refvals)
        new_fullrefvals = fullrefvals.copy()
        new_fullrefvals[isadj] = postvals
        real_propcss = mapping.propagate(datatable, new_fullrefvals)
        real_propcss = real_propcss[isobs]
        # calculate negative log likelihoods associated with
        # current parameter set and proposed one according to either
        # the true model (non-linearities considered) and the linearized model
        curpardiff = refvals - priorvals
        newpardiff = postvals - priorvals
        cur_measdiff = meas - current_propcss
        cur_neglogprior = curpardiff.T @ priorcovmat_fact(curpardiff)
        new_neglogprior = newpardiff.T @ priorcovmat_fact(newpardiff)
        exp_measdiff = meas - expected_propcss
        real_measdiff = meas - real_propcss
        cur_negloglike = cur_measdiff.T @ obscovmat_fact(cur_measdiff) + cur_neglogprior
        exp_negloglike = exp_measdiff.T @ obscovmat_fact(exp_measdiff) + new_neglogprior
        real_negloglike = real_measdiff.T @ obscovmat_fact(real_measdiff) + new_neglogprior
        # calculate expected and real improvement and use the ratio
        # as criterion to determine the adjustment of the damping term
        exp_improvement = cur_negloglike - exp_negloglike
        real_improvement = cur_negloglike - real_negloglike
        rho = (exp_improvement+1e-8) / (real_improvement+1e-8)
        if rho > 0.75:
            lmb /= 3
        elif rho < 0.25:
            lmb *= 2

        # only accept new parameter set
        # if the associated log likelihood is larger
        if real_negloglike < cur_negloglike:
            old_postvals = fullrefvals[isadj]
            fullrefvals[isadj] = postvals

        if print_status:
            print('###############')
            print('cur_loglike: ' + str(cur_negloglike))
            print('exp_loglike: ' + str(exp_negloglike))
            print('real_loglike: ' + str(real_negloglike))
            print('exp_improvement: ' + str(exp_improvement))
            print('real_improvement: ' + str(real_improvement))
            print('lambda: ' + str(lmb))

        if old_postvals is not None:
            absdiff = postvals - old_postvals
            if np.all(absdiff < atol + rtol*old_postvals):
                break

    return {'upd_vals': postvals, 'upd_covmat': None,
            'idcs': np.sort(datatable.index[isadj])}

