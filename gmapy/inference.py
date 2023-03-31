import numpy as np
import pandas as pd
from scipy.sparse import identity, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from sksparse.cholmod import cholesky
import warnings
from .mappings.priortools import propagate_mesh_css
from .data_management.uncfuns import scale_covmat


def gls_update(mapping, datatable, covmat, retcov=False):
    """Calculate updated values and covariance matrix."""
    # prepare quantities required for update
    priorvals = np.full(len(datatable), 0.)
    priorvals[datatable.index] = datatable['PRIOR']
    refvals = priorvals.copy()

    meas = np.full(len(datatable), 0.)
    meas[datatable.index] = datatable['DATA']

    preds = mapping.propagate(refvals, datatable)
    S = mapping.jacobian(refvals, datatable)

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
        maxiter=10, atol=1e-6, rtol=1e-6, lmb=1e-6, print_status=False,
        correct_ppp=False, ret_invcov=False, must_converge=True,
        no_reject=False):
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
    # restrict the measurement vector to the real measurement points
    meas = meas[isobs]
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
    orig_obscovmat = obscovmat.copy()
    # PPP correction
    if correct_ppp:
        tmp_preds = propagate_mesh_css(datatable, mapping, fullrefvals,
                                        prop_normfact=False, mt6_exp=True,
                                        prop_usu_errors=False)
        obscovmat = scale_covmat(orig_obscovmat, tmp_preds[isobs] / meas)
        obscovmat_fact = cholesky(obscovmat)
    else:
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

    old_postvals = None
    num_iter = 0
    converged = False
    first_cycle = True
    while num_iter < maxiter:
        # termination condition at end of loop
        num_iter += 1
        # get the predictions and Jacobian matrix
        preds = mapping.propagate(fullrefvals, datatable)
        S = mapping.jacobian(fullrefvals, datatable)
        # reduce the matrices for the LM solve
        refvals = fullrefvals[isadj]
        preds = preds[isobs]
        S = S[isobs,:].tocsc()
        S = S[:,isadj]
        # GLS update
        inv_post_cov = S.T @ obscovmat_fact(S) + inv_prior_cov + lmb * idmat
        zvec = S.T @ obscovmat_fact(meas-preds) + priorcovmat_fact(priorvals-refvals)
        postvals = refvals + spsolve(inv_post_cov, zvec)

        # calculate real prediction and expected prediction
        # according to linearization for posterior parameters
        current_propcss = preds
        expected_propcss = preds + S @ (postvals - refvals)
        new_fullrefvals = fullrefvals.copy()
        new_fullrefvals[isadj] = postvals
        real_propcss = mapping.propagate(new_fullrefvals, datatable)
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
        # in later iterations, we can adopt for cur_negloglike
        # one of the values from the previous iterations
        if first_cycle:
            cur_negloglike = cur_measdiff.T @ obscovmat_fact(cur_measdiff) + cur_neglogprior
        exp_negloglike = exp_measdiff.T @ obscovmat_fact(exp_measdiff) + new_neglogprior
        # calculate the PPP corrected matrix
        if correct_ppp:
            tmp_preds = propagate_mesh_css(datatable, mapping, new_fullrefvals,
                                            prop_normfact=False, mt6_exp=True,
                                            prop_usu_errors=False)
            new_obscovmat = scale_covmat(orig_obscovmat, tmp_preds[isobs] / meas)
            new_obscovmat_fact = cholesky(new_obscovmat)
        else:
            new_obscovmat = obscovmat
            new_obscovmat_fact = obscovmat_fact

        real_negloglike = real_measdiff.T @ new_obscovmat_fact(real_measdiff) + new_neglogprior
        # calculate expected and real improvement and use the ratio
        # as criterion to determine the adjustment of the damping term
        exp_improvement = cur_negloglike - exp_negloglike
        real_improvement = cur_negloglike - real_negloglike
        rho = (exp_improvement+1e-8) / (real_improvement+1e-8)
        lmb_used = lmb
        if real_improvement < 0:
            lmb *= 97
        elif rho > 0.75:
            lmb /= 3
        elif rho < 0.25:
            lmb *= 2

        old_negloglike = cur_negloglike
        # only accept new parameter set
        # if the associated log likelihood is larger
        accepted = False
        old_negloglike = cur_negloglike
        if real_improvement > 0 or no_reject:
            old_postvals = fullrefvals[isadj]
            fullrefvals[isadj] = postvals
            cur_negloglike = real_negloglike
            obscovmat = new_obscovmat
            obscovmat_fact = new_obscovmat_fact
            accepted = True

        if old_postvals is not None:
            absdiff = postvals - old_postvals
            reldiff = np.abs(absdiff) / (atol + np.abs(old_postvals))
            maxreldiff = np.max(reldiff)
            meanreldiff = np.mean(reldiff)

        if print_status:
            print('###############')
            print('cur_negloglike: ' + str(old_negloglike))
            print('exp_negloglike: ' + str(exp_negloglike))
            print('real_negloglike: ' + str(real_negloglike))
            print('exp_improvement: ' + str(exp_improvement))
            print('real_improvement: ' + str(real_improvement))
            print('lambda used: ' + str(lmb_used))
            print('rho: ' + str(rho))
            if old_postvals is not None:
                print('maximal relative parameter change: ' + str(maxreldiff))
            print('accepted' if accepted else 'REJECTED!')

        if old_postvals is not None:
            if np.all(absdiff < (atol + rtol*old_postvals)):
                converged = True
                break

        first_cycle = False

    if not converged:
        if must_converge:
            raise ValueError('LM algorithm did not converge!')
        else:
            warnings.warn('LM algorithm did not converge.')

    res = {'upd_vals': postvals, 'upd_covmat': None,
            'idcs': np.sort(datatable.index[isadj]), 'lmb': lmb,
            'last_rejected': (not accepted), 'converged': converged}

    if ret_invcov:
        inv_post_cov = S.T @ obscovmat_fact(S) + inv_prior_cov
        res['upd_invcov'] = inv_post_cov

    return res


def new_lm_update(dist_obj, startvals=None, maxiter=10,
                  atol=1e-6, rtol=1e-6, lmb=1e-6, print_status=False,
                  must_converge=True, no_reject=False):
    if startvals is None:
        startvals = dist_obj.get_priorvals()
    cur_vals = startvals
    cur_loglike = dist_obj.logpdf(cur_vals)
    for num_iter in range(maxiter):
        prop_vals = dist_obj.approximate_postmode(cur_vals, lmb=lmb)
        # calculate expected and real improvement and use the ratio
        # as criterion to determine the adjustment of the damping term
        expected_loglike = dist_obj.approximate_logpdf(cur_vals, prop_vals)
        proposed_loglike = dist_obj.logpdf(prop_vals)
        expected_improvement = expected_loglike - cur_loglike
        real_improvement = proposed_loglike - cur_loglike
        rho = (expected_improvement+1e-8) / (real_improvement+1e-8)
        accepted = real_improvement > 0 or no_reject
        if print_status:
            print('###############')
            print('cur_negloglike: ' + str(cur_loglike))
            print('exp_negloglike: ' + str(expected_loglike))
            print('real_negloglike: ' + str(proposed_loglike))
            print('exp_improvement: ' + str(expected_improvement))
            print('real_improvement: ' + str(real_improvement))
            print('lambda used: ' + str(lmb))
            print('rho: ' + str(rho))
            print('accepted: ' + str(accepted))
        converged = False
        if accepted:
            # auxiliary quantities for convergence criterion
            absdiff = prop_vals - cur_vals
            reldiff = np.abs(absdiff) / (atol + np.abs(cur_vals))
            maxreldiff = np.max(reldiff)
            if print_status:
                print('maximal relative parameter change: ' + str(maxreldiff))
            converged = np.all(absdiff < (atol + rtol*cur_vals))
        # rescale lambda according to improvement
        if real_improvement < 0:
            lmb *= 97
        elif rho > 0.75:
            lmb /= 3.
        elif rho < 0.25:
            lmb *= 2.
        if accepted:
            cur_vals = prop_vals
            cur_loglike = proposed_loglike
        if converged:
            break

    if not converged:
        if must_converge:
            raise ValueError('LM algorithm did not converge!')
        else:
            warnings.warn('LM algorithm did not converge.')

    ret = {
        'upd_vals': cur_vals,
        'lmb': lmb,
        'last_rejected': not accepted,
        'converged': converged
    }
    return ret


def compute_posterior_covmat(mapping, datatable, postvals, invcovmat,
        source_idcs, idcs=None,  unc_only=False, **mapargs):
    # calculate the refvals
    has_prior = np.logical_not(datatable.PRIOR.isna())
    refvals = np.zeros(len(datatable))
    refvals[has_prior] = datatable.loc[has_prior, 'PRIOR'].to_numpy()
    refvals[source_idcs] = postvals
    # calculate and trim sensitivity matrix
    S = mapping.jacobian(refvals, datatable, **mapargs)
    if idcs is not None:
        S = S[idcs,:]
    if source_idcs is not None:
        S = S[:, source_idcs]
    S = S.tocsr()
    invcovmat = invcovmat.tocsc(copy=True)

    cov_times_St = spsolve(invcovmat, S.T)
    if unc_only:
        uncs = np.sqrt(np.sum(S.multiply(cov_times_St.T), axis=1))
        return np.ravel(uncs)
    else:
        postcov = S @ cov_times_St
        return postcov
