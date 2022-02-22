import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from data_extraction_functions import (extract_prior_values,
        extract_predictions, extract_measurements,
        extract_sensitivity_matrix, extract_covariance_matrix,
        extract_prior_table, extract_experimental_table)

from mappings.compound_map import CompoundMap



def replace_submatrix(M, R):
    """Replace rows of M by non-zero rows of R."""
    if M.shape != R.shape:
        raise IndexError('M and R must have same shape')
    M = M.copy()
    h = np.split(R.indices, R.indptr)[1:-1]
    rows = np.array([pos for pos, el in enumerate(h) if len(el) > 0])
    cols = np.array(np.sort(np.unique(R.indices)))
    M[rows,:] = R[rows,:]
    return M



def new_gls_update(datablock_list, APR, retcov=False): 
    priorvals = extract_prior_values(APR)
    meas = extract_measurements(datablock_list)
    covmat = extract_covariance_matrix(datablock_list)
    # provisionary code during transition
    priortable = extract_prior_table(APR)
    exptable = extract_experimental_table(datablock_list)

    # create random permutations as a test
    refvals = priorvals
    Sold = extract_sensitivity_matrix(datablock_list, APR)
    preds = extract_predictions(datablock_list)

    comp_map = CompoundMap()
    isresp = comp_map.is_responsible(exptable)

    perm1 = np.random.permutation(priortable.shape[0])
    perm2 = np.random.permutation(exptable.shape[0])
    priortable = priortable.iloc[perm1].copy()
    exptable = exptable.iloc[perm2].copy()

    oldpreds = preds.copy()
    preds[isresp] = comp_map.propagate(priortable, exptable, refvals)[isresp]
    if not np.all(np.isclose(oldpreds, preds, atol=0, rtol=1e-12)):
        raise ValueError('New predictions do not match GMAP ones')

    Snew = comp_map.jacobian(priortable, exptable, refvals, ret_mat=True)
    S = replace_submatrix(Sold, Snew)
    if not np.all(np.isclose(Sold.todense(), S.todense(), atol=0, rtol=1e-12)):
        raise ValueError('New sensitivity elements do not match GMAP ones')

    inv_post_cov = S.T @ spsolve(covmat, S)
    upd_priorvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(covmat, meas-preds)))

    post_covmat = None
    if retcov is True:
        # following is equivalent to:
        # post_covmat = np.linalg.inv(inv_post_cov.toarray())
        cholfact, _ = dpotrf(inv_post_cov.toarray(), False, False)
        invres , info = dpotri(cholfact)
        if info != 0:
            raise ValueError('Experimental covariance matrix not positive definite')
        post_covmat = np.triu(invres) + np.triu(invres, k=1).T

    return {'upd_vals': upd_priorvals, 'upd_covmat': post_covmat}

