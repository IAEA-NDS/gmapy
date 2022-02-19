import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from data_extraction_functions import (extract_prior_values,
        extract_predictions, extract_measurements,
        extract_sensitivity_matrix, extract_covariance_matrix,
        extract_prior_table, extract_experimental_table)

from mappings.basic_maps import get_sensmat_exact, propagate_exact
from mappings.cross_section_map import CrossSectionMap
from mappings.cross_section_shape_map import CrossSectionShapeMap



def new_get_sensitivity_matrix(priortable, exptable):
    # locate all datasets with MT:1 in exptable
    # loop over the various R1 subsets
        # locate the reaction in priortable
            # perform the interpolation (at the moment just lookup)

    idcs1 = np.empty(0, dtype=int)
    idcs2 = np.empty(0, dtype=int)
    coeff = np.empty(0, dtype=float)
    concat = np.concatenate

    # deal with 'cross section' type (MT:1)
    xs_map = CrossSectionMap()
    resp = xs_map.is_responsible(exptable)
    if np.any(resp):
        exptable_red = exptable[resp]
        Sdic = xs_map.jacobian(priortable, exptable_red)
        # add to global arrays
        idcs1 = concat([idcs1, Sdic['idcs1']])
        idcs2 = concat([idcs2, Sdic['idcs2']])
        coeff = concat([coeff, Sdic['x']])

    # deal with 'cross section shape' type (MT:2)
    xsratio_map = CrossSectionShapeMap()
    resp = xsratio_map.is_responsible(exptable)
    if np.any(resp):
        exptable_red = exptable[resp]
        Sdic = xsratio_map.jacobian(priortable, exptable_red)
        # add to global arrays
        idcs1 = concat([idcs1, Sdic['idcs1']])
        idcs2 = concat([idcs2, Sdic['idcs2']])
        coeff = concat([coeff, Sdic['x']])

    # construct the sparse matrix
    S = csr_matrix((coeff, (idcs2, idcs1)),
            shape=(len(exptable.index),
                   len(priortable.index)))
    return S



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
    preds = extract_predictions(datablock_list)
    meas = extract_measurements(datablock_list)
    covmat = extract_covariance_matrix(datablock_list)
    # provisionary code during transition
    priortable = extract_prior_table(APR)
    exptable = extract_experimental_table(datablock_list)
    Sold = extract_sensitivity_matrix(datablock_list, APR)
    Snew = new_get_sensitivity_matrix(priortable, exptable)
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

