import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from data_extraction_functions import (extract_prior_values,
        extract_predictions, extract_measurements,
        extract_sensitivity_matrix, extract_covariance_matrix,
        extract_prior_table, extract_experimental_table)

from basic_maps import get_sensmat_exact, propagate_exact



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
    expmask = exptable['REAC'].str.match('MT:1')
    if expmask.any():
        priormask = priortable['REAC'].str.match('MT:1')
        reacs = exptable[expmask]['REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == curreac]
            exptable_red = exptable[exptable['REAC'] == curreac]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            vals1 = priortable_red['PRIOR']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index
            # calculate the sensitivity matrix
            Sdic = get_sensmat_exact(ens1, ens2)
            # obtain the indices associated with
            # the full prior and experimental table
            curidcs1 = idcs2red[Sdic['i']]
            curidcs2 = idcs1red[Sdic['j']]
            curcoeff = Sdic['x']
            # add to global arrays
            idcs1 = concat([idcs1, curidcs1])
            idcs2 = concat([idcs2, curidcs2])
            coeff = concat([coeff, curcoeff])

    # deal with 'cross section shape' type (MT:2)
    expmask = exptable['REAC'].str.match('MT:2')
    if expmask.any():
        priormask = priortable['REAC'].str.match('MT:2')
        reacs = exptable[expmask]['REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == \
                    curreac.replace('MT:2','MT:1')]
            exptable_red = exptable[exptable['REAC'] == curreac]
            ens1 = priortable_red['ENERGY']
            vals1 = priortable_red['PRIOR']
            idcs1red = priortable_red.index
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'] == dataset_id]
                # get the respective normalization factor from prior
                mask = priortable['NODE'] == dataset_id.replace('exp_', 'norm_')
                norm_index = priortable[mask].index
                norm_fact = np.asscalar(priortable.loc[norm_index, 'PRIOR'])
                if (len(norm_index) != 1):
                    raise IndexError('More than one normalization in prior for dataset ' + str(dataset_id))
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index
                # calculate the sensitivity matrix
                Sdic = get_sensmat_exact(ens1, ens2)
                # obtain the indices associated with
                # the full prior and experimental table
                curidcs1 = idcs2red[Sdic['i']]
                curidcs2 = idcs1red[Sdic['j']]
                curcoeff = np.array(Sdic['x']) * norm_fact
                # add the sensitivity to normalization factor in prior
                numel = len(curidcs1)
                propvals = propagate_exact(ens1, vals1, ens2)
                curidcs1 = concat([curidcs1, curidcs1])
                curidcs2 = concat([curidcs2, np.full(numel, norm_index)])
                curcoeff = concat([curcoeff, propvals])
                if len(curidcs1) != len(curidcs2):
                    raise ValueError
                if len(curidcs1) != len(curcoeff):
                    raise ValueError
                # add to global arrays
                idcs1 = concat([idcs1, curidcs1])
                idcs2 = concat([idcs2, curidcs2])
                coeff = concat([coeff, curcoeff])

        # construct the sparse matrix
        S = csr_matrix((coeff, (idcs1, idcs2)),
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

