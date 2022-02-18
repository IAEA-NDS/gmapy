import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf
from data_extraction_functions import (extract_prior_values,
        extract_predictions, extract_measurements,
        extract_sensitivity_matrix, extract_covariance_matrix,
        extract_prior_table, extract_experimental_table)



def new_get_sensitivity_matrix(priortable, exptable):
    # locate all datasets with MT:1 in exptable
    # loop over the various R1 subsets
        # locate the reaction in priortable
            # perform the interpolation (at the moment just lookup)
    expmask = exptable['REAC'].str.match('MT:1')
    if expmask.any():
        priormask = priortable['REAC'].str.match('MT:1')
        reacs = exptable[expmask]['REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == curreac]
            exptable_red = exptable[exptable['REAC'] == curreac]

            def get_sens_mat(ens1, ens2):
                ens1 = np.array(ens1)
                ens2 = np.array(ens2)
                ord = np.argsort(ens1)
                ens1 = ens1[ord]
                ridcs = np.searchsorted(ens1, ens2, side='left')
                if not np.all(ens1[ridcs] == ens2):
                    raise ValueError('mismatching energies encountered' +
                            str(ens1[ridcs]) + ' vs ' + str(ens2))

                idcs1 = np.arange(len(ens2))
                idcs2 = ord[ridcs]
                coeff = np.ones(len(ens2))
                return {'i': idcs1, 'j': idcs2, 'x': coeff}

            def propagate(ens1, vals1, ens2):
                Sraw = get_sens_mat(ens1, ens2)
                S = csr_matrix((Sraw['x'], (Sraw['i'], Sraw['j'])),
                          shape = (len(ens2), len(ens1)))
                return S @ vals1


            ens1 = priortable_red['ENERGY']
            vals1 = priortable_red['PRIOR']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index

            Sdic = get_sens_mat(ens1, ens2)
            idcs1 = idcs2red[Sdic['i']]
            idcs2 = idcs1red[Sdic['j']]
            coeff = Sdic['x']
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

