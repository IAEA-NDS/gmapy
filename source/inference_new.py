import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, block_diag
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.linalg.lapack import dpotri, dpotrf

from data_management import SIZE_LIMITS
from gmap_snippets import get_prior_range



def extract_prior_energies(APR):
    energylist = []
    for K in range(1, APR.NR+1):
        curval = APR.EN[K]
        energylist.append(curval)
    for K in range(APR.NSHP):
        # normalization factors do not have
        # an associated energy
        energylist.append(0)
    return energylist



def extract_prior_values(APR):
    num_prior_vars = APR.NR + APR.NSHP
    ret = APR.CS[1:(num_prior_vars+1)].copy()
    return ret



def extract_prior_ids(APR):
    idlist = []
    for xsid in range(1, APR.NC+1):
        start, end = get_prior_range(xsid, APR)
        for idx in range(start, end+1):
            cur_id = 'xsid_' + str(xsid)
            idlist.append(cur_id)
    for i in range(1, APR.NSHP+1):
        cur_id = 'norm_' + str(APR.NSETN[i])
        idlist.append(cur_id)
    return idlist



def extract_prior_table(APR):
    dt = pd.DataFrame.from_dict({
        'NODE': extract_prior_ids(APR),
        'PRIOR': extract_prior_values(APR),
        'ENERGY': extract_prior_energies(APR)
        })
    return dt



def extract_predictions(datablock_list):
    predictions = np.zeros(SIZE_LIMITS.MAX_NUM_MEASUREMENTS)
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        predictions[cur_start_idx:next_start_idx] = \
                datablock.predCSS[1:(num_points+1)]
        cur_start_idx = next_start_idx

    predictions = predictions[:next_start_idx].copy()
    return predictions



def extract_experimental_energies(datablock_list):
    energies = np.zeros(SIZE_LIMITS.MAX_NUM_MEASUREMENTS)
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        energies[cur_start_idx:next_start_idx] = \
                datablock.E[1:(num_points+1)]
        cur_start_idx = next_start_idx

    energies = energies[:next_start_idx].copy()
    return energies



def extract_measurements(datablock_list):
    measurements = np.zeros(SIZE_LIMITS.MAX_NUM_MEASUREMENTS)
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        measurements[cur_start_idx:next_start_idx] = \
                datablock.CSS[1:(num_points+1)]
        cur_start_idx = next_start_idx

    measurements = measurements[:next_start_idx].copy()
    return measurements



def extract_effDCS_values(datablock_list):
    effDCS_values = np.zeros(SIZE_LIMITS.MAX_NUM_MEASUREMENTS)
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        effDCS_values[cur_start_idx:next_start_idx] = \
                datablock.effDCS[1:(num_points+1)]
        cur_start_idx = next_start_idx

    effDCS_values = effDCS_values[:next_start_idx].copy()
    return effDCS_values



def extract_sensitivity_matrix(datablock_list, APR):
    max_elnum = SIZE_LIMITS.MAX_NUM_MEASUREMENTS*10
    row_idx = np.full(max_elnum, -1)
    col_idx = np.full(max_elnum, -1)
    vals = np.zeros(max_elnum)

    num_els = 0
    num_prior_vars = APR.NR + APR.NSHP 
    num_pred_vars = 0
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        num_pred_vars += num_points
        KA = datablock.KA
        AA = datablock.AA
        effDCS = datablock.effDCS
        CSS = datablock.CSS
        predCSS = datablock.predCSS
        for i in range(num_prior_vars): 
            for j in range(KA[i+1, 1]):
                col_idx[num_els] = i
                cur_row_idx = KA[i+1, j+2] - 1
                row_idx[num_els] = cur_start_idx + cur_row_idx
                vals[num_els] = AA[i+1, j+1]  
                # values in sensitivity matrix should be for the original prior variables x,
                # which are linked to transformed ones x' by x' = (x-xref)/xref
                vals[num_els] /= APR.CS[i+1]
                # and we also want the sensitivity matrix with respect to the original
                # measurements, not the transformed ones
                vals[num_els] *= (effDCS[cur_row_idx+1]*0.01)*CSS[cur_row_idx+1]
                num_els += 1
        cur_start_idx += num_points

    sens_mat = csr_matrix((vals[:num_els], (row_idx[:num_els], col_idx[:num_els])),
            shape = (num_pred_vars, num_prior_vars))  

    return sens_mat



def extract_covariance_matrix(datablock_list):
    max_elnum = SIZE_LIMITS.MAX_NUM_CORELEMS
    row_idx = np.full(max_elnum, -1)
    col_idx = np.full(max_elnum, -1)
    vals = np.zeros(max_elnum)

    mat_list = []
    for datablock in datablock_list:
        effDCS = datablock.effDCS
        CSS = datablock.CSS
        effECOR = datablock.effECOR
        num_pts = datablock.num_datapoints
        varvec = effDCS[1:(num_pts+1)] * 0.01 * CSS[1:(num_pts+1)]  
        varvec = varvec[:, np.newaxis]
        curcovmat = effECOR[1:(num_pts+1), 1:(num_pts+1)] * np.matmul(varvec, varvec.T) 
        mat_list.append(curcovmat)

    covmat = block_diag(mat_list, format='csr')
    return covmat



def new_gls_update(datablock_list, APR, retcov=False): 
    priorvals = extract_prior_values(APR)
    preds = extract_predictions(datablock_list)
    meas = extract_measurements(datablock_list)
    S = extract_sensitivity_matrix(datablock_list, APR)
    covmat = extract_covariance_matrix(datablock_list)

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

