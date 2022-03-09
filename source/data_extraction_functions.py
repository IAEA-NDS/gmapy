import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, block_diag
from scipy import sparse

from data_management import SIZE_LIMITS
from gmap_snippets import get_prior_range, get_dataset_range


def extract_prior_reacs(APR):
    reaclist = []
    energylist = []
    for xsid in range(1, APR.NC+1):
        start, end = get_prior_range(xsid, APR)
        for idx in range(start, end+1):
            curreac = 'MT:' + str(1) + '-R1:' + str(xsid)
            reaclist.append(curreac)
    if APR.fisdata is not None:
        for K in range(APR.fisdata.NFIS):
            reaclist.append('NA')
    for K in range(APR.NSHP):
        # normalization factors do not have
        # an associated energy
        reaclist.append('NA')
    return reaclist



def extract_prior_energies(APR):
    energylist = []
    for K in range(1, APR.NR+1):
        curval = APR.EN[K]
        energylist.append(curval)
    if APR.fisdata is not None:
        for K in range(1, APR.fisdata.NFIS+1):
            energylist.append(APR.fisdata.ENFIS[K])
    for K in range(APR.NSHP):
        # normalization factors do not have
        # an associated energy
        energylist.append(0)
    return energylist



def extract_prior_values(APR):
    num_prior_vars = APR.NR + APR.NSHP
    ret = np.concatenate([
        APR.CS[1:(APR.NR+1)],
        APR.fisdata.FIS[1:(APR.fisdata.NFIS+1)],
        APR.CS[(APR.NR+1):(num_prior_vars+1)]
    ])
    return ret



def extract_prior_ids(APR):
    idlist = []
    for xsid in range(1, APR.NC+1):
        start, end = get_prior_range(xsid, APR)
        for idx in range(start, end+1):
            cur_id = 'xsid_' + str(xsid)
            idlist.append(cur_id)
    if APR.fisdata is not None:
        for K in range(1, APR.fisdata.NFIS+1):
            idlist.append('fis')
    for i in range(1, APR.NSHP+1):
        cur_id = 'norm_' + str(APR.NSETN[i])
        idlist.append(cur_id)
    return idlist



def extract_prior_table(APR):
    dt = pd.DataFrame.from_dict({
        'NODE': extract_prior_ids(APR),
        'REAC': extract_prior_reacs(APR),
        'PRIOR': extract_prior_values(APR),
        'ENERGY': extract_prior_energies(APR)
        })
    return dt



def extract_experimental_ids(datablock_list):
    idlist = []
    cur_start_idx = 0
    for datablock in datablock_list:
        for dsidx in range(1, datablock.num_datasets+1):
            dataset_id = datablock.IDEN[dsidx,6]
            sidx, fidx = get_dataset_range(dsidx, datablock)
            for k in range(sidx, fidx+1):
                curid = 'exp_' + str(dataset_id)
                idlist.append(curid)
    return idlist



def extract_experimental_reacs(datablock_list):
    reaclist = []
    cur_start_idx = 0
    for datablock in datablock_list:
        for dsidx in range(1, datablock.num_datasets+1):
            MT = datablock.IDEN[dsidx,7]
            NCT = datablock.NCT[dsidx]
            NT = datablock.NT[dsidx, 1:(NCT+2)]
            NTstr = ['R' + str(i+1) + ':' + str(NT[i]) for i in range(NCT)]
            sidx, fidx = get_dataset_range(dsidx, datablock)
            for k in range(sidx, fidx+1):
                curreac = 'MT:' + str(MT) + '-' + '-'.join(NTstr)
                reaclist.append(curreac)
    return reaclist



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



def extract_experimental_table(datablock_list):
    dt = pd.DataFrame.from_dict({
        'NODE': extract_experimental_ids(datablock_list),
        'REAC': extract_experimental_reacs(datablock_list),
        'DATA': extract_measurements(datablock_list),
        'ENERGY': extract_experimental_energies(datablock_list)
        })
    return dt



def extract_DCS_values(datablock_list):
    DCS_values = np.zeros(SIZE_LIMITS.MAX_NUM_MEASUREMENTS)
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        DCS_values[cur_start_idx:next_start_idx] = \
                datablock.DCS[1:(num_points+1)]
        cur_start_idx = next_start_idx

    DCS_values = DCS_values[:next_start_idx].copy()
    return DCS_values



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



def update_effDCS_values(datablock_list, effDCS_values):
    cur_start_idx = 0
    for datablock in datablock_list:
        num_points = datablock.num_datapoints
        next_start_idx = cur_start_idx + datablock.num_datapoints
        datablock.effDCS[1:(num_points+1)] = \
            effDCS_values[cur_start_idx:next_start_idx]
        cur_start_idx = next_start_idx



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

