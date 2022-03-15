import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf

from mappings.compound_map import CompoundMap
from mappings.helperfuns import SHAPE_MT_IDS
from collections import OrderedDict



def create_priortable(prior_list):
    curid = 0
    df = []
    for item in prior_list:
        if item['type'] == 'legacy-prior-cross-section':
            curid += 1
            xsid = item['ID']
            # error checking
            if curid != xsid:
                raise IndexError('prior items must be sorted according to ID in prior_list ' +
                        'but prior item with ID %d violates this constraint' % xsid)
            if not np.all(np.sort(item['EN']) == item['EN']):
                raise ValueError('Energies of prior mesh must be sorted, but this is ' +
                        'not the case for prior block with ID %d' % xsid)
            if not len(item['EN']) == len(item['CS']):
                raise IndexError('Energy mesh and cross sections are of unequal length ' +
                        'for prior block with ID %d' % xsid)

            # append to the dataframe
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'xsid_' + str(xsid)
            prd['REAC'] = 'MT:1-R1:' + str(xsid)
            prd['ENERGY'] = item['EN']
            prd['PRIOR'] = item['CS']
            prd['DESCR'] = item['CLAB'].strip()
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)

        elif item['type'] == 'legacy-fission-spectrum':
            # error checking
            if not np.all(np.sort(item['ENFIS']) == item['ENFIS']):
                raise ValueError('Energies of prior mesh of fission spectrum must be sorted ' +
                        'but this is not the case for the the legacy-fission-spectrum')
            if not len(item['ENFIS']) == len(item['FIS']):
                raise IndexError('Energy mesh and fission spectrum values must be of same length')
            # append to the dataframe
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'fis'
            prd['REAC'] = 'NA'
            prd['ENERGY'] = item['ENFIS']
            prd['PRIOR'] = item['FIS']
            prd['DESCR'] = 'fission spectrum'
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)

        else:
            raise ValueError('Unknown type "%s" of prior block' % item['type'])

    df = pd.concat(df)
    return df



def create_experiment_table(datablock_list):
    """Extract experiment dataframe from datablock list."""
    df_list = []
    for dbidx, db in enumerate(datablock_list):
        if db['type'] != 'legacy-experiment-datablock':
            raise ValueError('Datablock must be of type "legacy-experiment-datablock"')
        for dsidx, ds in enumerate(db['datasets']):
            curdf = pd.DataFrame.from_dict({
                'NODE': 'exp_' + str(ds['NS']),
                'REAC': 'MT:' + str(ds['MT']) +
                        ''.join(['-R%d:%d'%(i+1,r) for i,r in enumerate(ds['NT'])]),
                'ENERGY': ds['E'],
                'DATA': ds['CSS'],
                'DB_IDX': dbidx,
                'DS_IDX': dsidx
            })
            df_list.append(curdf)

    expdf = pd.concat(df_list, ignore_index=True)
    cols = ['NODE', 'REAC', 'ENERGY', 'DATA', 'DB_IDX', 'DS_IDX']
    expdf = expdf.reindex(columns = cols)
    return expdf



def compute_DCS_vector(datablock_list):
    DCS_list = []
    for datablock in datablock_list:
        dataset_list = datablock['datasets']
        for dataset in dataset_list:
            XNORU = 0.
            if dataset['MT'] not in SHAPE_MT_IDS:
                # calculate total normalization uncertainty squared
                XNORU = np.sum(np.square(dataset['ENFF']))

            effCO = np.array(dataset['CO'])
            # Axton special: downweight if NNCOX flag set
            if dataset['NNCOX'] != 0:
                effCO /= 10

            # calculate total uncertainty
            # NOTE: The last element of effCO is ignored!
            RELU = np.sum(np.square(effCO[:,2:11]), axis=1)
            curDCS = np.sqrt(XNORU + RELU)
            DCS_list.append(curDCS)

    DCS = np.concatenate(DCS_list)
    return DCS



def new_gls_update(priortable, exptable, expcovmat, retcov=False):
    """Calculate updated values and covariance matrix."""
    # prepare quantities required for update
    priorvals = priortable['PRIOR'].to_numpy()
    refvals = priorvals.copy()

    meas = exptable['DATA'].to_numpy()

    comp_map = CompoundMap()
    preds = comp_map.propagate(priortable, exptable, refvals)
    S = comp_map.jacobian(priortable, exptable, refvals, ret_mat=True)

    # for the time being mask out the fisdata block
    isfis = priortable['NODE'] == 'fis'
    not_isfis = np.logical_not(isfis)
    priorvals = priorvals[not_isfis]
    S = S[:,not_isfis].copy()

    # perform the update
    inv_post_cov = S.T @ spsolve(expcovmat, S)
    upd_priorvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(expcovmat, meas-preds)))

    # introduce the unmodified fission spectrum in the posterior
    ext_upd_priorvals = refvals.copy()
    ext_upd_priorvals[not_isfis] = upd_priorvals

    post_covmat = None
    if retcov is True:
        # following is equivalent to:
        # post_covmat = np.linalg.inv(inv_post_cov.toarray())
        cholfact, _ = dpotrf(inv_post_cov.toarray(), False, False)
        invres , info = dpotri(cholfact)
        if info != 0:
            raise ValueError('Experimental covariance matrix not positive definite')

        # extend posterior covariance matrix with fission block
        ext_post_covmat = np.full((len(priortable), len(priortable)), 0., dtype=float)
        post_covmat = np.triu(invres) + np.triu(invres, k=1).T
        ext_post_covmat[np.ix_(not_isfis, not_isfis)] = post_covmat

    return {'upd_vals': ext_upd_priorvals, 'upd_covmat': ext_post_covmat}

