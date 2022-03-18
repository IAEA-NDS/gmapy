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



def fix_cormat(cormat):
    """Fix non positive-definite correlation matrix."""
    if np.any(cormat.diagonal() != 1):
        raise ValueError('All diagonal elements of correlation matrix must be one')
    tries = 0
    cormat = cormat.copy()
    while tries < 15:
        success = True
        try:
            np.linalg.cholesky(cormat)
        except np.linalg.LinAlgError:
            tries += 1
            success = False
            # select all off-diagonal elements
            sel = np.logical_not(np.eye(cormat.shape[0]))
            cormat[sel] /= 1.1
        if success:
            break
    return cormat



def create_dataset_cormat(dataset, uncs):
    """Create correlation matrix of dataset."""
    numpts = len(dataset['CSS'])
    if len(uncs) != numpts:
        raise IndexError('length of uncs must equal length of datapoints')
    # some abbreviations
    MT = dataset['MT']
    EPAF = np.array(dataset['EPAF'])
    E = np.array(dataset['E'])
    CO = np.array(dataset['CO'])
    ENFF = np.array(dataset['ENFF']) if 'ENFF' in dataset else None
    NNCOX = dataset['NNCOX']
    # ad-hoc downscaling of CO if NNCOX flat set
    if NNCOX != 0:
        CO /= 10
    # construct correlation matrix for dataset
    cormat = np.zeros((numpts, numpts), dtype=float)
    for KS in range(numpts):
        C1 = uncs[KS]
        for KT in range(KS):
            Q1 = 0.
            C2 = uncs[KT]
            for L in range(2,11):
                if dataset['NETG'][L] not in (0,9):
                    FKS = EPAF[0,L] + EPAF[1,L]
                    XYY = EPAF[1,L] - (E[KS]-E[KT])/(EPAF[2,L]*E[KS])
                    XYY = max(XYY, 0.)
                    FKT = EPAF[0,L] + XYY
                    Q1 += CO[KS,L]*CO[KT,L]*FKS*FKT

            XNORU = 0.
            if MT not in SHAPE_MT_IDS:
                XNORU = np.sum(np.square(ENFF))

            CERR = (Q1 + XNORU) / (C1*C2)
            CERR = min(CERR, 0.99)

            cormat[KS, KT] = CERR
            cormat[KT, KS] = CERR

        cormat[KS, KS] = 1.

    return cormat



def create_datablock_cormat(datablock, uncs, effuncs=None, shouldfix=True):
    """Create correlation matrix of datablock."""
    if effuncs is None:
        effuncs = uncs.copy()

    dslist = datablock['datasets']

    # get number of points in datablock
    # and build up dictionary with dataset indices
    numpts = 0
    dsdic = {}
    start_ofs_dic = {}
    next_ofs_dic = {}
    for i, ds in enumerate(dslist):
        if len(ds['CSS']) != len(ds['E']):
            raise IndexError('Incompatible E and CSS mesh for dataset %d' % ds['NS'])
        dsdic[ds['NS']] = ds
        start_ofs_dic[ds['NS']] = numpts
        numpts += len(ds['CSS'])
        next_ofs_dic[ds['NS']] = numpts

    # if correlation matrix is explicitly provided for the
    # complete datablock, just return it and ignore
    # any other specifications
    if 'ECOR' in datablock:
        cormat = np.array(datablock['ECOR'])
        if len(cormat.shape) != 2:
            raise IndexError('ECOR must be a matrix')
        if cormat.shape[0] != cormat.shape[1]:
            raise IndexError('ECOR must be a square matrix')
        if shouldfix:
            cormat = fix_cormat(cormat)
        return cormat

    # fill the correlations into the correlation matrix
    cormat = np.zeros((numpts, numpts), dtype=float)
    for ds in dslist:
        dsid = ds['NS']
        start_ofs1 = start_ofs_dic[dsid]
        next_ofs1 = next_ofs_dic[dsid]
        numpts = len(ds['CSS'])
        # calculate the correlations within a dataset
        cormat[start_ofs1:next_ofs1, start_ofs1:next_ofs1] = \
                create_dataset_cormat(ds, uncs[start_ofs1:next_ofs1])
        # only continue for current dataset if cross-correlation
        # to other datasets are provided
        if 'NCSST' not in ds:
            continue
        # some abbreviations
        # we also convert the nested lists in JSON to numpy arrays
        MT = ds['MT']
        NEC = np.array(ds['NEC'])
        FCFC = np.array(ds['FCFC'])
        NETG = np.array(ds['NETG'])
        EPAF = np.array(ds['EPAF'])
        CO = np.array(ds['CO'])
        NNCOX = ds['NNCOX']
        if NNCOX != 0:
            CO /= 10

        # NOTE: In Fortran GMAP the ENFF array is not read
        #       if a dataset is declared as shape. However,
        #       this routine still accesses the ENFF array
        #       for shape datasets, but they only contain
        #       zeros in this case. The JSON datablocks are
        #       more explicit and don't contain the ENFF array
        #       for shape datasets. Therefore, their presence
        #       must be explicitly verified and the computations
        #       adapted accordingly (by checking for ENFF is NONE)
        #       later.
        ENFF = np.array(ds['ENFF']) if 'ENFF' in ds else None
        E = np.array(ds['E'])
        # loop over datasets to which correlations exist
        for pos2, dsid2 in enumerate(ds['NCSST']):
            # some error checking
            if not dsid2 in dsdic:
                if start_ofs1 > 0:
                    # NOTE: For the time being, we don't stop the
                    #       computation if datasets specified to
                    #       be correlated are missing in the datablock.
                    #       This behavior is the same in Fortran GMAP.
                    continue
                    raise IndexError('Dataset %d is marked to be correlated ' % dsid +
                            'with dataset %d, but the latter dataset does not ' %dsid2  +
                            'exist in the datablock')
                else:
                    # For instance, dataset 403 is the first dataset in a block
                    # and correlated to dataset 710 and others, which are in another
                    # datablock. However, the Fortran version ignores this,
                    # because it never attempts to calculate cross-dataset correlations
                    # for the first dataset
                    continue
            if start_ofs_dic[dsid2] > start_ofs_dic[dsid]:
                raise IndexError('In dataset %d, the correlations to ' +
                        'dataset %d are given, but the latter dataset ' +
                        'must not appear before the former in the list' %
                        (dsid, dsid2))
            start_ofs2 = start_ofs_dic[dsid2]
            # some abbreviations
            ds2 = dsdic[dsid2]
            numpts2 = len(ds2['CSS'])
            MT2 = ds2['MT']
            NETG2 = np.array(ds2['NETG'])
            EPAF2 = np.array(ds2['EPAF'])
            CO2 = np.array(ds2['CO'])
            NNCOX2 = ds2['NNCOX']
            if NNCOX2 != 0:
                CO2 /= 10
            ENFF2 = np.array(ds2['ENFF']) if 'ENFF' in ds2 else None
            E2 = np.array(ds2['E'])

            # We skip if one of the two datasets contains
            # shape data and there is only one measurement
            # point in the other dataset. This seems to be
            # an ad-hoc approach in Fortran GMA to neglect
            # those correlations as they are present in the
            # GMA standards database.
            if ((MT2 in SHAPE_MT_IDS and numpts==1) or
                    MT in SHAPE_MT_IDS and numpts2==1):
                continue

            # The following loops calculate the correlation block between
            # dataset 1 and dataset 2
            for K in range(numpts):
                ofs1 = start_ofs1 + K
                C1 = uncs[ofs1]
                for KK in range(numpts2):
                    ofs2 = start_ofs2 + KK
                    C2 = effuncs[ofs2]
                    Q1 = 0.
                    for KKK in range(10):
                        # uncertainty component NC1 of dataset 1 is correlated
                        # with uncertainty component NC2 of dataset 2
                        NC1 = NEC[0, KKK, pos2]
                        NC2 = NEC[1, KKK, pos2]
                        if NC1 > 21 or NC2 > 21:
                            continue
                        if NC1 == 0 or NC2 == 0:
                            break
                        # for 0-based Python indexing
                        pNC1 = NC1 - 1
                        pNC2 = NC2 - 1

                        AMUFA = FCFC[KKK, pos2]
                        if NC1 > 10:
                            NC1 -= 10
                            pNC1 = NC1 - 1
                            if NETG[pNC1] == 9:
                                FKT = 1.
                            else:
                                FKT = EPAF[0, pNC1] + EPAF[1, pNC1]
                            C11 = FKT * CO[K, pNC1]
                        else:
                            C11 = ENFF[pNC1] if ENFF is not None else 0

                        if NC2 > 10:
                            NC2 -= 10
                            pNC2 = NC2 - 1
                            if NETG2[pNC2] == 9:
                                FKS = 1.
                            else:
                                XYY = (EPAF2[1, pNC2] - np.abs(E[K] - E2[KK]) /
                                        (EPAF2[2,pNC2]*E2[KK]))
                                XYY = max(XYY, 0.)
                                FKS = EPAF2[0, pNC2] + XYY
                            C22 = FKS * CO2[KK, pNC2]
                        else:
                            C22 = ENFF2[pNC2] if ENFF is not None else 0

                        Q1 += AMUFA*C11*C22

                    cormat[ofs1, ofs2] = Q1 / (C1*C2)
                    cormat[ofs2, ofs1] = cormat[ofs1, ofs2]

    if shouldfix:
        cormat = fix_cormat(cormat)
    return cormat



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

