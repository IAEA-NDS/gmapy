import numpy as np
from scipy.sparse import block_diag, csr_matrix
from collections import OrderedDict

from ..mappings.priortools import SHAPE_MT_IDS



def scale_covmat(covmat, sclvec):
    newcovmat = covmat * sclvec.reshape(1,-1) * sclvec.reshape(-1,1)
    return newcovmat



def cov2cor(covmat):
    invuncs = 1. / np.sqrt(covmat.diagonal())
    cormat = scale_covmat(covmat, invuncs)
    np.fill_diagonal(cormat, 1.)
    return cormat



def cor2cov(cormat, uncs):
    covmat = scale_covmat(cormat, uncs)
    return covmat



# this function is here to reproduce
# a bug in GMAP Fortran in the PPP correction
def relcov_to_wrong_cor(relcovmat, uncs, effuncs, datasets):
    cormat = np.copy(relcovmat)
    # get number of points in datablock
    # and build up dictionary with dataset indices
    numpts = 0
    dsdic = {}
    start_ofs_dic = {}
    next_ofs_dic = {}
    for i, ds in enumerate(datasets):
        if len(ds['CSS']) != len(ds['E']):
            raise IndexError('Incompatible E and CSS mesh for dataset %d' % ds['NS'])
        dsdic[ds['NS']] = ds
        start_ofs_dic[ds['NS']] = numpts
        numpts += len(ds['CSS'])
        next_ofs_dic[ds['NS']] = numpts
    # start treating the correlations
    for ds in datasets:
        dsid = ds['NS']
        start_ofs1 = start_ofs_dic[dsid]
        next_ofs1 = next_ofs_dic[dsid]
        numpts = len(ds['CSS'])
        # calculate the correlations within a dataset
        cormat[start_ofs1:next_ofs1, start_ofs1:next_ofs1] = \
                cov2cor(cormat[start_ofs1:next_ofs1, start_ofs1:next_ofs1])
        # only continue for current dataset if cross-correlation
        # to other datasets are provided
        if 'NCSST' not in ds:
            continue
        # some abbreviations
        # we also convert the nested lists in JSON to numpy arrays
        MT = ds['MT']

        # loop over datasets to which correlations exist
        for pos2, dsid2 in enumerate(set(ds['NCSST'])):
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

            # We skip if one of the two datasets contains
            # shape data and there is only one measurement
            # point in the other dataset. This seems to be
            # an ad-hoc approach in Fortran GMA to neglect
            # those correlations as they are present in the
            # GMA standards database.
            if ((MT2 in SHAPE_MT_IDS and numpts==1) or
                    MT in SHAPE_MT_IDS and numpts2==1):
                continue

            for K in range(numpts):
                ofs1 = start_ofs1 + K
                C1 = uncs[ofs1]
                for KK in range(numpts2):
                    ofs2 = start_ofs2 + KK
                    C2 = effuncs[ofs2]
                    cormat[ofs1, ofs2] /= (C1*C2)
                    cormat[ofs2, ofs1] = cormat[ofs1, ofs2]

                cormat[ofs1, ofs1] = 1.
    # symmetrize the matrix
    cormat[np.triu_indices_from(cormat,k=1)] = \
            cormat.T[np.triu_indices_from(cormat, k=1)]
    assert np.all(cormat.diagonal() == 1.)
    return cormat



def create_dataset_relunc_vector(dataset):
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
    return curDCS



def create_relunc_vector(datablock_list):
    DCS_list = []
    for datablock in datablock_list:
        dataset_list = datablock['datasets']
        for dataset in dataset_list:
            curDCS = create_dataset_relunc_vector(dataset)
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



def create_relative_dataset_covmat(dataset):
    """Create correlation matrix of dataset."""
    numpts = len(dataset['CSS'])
    uncs = create_dataset_relunc_vector(dataset)
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
    covmat = np.zeros((numpts, numpts), dtype=float)
    for KS in range(numpts):
        for KT in range(KS):
            Q1 = 0.
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

            CERR = Q1 + XNORU
            # CERR contains the covariance element.
            # if corresponding correlation too large, reduce it
            if CERR / (uncs[KS] * uncs[KT]) > 0.99:
                CERR = 0.99 * uncs[KS] * uncs[KT]

            covmat[KS, KT] = CERR
            covmat[KT, KS] = CERR

        covmat[KS, KS] = uncs[KS]*uncs[KS]
    return covmat



def create_dataset_cormat(dataset):
    covmat = create_relative_dataset_covmat(dataset)
    uncs = np.sqrt(np.diag(covmat))
    cormat = covmat / uncs.reshape(1,-1) / uncs.reshape(-1,1)
    np.fill_diagonal(cormat, 1.)
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
        return cormat

    # fill the correlations into the correlation matrix
    covmat = np.zeros((numpts, numpts), dtype=float)
    for ds in dslist:
        dsid = ds['NS']
        start_ofs1 = start_ofs_dic[dsid]
        next_ofs1 = next_ofs_dic[dsid]
        numpts = len(ds['CSS'])
        # calculate the correlations within a dataset
        covmat[start_ofs1:next_ofs1, start_ofs1:next_ofs1] = \
                create_relative_dataset_covmat(ds)
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
                            C22 = ENFF2[pNC2] if ENFF2 is not None else 0

                        Q1 += AMUFA*C11*C22

                    covmat[ofs1, ofs2] = Q1
                    covmat[ofs2, ofs1] = covmat[ofs1, ofs2]

                covmat[ofs1, ofs1] = uncs[ofs1]*uncs[ofs1]

    return covmat



def create_experimental_covmat(datablock_list, css, uncs,
        effuncs=None, fix_ppp_bug=True):
    """Calculate experimental covariance matrix."""
    if effuncs is None:
        effuncs = uncs.copy()
    absuncvec = effuncs.copy()
    absuncvec *= 0.01 * css
    covmat_list = []
    start_idx = 0
    for db in datablock_list:
        numpts = 0
        for ds in db['datasets']:
            numpts += len(ds['CSS'])
        next_idx = start_idx + numpts
        curuncs = uncs[start_idx:next_idx]
        cureffuncs = effuncs[start_idx:next_idx]
        curabsuncs = absuncvec[start_idx:next_idx]
        sclmat = np.outer(curabsuncs, curabsuncs)
        curcormat = create_datablock_cormat(db,
                uncs = curuncs,
                effuncs = cureffuncs if not fix_ppp_bug else None)

        if 'ECOR' not in db:
            curcormat = relcov_to_wrong_cor(curcormat, curuncs, cureffuncs, db['datasets'])

        curcormat = fix_cormat(curcormat)

        curcovmat = curcormat * sclmat
        covmat_list.append(csr_matrix(curcovmat))
        start_idx = next_idx

    covmat = block_diag(covmat_list, format='csr')
    return covmat


