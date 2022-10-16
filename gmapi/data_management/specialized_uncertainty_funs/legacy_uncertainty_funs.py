import numpy as np
from ...mappings.priortools import SHAPE_MT_IDS
from ..unc_utils import calculate_ppp_factors, cov2cor, cor2cov



# this function is here to reproduce
# a bug in GMAP Fortran in the PPP correction
def relcov_to_wrong_cor(relcovmat, datasets, css):
    ppp_factors = calculate_ppp_factors(datasets, css)
    uncs = np.sqrt(np.diagonal(relcovmat))
    effuncs = uncs * ppp_factors
    cormat = np.copy(relcovmat)
    # get number of points in datablock
    cur_idx = 0
    ds_idcs = []
    for ds in datasets:
        ds_idcs.append(cur_idx)
        numpts = len(ds['CSS'])
        cur_idx += numpts
    ds_idcs.append(cur_idx)
    # start treating the correlations
    for start_idx, end_idx in zip(ds_idcs[:-1], ds_idcs[1:]):
        # correct calculation within dataset
        cormat[start_idx:end_idx, start_idx:end_idx] = \
                cov2cor(cormat[start_idx:end_idx, start_idx:end_idx])
        # but for correlations between datasets
        # corrected and uncorrected uncertainties are mixed
        cormat[start_idx:end_idx, :start_idx] /= \
                uncs[start_idx:end_idx].reshape(-1,1)
        cormat[start_idx:end_idx, :start_idx] /= \
                effuncs[:start_idx].reshape(1,-1)
    np.fill_diagonal(cormat, 1.)
    # symmetrize the matrix
    cormat[np.triu_indices_from(cormat,k=1)] = \
            cormat.T[np.triu_indices_from(cormat, k=1)]
    assert np.all(cormat.diagonal() == 1.)
    return cormat



def create_dataset_relunc_vector(dataset):
    XNORU = 0.
    # debug #6
    if (dataset['MT'] not in SHAPE_MT_IDS or dataset['NS'] in (1029,1030)):
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



def create_relunc_vector(datablock):
    DCS_list = []
    dataset_list = datablock['datasets']
    for dataset in dataset_list:
        curDCS = create_dataset_relunc_vector(dataset)
        DCS_list.append(curDCS)
    DCS = np.concatenate(DCS_list)
    return DCS



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
    problematic_datasets = {}
    for KS in range(numpts):
        for KT in range(KS):
            Q1 = 0.
            for L in range(2,11):
                if dataset['NETG'][L] not in (0,9):
                    FKS = EPAF[0,L] + EPAF[1,L]
                    if EPAF[2,L] == 0.:
                        if dataset['NS'] not in problematic_datasets:
                            print(f'Warning: EPAF[2,{L}] is zero for dataset {dataset["NS"]} '
                                  f'(MT: {dataset["MT"]}, {dataset.get("BREF","").strip()}, '
                                  f'{dataset.get("CLABL").strip()}). '
                                   'You may want to check the uncertainty specifications '
                                   'of this datasset.')
                            problematic_datasets[dataset['NS']] = True
                    XYY = EPAF[1,L] - (E[KS]-E[KT])/(EPAF[2,L]*E[KS])
                    XYY = max(XYY, 0.)
                    FKT = EPAF[0,L] + XYY
                    Q1 += CO[KS,L]*CO[KT,L]*FKS*FKT

            XNORU = 0.
            # debug #5
            if (MT not in SHAPE_MT_IDS or dataset['NS'] in (1029,1030)):
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



def create_relative_datablock_covmat(datablock, shouldfix=True):
    """Create correlation matrix of datablock."""

    uncs = create_relunc_vector(datablock)
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
        covmat = cor2cov(cormat, uncs)
        return covmat

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
        if ds['MT'] in SHAPE_MT_IDS:
            ENFF = np.full(10, 0.)

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
            if ds['MT'] in SHAPE_MT_IDS:
                ENFF2 = np.full(10, 0.)
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
                for KK in range(numpts2):
                    ofs2 = start_ofs2 + KK
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

