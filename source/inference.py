from fortran_utils import fort_range
from gmap_snippets import get_dataset_range
from data_management import SIZE_LIMITS, init_gauss
from gmap_functions import (init_shape_prior, construct_Ecor, apply_PPP_correction,
        fill_AA_AM_COV)

import numpy as np



# constants for size limits
#      LDA   MAX NO IN DATA BLOCK
#      LDB NO OF UNKNOWNS
LDA = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
LDB = SIZE_LIMITS.MAX_NUM_UNKNOWNS
LDF = 200



def construct_effECOR(data):
    #
    #      INVERT ECOR
    #
    MODC = data.MODC
    N = data.num_datapoints
    curmat = data.ECOR[1:(N+1), 1:(N+1)].copy()
    if np.any(curmat.diagonal() != 1):
        raise ValueError('All diagonal elements of correlation matrix must be one')

    IREP = 0
    failed = True
    data.num_inv_tries = 0
    while failed and IREP < 15:
        failed = False
        try:
            np.linalg.cholesky(curmat)
        except np.linalg.LinAlgError:
            #
            #      ATTEMPT TO MAKE CORR. MATRIX POSITIVE DEFINITE
            #
            data.num_inv_tries += 1
            failed = True
            IREP=IREP+1

            if MODC == 2:
                curmat = np.identity(N)
            else:
                CXZ=0.10
                mask = np.ones(curmat.shape, dtype=bool)
                np.fill_diagonal(mask, 0)
                curmat[mask] /= (1.+CXZ)

    data.effECOR[1:(N+1), 1:(N+1)] = curmat
    return not failed



def update_dummy_dataset(ID, data, APR):
        NS = data.IDEN[ID, 6]
        NCT = data.NCT[ID]
        if NCT != 1 and (NS >= 900 and NS <= 909):
            raise IndexError('dummy datasets must not be composite')

        start_idx, end_idx = get_dataset_range(ID, data)
        for idx in fort_range(start_idx, end_idx):
            #
            #      Exception for dummy data sets
            #
            if NS >= 900 and NS <= 909:
                data.CSS[idx] = APR.CS[data.KAS[idx,1]]
            # .lbl48



def update_prior_estimates(APR, updated_values):
    APR.CS[1:(APR.NR+1)] = updated_values[:APR.NR]



def update_prior_shape_estimates(APR, updated_values):
    NR = APR.NR
    NRS = NR + APR.NSHP
    APR.CS[(NR+1):(NRS+1)] = updated_values[NR:]



def accounting(ID, data, APR):
    #
    #      ACCOUNTING
    #
    #      N,NADD      NO OF TOTAL DATA POINTS SO FAR IN BLOCK
    #      ID          NO OF EXPERIMENTAL DATA SETS
    #      NP          NO OF DATA POINTS IN THIS SET
    #
    IDEN = data.IDEN
    NS = IDEN[ID, 6]
    MT = IDEN[ID, 7]
    KAS = data.KAS
    NT = data.NT[ID,:]
    NCT = data.NCT[ID]

    NADD_MIN, NADD_MAX = get_dataset_range(ID, data)
    for NADD in fort_range(NADD_MIN, NADD_MAX):  # .lbl21

        #
        #      SORT EXP ENERGIES  TO FIND CORRESPONDING INDEX OF EVALUATION EN
        #
        #      KAS(I,L)   GIVES INDEX OF EVALUATION ENERGY FOR I.TH EXP POINT
        #                 AND L.TH CROSS SECTION
        #
        if MT != 6:
            #
            #      NCT is the number of cross sections involved
            #
            for L in fort_range(1,NCT):  # .lbl48
                JE = APR.MCS[NT[L], 2]
                JI = APR.MCS[NT[L], 3]

                found = False
                for K in fort_range(JE, JI):  # .lbl12
                    E1 = .999*APR.EN[K]
                    E2 = 1.001*APR.EN[K]
                    if data.E[NADD] > E1 and data.E[NADD] < E2:
                        found = True
                        break

                if not found:
                    print('ERROR: experimental energy does not match energy mesh')
                    exit()

                KAS[NADD, L] = K

    return



def link_prior_and_datablocks(APR, datablock_list):
    APR.NSHP = 0
    for data in datablock_list:
        if data.num_datasets == 0:
            continue

        for ID in fort_range(1, data.num_datasets):
            NS = data.IDEN[ID,6]
            if data.IDEN[ID, 7] != 6:
                MTTP = data.IDEN[ID, 8]
                if MTTP == 2:
                    APR.NSHP += 1
                    APR.NSETN[APR.NSHP] = NS
                    data.problematic_L_dimexcess[ID] = APR.NR + APR.NSHP
                    if APR.NR + APR.NSHP > SIZE_LIMITS.MAX_NUM_UNKNOWNS:
                        raise IndexError('Too many shape datasets')

        for ID in fort_range(1, data.num_datasets):
            accounting(ID, data, APR)

        for ID in fort_range(1, data.num_datasets):
            MT = data.IDEN[ID,7]
            MTTP = data.IDEN[ID,8]
            if MT != 6 and MTTP == 2:
                init_shape_prior(ID, data, APR)



def add_compinfo_to_datablock(datablock, fisdata, APR, MPPP):

    data = datablock
    if data.num_datasets == 0:
        return

    for ID in fort_range(1, data.num_datasets):
        update_dummy_dataset(ID, data, APR)

    data.ECOR.fill(0)
    for ID in fort_range(1, data.num_datasets):
        construct_Ecor(ID, data)

        #VPBEG Assigning uncertainties as % error relative the prior
        if MPPP == 1 and data.IDEN[ID,7] != 6:
            apply_PPP_correction(ID, data, APR)

    fill_AA_AM_COV(datablock, fisdata, APR)

    success = construct_effECOR(data)
    if not success:
       raise ValueError('Correlation matrix of datablock is not invertible\n' + \
                        '(starting with dataset ' + str(data.IDEN[1,6]) + ')')

