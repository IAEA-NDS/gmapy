from fortran_utils import fort_range
from gmap_snippets import get_dataset_range, count_usable_datapoints 
from data_management import SIZE_LIMITS, init_gauss
from gmap_functions import (init_shape_prior, construct_Ecor, apply_PPP_correction,
        fill_AA_AM_COV)

import linpack_slim
from linpack_utils import (pack_symmetric_matrix, unpack_symmetric_matrix,
                           unpack_utriang_matrix)

import numpy as np



# constants for size limits
#      LDA   MAX NO IN DATA BLOCK
#      LDB NO OF UNKNOWNS
LDA = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
LDB = SIZE_LIMITS.MAX_NUM_UNKNOWNS
LDF = 200



def invert_Ecor(data):
    #
    #      INVERT ECOR
    #
    MODC = data.MODC
    N = data.num_datapoints

    data.effECOR = data.ECOR.copy()
    data.num_inv_tries = 0

    if MODC == 2 or N == 1:
        data.invECOR = data.ECOR.copy()
        return True

    IREP = 0
    while True:
        # cholesky decomposition
        #CALL DPOFA(ECOR,LDA,N,INFO)
        # INFO = np.array(0)
        # choleskymat = np.array(data.effECOR[1:(N+1),1:(N+1)], dtype='float64', order='F')
        # linpack_slim.dpofa(a=choleskymat, info=INFO)

        # ALTERNATIVE USING NUMPY FUNCTION cholesky
        INFO = 0
        try:
            choleskymat = np.linalg.cholesky(data.effECOR[1:(N+1), 1:(N+1)]).T
        except np.linalg.LinAlgError:
            INFO = 1

        if INFO == 0:
            break
        else:
            #
            #      ATTEMPT TO MAKE CORR. MATRIX POSITIVE DEFINITE
            #
            data.num_inv_tries += 1
            IREP=IREP+1
            N1=N-1
            for K in fort_range(1,N1):  # .lbl2211
                K1=K+1
                for L in fort_range(K1, N):  # .lbl2211
                    if MODC == 2:
                        data.effECOR[L,K] = 0.
                    data.effECOR[K,L] = data.effECOR[L,K]

            for K in fort_range(1,N):  # .lbl2212
                data.effECOR[K,K] = 1.

            CXZ=0.10
            for K in fort_range(1,N):  # .lbl37
                for L in fort_range(1,N):
                    data.effECOR[K,L]=data.effECOR[K,L]/(1.+CXZ)
                    if K == L:
                        data.effECOR[K,L] = 1.

            if IREP >= 15:
                return False

    JOB=1
    # CALL DPODI(ECOR,LDA,N,DET,JOB)
    tmp = np.array(choleskymat, dtype='float64', order='F')
    tmp_det = np.array([0., 0.], dtype='float64', order='F') 
    linpack_slim.dpodi(tmp, det=tmp_det, job=JOB)
    data.invECOR[1:(N+1),1:(N+1)] = tmp

    # ALTERNATIVE USING NUMPY inv function
    # tmp = inv(data.ECOR[1:(N+1),1:(N+1)])
    # data.ECOR[1:(N+1),1:(N+1)] = np.matmul(tmp.T, tmp)

    for K in fort_range(2,N):  # .lbl17
        L1=K-1
        for L in fort_range(1,L1):
            data.invECOR[K,L] = data.invECOR[L,K]
        L = L + 1  # to match L value of fortran after loop

    return True



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

    for ID in fort_range(1, data.num_datasets):
        if ID > 1:
            start_idx, end_idx = get_dataset_range(ID-1, data)
            num_datapoints_used = count_usable_datapoints(data, end_idx)
        else:
            num_datapoints_used = 0

        data.IDEN[ID,2] = num_datapoints_used + 1

    data.ECOR.fill(0)
    for ID in fort_range(1, data.num_datasets):
        construct_Ecor(ID, data)

        #VPBEG Assigning uncertainties as % error relative the prior
        if MPPP == 1 and data.IDEN[ID,7] != 6:
            apply_PPP_correction(ID, data, APR)

        fill_AA_AM_COV(datablock, fisdata, APR)

    invertible = invert_Ecor(data)
    if not invertible:
       raise ValueError('Correlation matrix of datablock is not invertible\n' + \
                        '(starting with dataset ' + str(data.IDEN[1,6]) + ')')

