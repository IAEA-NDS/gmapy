from generic_utils import unflatten, Bunch
from fortran_utils import fort_range, fort_read, fort_write
from data_management import init_datablock, init_gauss, SIZE_LIMITS
from gmap_snippets import (should_downweight, get_AX, get_prior_range,
                           get_dataset_range, get_dataset_id_from_idx)

import numpy as np

import linpack_slim
from linpack_utils import (pack_symmetric_matrix, unpack_symmetric_matrix,
                           unpack_utriang_matrix)


# constants for size limits
#      LDA   MAX NO IN DATA BLOCK
#      LDB NO OF UNKNOWNS
LDA = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
LDB = SIZE_LIMITS.MAX_NUM_UNKNOWNS
LDF = 200

#
#      test option:  forced stop for testing purpose
#
def force_stop(file_IO4):
    format107 = "( '  REQUESTED STOP ' )"
    fort_write(file_IO4, format107)
    exit()





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



def construct_Ecor(ID, data):
    #
    #      CONSTRUCT ECOR
    #
    #         MODE  1   INPUT OF ECOR
    #               2   UNCORRELATED
    #               3   ALL CORRELATED ERRORS GIVEN
    #
    IDEN = data.IDEN
    NETG = data.NETG
    NCCS = IDEN[ID, 5]
    MTTP = IDEN[ID, 8]
    NS = IDEN[ID, 6]
    NALT, NADD1 = get_dataset_range(ID, data)
    NP = IDEN[ID, 1]
    NCSST = data.NCSST
    NEC = data.NEC
    NCOX = data.NCOX[ID]
    MODC = data.MODC

    data.missing_datasets[NS] = []

    if NCOX != 0:
        MODAL = MODC
        MODC = 1

    if MODC == 1:
        # related to INPUT OF ECOR
        L = NCOX+1   # to match the value of L after READ (label 61 in Fortran code)
        MODC = MODAL


    elif MODC == 2:
        #
        #       UNCORRELATED OR SINGLE VALUE
        #
        L = 13  # this values is due to line 252
                # caused by read statement starting with
                # E[NADD], ... in accounting
        for KLK in fort_range(NALT, NADD1):  # .lbl74
            data.ECOR[KLK,KLK] = 1.


    elif MODC >= 3 and MODC <= 6:
        #
        #       CONSTRUCT ECOR FROM UNCERTAINTY COMPONENTS
        #
        data.ECOR[NALT, NALT] = 1.
        if NP == 1:
            L = 13  # this value is due to line 252
                    # caused by read statement starting with
                    # N[NADD], ... in accounting
        else:
            NALT1 = NALT + 1
            for KS in fort_range(NALT1, NADD1):  # .lbl62
                C1 = data.DCS[KS]
                KS1 = KS - 1

                for KT in fort_range(NALT, KS1):  # .lbl162
                    Q1 = 0.
                    C2 = data.DCS[KT]
                    for L in fort_range(3,11):  # .lbl215
                        if NETG[L, ID] == 9:
                            continue
                        if NETG[L, ID] == 0:
                            continue

                        FKS = data.EPAF[1,L,ID] + data.EPAF[2,L,ID]
                        XYY = data.EPAF[2,L,ID] - (data.E[KS]-data.E[KT])/(data.EPAF[3,L,ID]*data.E[KS])
                        if XYY < 0.:
                            XYY = 0.
                        FKT = data.EPAF[1, L, ID] + XYY
                        Q1=Q1+data.CO[L,KS]*data.CO[L,KT]*FKS*FKT

                    L = L + 1  # to match L value of fortran after loop

                    #
                    #       CALCULATE TOTAL NORMALIZATION UNCERTAINTY
                    #
                    XNORU = 0.
                    if data.MTTP[ID] != 2:
                        for K in fort_range(1,10):  # .lbl208
                            XNORU = XNORU + (data.ENFF[ID,K])**2

                    CERR = (Q1 + XNORU) / (C1*C2)

                    if CERR > .99:
                        CERR = .99
                    # limit accuracy of comparison to reflect
                    # Fortran behavior
                    data.ECOR[KS,KT] = CERR
                    data.ECOR[KT,KS] = data.ECOR[KS,KT]

                data.ECOR[KS, KS] = 1.

        #
        #   ADD CROSS CORRELATIONS OF EXPERIMENTAL DATA BLOCK
        #
        if ID != 1 and NCCS != 0:

            ID1 = ID - 1
            for I in fort_range(1,NCCS):  # .lbl271

                NSET = NCSST[ID, I]
                found = False
                for II in fort_range(1,ID1):  # .lbl272
                    if IDEN[II,6] == NSET:
                        found = True
                        break

                if not found:
                    #
                    #   CORRELATED DATA SET NOT FOUND AHEAD OF PRESENT DATA
                    #   SET WITHIN DATA BLOCK
                    #
                    data.missing_datasets[NS].append(NSET)

                else:
                    NCPP = IDEN[II, 1]
                    #
                    #      cross correlation
                    #
                    MTT = IDEN[II, 8]
                    if not (MTT == 2 and NP == 1) and \
                       not (MTTP == 2 and NCPP == 1):

                        NCST = IDEN[II, 2]
                        NCED = NCPP + NCST - 1
                        for K in fort_range(NALT, NADD1):  # .lbl278
                            C1 = data.DCS[K]
                            for KK in fort_range(NCST, NCED):  # .lbl279
                                # TODO: The need to use effDCS to have consistency with the
                                #       Fortran version is probably a bug. Either original
                                #       or PPP corrected uncertainties should be used everywhere.
                                #       Here we use a mix of uncorrected (C1) and corrected (C2)
                                #       uncertainties.
                                C2 = data.effDCS[KK]
                                Q1 = 0.
                                for KKK in fort_range(1,10):  # .lbl281
                                    NC1 = NEC[ID, 1, KKK, I]
                                    NC2 = NEC[ID, 2, KKK, I]
                                    if NC1 > 21 or NC2 > 21:
                                        continue

                                    if NC1 == 0 or NC2 == 0:
                                        break

                                    AMUFA = data.FCFC[ID, KKK, I]
                                    if NC1 > 10:
                                        NC1 = NC1 - 10
                                        if NETG[NC1, ID] == 9:
                                            FKT = 1.
                                        else:
                                            FKT = data.EPAF[1, NC1, ID] + data.EPAF[2, NC1, ID]

                                        C11 = FKT*data.CO[NC1, K]
                                    else:
                                        C11 = data.ENFF[ID, NC1]

                                    if NC2 > 10:
                                        NC2 = NC2 - 10

                                        if NETG[NC2, II] == 9:
                                            FKS = 1.
                                        else:
                                            XYY = data.EPAF[2,NC2,II] - np.abs(data.E[K]-data.E[KK])/ (data.EPAF[3,NC2,II]*data.E[KK])
                                            if XYY < 0.:
                                                XYY = 0.
                                            FKS = data.EPAF[1, NC2, II] + XYY

                                        C22 = FKS * data.CO[NC2, KK]

                                    else:
                                        C22 = data.ENFF[II, NC2]

                                    Q1 = Q1 + AMUFA*C11*C22

                                data.ECOR[K,KK] = Q1/(C1*C2)
                                data.ECOR[KK,K] = data.ECOR[K,KK]

    data.problematic_L_Ecor[ID] = L
    return



def apply_PPP_correction(ID, data, APR):

    start_idx, end_idx = get_dataset_range(ID, data)
    for idx in fort_range(start_idx, end_idx):
        AX = get_AX(ID, idx, data, APR)
        AZ = AX / data.CSS[idx]
        data.effDCS[idx] = AZ*data.DCS[idx]



def init_shape_prior(ID, data, APR):
    #
    #      DETERMINE APRIORI NORMALIZATION FOR SHAPE MEASUREMENTS
    #

    MT = data.IDEN[ID, 7]
    MTTP = data.IDEN[ID,8]
    if MT == 6 or MTTP != 2:
        raise ValueError('init_shape_prior can only be used for shape data and MT != 6')

    start_idx, end_idx = get_dataset_range(ID, data)

    AP = 0.
    WWT = 0.
    for K in fort_range(start_idx, end_idx):  # .lbl29
        CSSK = data.CSS[K]
        DCSK = data.DCS[K]
        WXX = 1./(DCSK*DCSK)
        WWT = WWT + WXX

        AX = get_AX(ID, K, data, APR)
        AZ = AX / CSSK

        AP=AP+AZ*WXX

    NS = data.IDEN[ID,6]
    L = APR.NR + np.where(APR.NSETN == NS)[0][0]
    AP=AP/WWT
    AP = 1.0 / AP
    APR.CS[L] = AP

    return



def is_usable_datapoint(idx, data):

    ID = get_dataset_id_from_idx(idx, data)
    J = data.KAS[idx,1]
    I = data.KAS[idx,2]
    I8 = data.KAS[idx,3]
    MT = data.IDEN[ID, 7]

    if (J == 0 and MT != 6) or \
       (I == 0 and MT in (3,4,5,7,8,9)) or \
       (I8 == 0 and MT in (7,9)):
           return False
    else:
        return True



def count_usable_datapoints(data, end_idx=None):
    end_idx = data.num_datapoints if end_idx is None else end_idx

    N = 0
    if data.num_datapoints > 0:
        for idx in fort_range(1, end_idx):
            if is_usable_datapoint(idx, data):
                N += 1
    return N



def fill_AA_AM_COV(data, fisdata, APR):
    #
    #      FILL AA,AM,AND COV
    #
    IDEN = data.IDEN
    KAS = data.KAS
    KA = data.KA

    KA.fill(0)
    N = 0

    data.AM.fill(0.)
    data.AA.fill(0.)

    for KS in fort_range(1, data.num_datapoints):  # .lbl18

        ID = get_dataset_id_from_idx(KS, data)

        MT =  IDEN[ID, 7]
        NT = data.NT[ID,:]
        NCT = data.NCT[ID]
        NS = IDEN[ID,6]
        MTTP = data.IDEN[ID, 8]

        if MTTP == 2:
            NSHP_IDX = np.where(APR.NSETN == NS)[0]
            if len(NSHP_IDX) > 1:
                raise IndexError('Dataset ' + str(NS) + ' appears multiple times')
            NSHP_IDX = NSHP_IDX[0]

        data.invalid_datapoints[NS] = []

        DQQQ = data.effDCS[KS]*data.CSS[KS]*0.01

        if not is_usable_datapoint(KS, data):
            data.invalid_datapoints[NS].append(KS)
            continue

        N = N + 1

        if MT == 6:
            #
            #      FISSION AVERAGE
            #
            K = 0
            if NT[1] == 9:
                K = 1

            JA, JE = get_prior_range(NT[1], APR)
            NW = 2 if NT[1]==9 else 1
            FL=0.
            SFL=0.

            for LI in fort_range(JA+1, JE-1):  # .lbl53
                NW=NW+1
                FL=FL+fisdata.FIS[NW]
                EL1=(APR.EN[LI]+APR.EN[LI-1])*0.5
                EL2=(APR.EN[LI]+APR.EN[LI+1])*0.5
                DE1=(APR.EN[LI]-EL1)*0.5
                DE2=(EL2-APR.EN[LI])*0.5
                SS1=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI-1]))
                SS2=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI+1]))
                CSSLI=(SS1*DE1+SS2*DE2)/(DE1+DE2)
                SFL=SFL+CSSLI*fisdata.FIS[NW]

            FL=FL+fisdata.FIS[1]+fisdata.FIS[NW+1]
            SFL=SFL+fisdata.FIS[1]*APR.CS[JA]+fisdata.FIS[NW+1]*APR.CS[JE]
            SFIS=SFL/FL

            EAVR = 0.

            data.EAVR[KS] = EAVR
            data.SFIS[KS] = SFIS
            data.FL[KS] = FL

            CX=SFIS
            for J in fort_range(JA, JE):  # .lbl39
                K=K+1
                KA[J,1]=KA[J,1]+1
                KR=KA[J,1]
                KA[J,KR+1]=N
                if J == JA or J == JE:
                    CSSJ = APR.CS[J]
                else:
                    EL1=(APR.EN[J]+APR.EN[J-1])*0.5
                    EL2=(APR.EN[J]+APR.EN[J+1])*0.5
                    DE1=(APR.EN[J]-EL1)*0.5
                    DE2=(EL2-APR.EN[J])*0.5
                    SS1=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J-1]))
                    SS2=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J+1]))
                    CSSJ=(SS1*DE1+SS2*DE2)/(DE1+DE2)

                data.AA[J,KR]=CSSJ*fisdata.FIS[K]/DQQQ

            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 5:
            #
            #      TOTAL CROSS SECTION
            #
            CX = 0.
            for I in fort_range(1,NCT):  # .lbl49
                II = KAS[KS,I]
                CX = CX+APR.CS[II]


            for I in fort_range(1,NCT):  # .lbl60
                J  = KAS[KS,I]
                KA[J,1] = KA[J,1]+1
                KR = KA[J,1]
                KA[J,KR+1] = N
                data.AA[J,KR] = APR.CS[J]/DQQQ

            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 8:
            #
            #   SHAPE OF SUM
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = 0.
            for I in fort_range(1,NCT):  # .lbl253
                II=KAS[KS,I]
                CX=CX+APR.CS[II]*AP

            APDQ=AP/DQQQ
            for I in fort_range(1,NCT):  # .lbl254
                J=KAS[KS,I]
                KA[J,1]=KA[J,1]+1
                KR=KA[J,1]
                KA[J,KR+1]=N
                data.AA[J,KR]=APR.CS[J]*APDQ

            KA[L,1]=KA[L,1]+1
            KR=KA[L,1]
            KA[L,KR+1]=N
            data.AA[L,KR]=CX/DQQQ
            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue


        J = KAS[KS,1]
        I = KAS[KS,2]
        I8 = KAS[KS,3]

        KA[J,1] = KA[J,1] + 1
        KR = KA[J,1]
        KA[J,KR+1] = N

        if MT == 1:
            #
            #      CROSS SECTION
            #
            CX = APR.CS[J]
            data.AA[J,KR] = CX / DQQQ
            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 2:
            #
            #      CROSS SECTION SHAPE    L is shape data norm. const. index
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = APR.CS[J]*AP
            CXX = CX/DQQQ
            data.AA[J,KR] = CXX
            KA[L,1] = KA[L,1]+1
            KR = KA[L,1]
            KA[L,KR+1] = N
            data.AA[L,KR] =  CXX
            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 3:
            #
            #      RATIO
            #
            CX = APR.CS[J]/APR.CS[I]
            CCX = CX/DQQQ
            data.AA[J,KR] = CCX
            KA[I,1] = KA[I,1]+1
            KR = KA[I,1]
            KA[I,KR+1] = N
            data.AA[I,KR] = -CCX
            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 4:
            #
            #      RATIO SHAPE
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = APR.CS[J]*AP/APR.CS[I]
            CXX = CX/DQQQ
            data.AA[J,KR] = CXX
            KA[I,1] = KA[I,1]+1
            KR = KA[I,1]
            KA[I,KR+1] = N
            data.AA[I,KR] = -CXX
            KA[L,1] = KA[L,1]+1
            KR = KA[L,1]
            KA[L,KR+1] = N
            data.AA[L,KR] =  CXX
            data.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 7:
            #
            #   ABSOLUTE RATIO S1/(S2+S3)
            #
            CX=APR.CS[J]/(APR.CS[I]+APR.CS[I8])
            if I == J:
                CBX=CX*CX*APR.CS[I8]/(APR.CS[J]*DQQQ)
                data.AA[J,KR]=CBX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                data.AA[I8,KR]=-CBX
                data.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue
            else:
                CBX=CX/DQQQ
                data.AA[J,KR]=CBX
                KA[I,1]=KA[I,1]+1
                KR=KA[I,1]
                KA[I,KR+1]=N
                CBX2=CBX*CBX*DQQQ/APR.CS[J]
                CCX=CBX2*APR.CS[I]
                data.AA[I,KR]=-CCX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                CCX=CBX2*APR.CS[I8]
                data.AA[I8,KR]=-CCX
                data.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue

        elif MT == 9:
            #
            #   SHAPE OF RATIO S1/(S2+S3)
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CII8=APR.CS[I]+APR.CS[I8]
            CX=AP*APR.CS[J]/CII8
            CBX=CX/DQQQ
            if I == J:
                CCX=CBX*APR.CS[I8]/CII8
                data.AA[J,KR]=CCX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                data.AA[I8,KR]=-CCX
                KA[L,1]=KA[L,1]+1
                KR=KA[L,1]
                KA[L,KR+1]=N
                data.AA[L,KR]=CBX
                data.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue
            else:
                data.AA[J,KR]=CBX
                KA[I,1]=KA[I,1]+1
                KR=KA[I,1]
                KA[I,KR+1]=N
                CDX=CBX*APR.CS[I]/CII8
                data.AA[I,KR]=-CDX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                CDX=CBX*APR.CS[I8]/CII8
                data.AA[I8,KR]=-CDX
                KA[L,1]=KA[L,1]+1
                KR=KA[L,1]
                KA[L,KR+1]=N
                data.AA[L,KR]=CBX
                data.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue

    return



def complete_symmetric_Ecor(data):
    #
    #      FILL IN SYMMETRIC TERM
    #
    MODC = data.MODC
    N = data.num_datapoints_used
    for K in fort_range(1,N-1):  # .lbl25
        for L in fort_range(K+1, N):  # .lbl25
            if MODC == 2:
                data.ECOR[L, K] = 0.
            data.ECOR[K, L] = data.ECOR[L, K]
            # label .lbl25



def invert_Ecor(data):
    #
    #      INVERT ECOR
    #
    MODC = data.MODC
    N = count_usable_datapoints(data)

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



def get_matrix_products(gauss, data, APR):
    #
    #      GET MATRIX PRODUCTS
    #
    NRS=APR.NR + APR.NSHP
    KA = data.KA
    N = data.num_datapoints_used
    gauss.NTOT += data.num_datapoints_used
    NTOT = gauss.NTOT
    SIGMA2 = gauss.SIGMA2

    for I in fort_range(1,NRS):  # .lbl90
        NI=KA[I,1]
        if NI == 0:
            continue

        for J in fort_range(I, NRS):  # .lbl83
            NJ=KA[J,1]
            if NJ == 0:
                continue
            IJ=J*(J-1)//2+I

            for MI in fort_range(1,NI):  # .lbl85
                MIX=KA[I,MI+1]
                for MJ in fort_range(1,NJ):  # .lbl85
                    MJX=KA[J,MJ+1]
                    gauss.B[IJ]=gauss.B[IJ]+data.AA[I,MI]*data.AA[J,MJ]*data.invECOR[MIX,MJX]

    for I in fort_range(1,NRS):  # .lbl91
        NI=KA[I,1]
        if NI == 0:
            continue

        for MI in fort_range(1,NI):  # .lbl86
            MIX=KA[I,MI+1]
            for MJ in fort_range(1,N):  #.lbl86
                gauss.BM[I]=gauss.BM[I]+data.AA[I,MI]*data.invECOR[MIX,MJ]*data.AM[MJ]

    for I in fort_range(1,N):  # .lbl26
        SUX=0.
        for J in fort_range(1,N):  # .lbl52
            SUX=SUX+data.invECOR[I,J]*data.AM[J]
        
        SIGMA2=SIGMA2+data.AM[I]*SUX

    data.NTOT = NTOT
    data.SIGL=SIGMA2/NTOT
    if N > LDA:
        exit()
    if NRS > LDB:
        exit()

    gauss.SIGMA2 = SIGMA2



def get_result(gauss, APR):
    #
    #      GETTING THE RESULT
    #
    NRS = APR.NR + APR.NSHP
    NUMEL = NRS*(NRS+1)//2
    tmp = np.array(gauss.B[1:(NUMEL+1)], dtype='float64', order='F')
    tmp = unpack_utriang_matrix(tmp)
    try:
        tmp = np.linalg.cholesky(tmp.T).T 
    except np.linalg.LinAlgError:
        format105 = "(/' EXP BLOCK CORREL. MATRIX NOT PD',20X,'***** WARNING *')" 
        format106 = "( '  SOLUTION  CORREL. MATRIX NOT PD ' )"
        fort_write(None, format106)
        exit()
    tmp = np.linalg.inv(tmp)
    tmp = np.matmul(tmp, tmp.T)
    gauss.DE[1:(NRS+1)]=gauss.DE[1:(NRS+1)]+ np.matmul(tmp, gauss.BM[1:(NRS+1)])

    gauss.B[1:(NUMEL+1)] = pack_symmetric_matrix(tmp)



def update_prior_estimates(APR, gauss):
    for L in fort_range(1, APR.NC):  # .lbl14
        JA=APR.MCS[L,2]
        JI=APR.MCS[L,3]
        for K in fort_range(JA, JI):  # .lbl77
            APR.CS[K] = APR.CS[K]*(1.+gauss.DE[K])


def update_prior_shape_estimates(APR, gauss):
    NR = APR.NR
    NRS = NR + APR.NSHP
    NR1=NR+1
    if APR.NSHP != 0:
        for K in fort_range(NR1, NRS):  # .lbl82
            APR.CS[K] = APR.CS[K]*(1.+gauss.DE[K])



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
            data.num_datapoints_used = count_usable_datapoints(data)

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
        if data.NCOX[ID] != 0:
            if ID != data.num_datasets:
                raise IndexError('user correlation matrix must be given in last dataset of datablock')
            if data.NCOX[ID] != data.num_datapoints:
                raise IndexError('user correlation matrix dimension must match number of datapoints in datablock')
            data.ECOR = data.userECOR.copy()

        #VPBEG Assigning uncertainties as % error relative the prior
        if MPPP == 1 and data.IDEN[ID,7] != 6:
            apply_PPP_correction(ID, data, APR)

        fill_AA_AM_COV(datablock, fisdata, APR)

    invertible = invert_Ecor(data)
    if not invertible:
       raise ValueError('Correlation matrix of datablock is not invertible\n' + \
                        '(starting with dataset ' + str(data.IDEN[1,6]) + ')')



def gls_update(datablock_list, APR):
    gauss = init_gauss()

    for data in datablock_list:
        get_matrix_products(gauss, data, APR)

    get_result(gauss, APR)
    return gauss

