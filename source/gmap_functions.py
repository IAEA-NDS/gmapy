from fortran_utils import fort_range, fort_write
from gmap_snippets import (get_AX, get_prior_range,
                           get_dataset_range, get_dataset_id_from_idx,
                           is_usable_datapoint)

import numpy as np



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

        if ID != data.num_datasets:
            raise IndexError('user correlation matrix must be given in last dataset of datablock')
        if data.NCOX[ID] != data.num_datapoints:
            raise IndexError('user correlation matrix dimension must match number of datapoints in datablock')
        data.ECOR = data.userECOR.copy()

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




def fill_AA_AM_COV(data, APR):
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
            raise ValueError('Unusable datapoints as present in dataset ' + str(NS) + ' are not permitted')

        N = N + 1

        if MT == 6:
            #
            #      FISSION AVERAGE
            #
            fisdata = APR.fisdata
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
            data.predCSS[N] = CX
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
                data.predCSS[N] = CX
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
                data.predCSS[N] = CX
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
                data.predCSS[N] = CX
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
                data.predCSS[N] = CX
                continue

    return

