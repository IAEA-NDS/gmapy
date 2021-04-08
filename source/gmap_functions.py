from goto import with_goto
from generic_utils import unflatten
from fortran_utils import fort_range, fort_read, fort_write
import numpy as np

@with_goto
def read_prior(MC1, MC2, APR, LABL, IPP, file_IO3, file_IO4):
    #
    #      INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
    #
    #      MC1=NE      NO OF ENERGIES/PARAMETERS (total)
    #      MC2=NC      NO OF CROSS SECTION TYPES
    #      NR          NO OF PARAMETERS (cross sections)
    #
    label .lbl1
    NE = MC1
    NC = MC2
    NE1 = NE+1
    NR = 0
    for K in fort_range(1,NC):  # .lbl33
        format120 = r"(2A8)"
        LABL.CLAB[K, 1:3] = fort_read(file_IO3, format120)
        for L in fort_range(1,NE1):  # .lbl34
            format103 = "(2E10.4)"
            EX1, CSX1 = fort_read(file_IO3, format103)
            if EX1 == 0:
                goto .lbl35
            NR += 1
            APR.EN[NR] = EX1
            APR.CS[NR] = CSX1
            # label .lbl34
        L = L + 1  # to match L value of fortran after loop
        label .lbl35
        APR.MCS[K,1] = L-1
    # label .lbl33

    #
    #  MCS(k,l) CONTAINES IN l= 1 TOTAL NUMBER EACH CS
    #                           2 START BOUNDARY
    #                           3 END OF BOUNDARY
    #
    APR.MCS[1, 2] = 1
    APR.MCS[1, 3] = APR.MCS[1,1]
    for K in fort_range(2,NC):
        APR.MCS[K, 2] = APR.MCS[K-1, 2] + APR.MCS[K-1, 1]
        APR.MCS[K, 3] = APR.MCS[K-1, 3] + APR.MCS[K, 1]
    # label .lbl30
    format134 = r"(//2X,36HCROSS SECTIONS OF PRESENT EVALUATION//)" 
    fort_write(file_IO4, format134, [])
    format135 = "(10X,I3,5X,2A8)"
    for K in fort_range(1,NC):
        fort_write(file_IO4, format135, [K, LABL.CLAB[K, 1:3]])
    # label .lbl22
    if IPP[1] == 0:
        goto .lbl665
    format136 = "(1H1//,2X,35HENERGIES AND APRIORI CROSS SECTIONS//)" 
    fort_write(file_IO4, format136, [])
    format137 = "(/ '     INDEX     E/MEV   ',7X,2A8 /)"
    for  K in fort_range(1,NC):  # .lbl24
        fort_write(file_IO4, format137, LABL.CLAB[K,1:3])
        JC1 = APR.MCS[K, 2]
        JC2 = APR.MCS[K, 3]
        LQ = 0
        format138 = "(2X,2I4,3X,E10.4,3X,F15.8)"
        for L in fort_range(JC1, JC2):
            LQ += 1
            fort_write(file_IO4, format138, [LQ, L, APR.EN[L], APR.CS[L]])
            # label .lbl 23
        L = L + 1  # to match L value of fortran after loop
    # .lbl24
    K = K + 1
    label .lbl665
    format113 = "(/,' TOTAL NO OF PARAMETERS ',I4/)"
    fort_write(file_IO4, format113, [NR])

#
#      for checking
#
    if IPP[7] == 0:
        return (NC, NR)

    format4390 = "(' No of Parameters per Cross Section '/)"
    fort_write(file_IO4, format4390, [])
    format154 = "(3X,3HAPR.MCS,10I5)" 
    fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 1]])
    format4391 = "(/' Start Address '/)"
    fort_write(file_IO4, format4391, [])
    fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 2]])
    format4392 = "(/' End Address '/)"
    fort_write(file_IO4, format4392, [])
    fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 3]])
    return (NC, NR)


@with_goto
def read_block_input(data, gauss, LDA, LDB, KA, KAS, MODREP, file_IO4):
    #
    #      BLOCK INPUT
    #
    #      N     TOTAL NO OF DATA POINTS IN BLOCK
    #      ID    TOTAL NO OF DATA SETS IN BLOCK
    #

    N = 0
    ID = 0
    for K in fort_range(1,LDA):  # .lbl32
        gauss.AM[K] = 0.
        for I in fort_range(1,LDB):  # .lbl81
            # VP line with BM(I)=0.D0  was commented because BM(I) cleaning should 
            # VP done outside of the cycle on measured data 
            # VPBEG*****************************************************************
            # VP      BM(I)=0.D0
            # VPEND*****************************************************************
            KA[I,K] = 0
            gauss.AA[I,K] = 0.
            # .lbl81
        for J in fort_range(1,5):
            KAS[K, J] = 0
        for L in fort_range(1,LDA):
            data.ECOR[K,L] = 0.
        L = L + 1  # to match L value of fortran after loop
        # .lbl32
    NADD = 1
    if MODREP != 0:
        return (ID, N, NADD)   
    format108 = "(/' DATABLOCK************************DATABLOCK**************" + \
                "******************************************DATABLOCK '/)"
    fort_write(file_IO4, format108, [])
    return (ID, N, NADD)


@with_goto
def read_dataset_input(MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
        data, LABL, IDEN, NENF, NETG, NCSST, NEC, NT,
        ID, N, file_IO3, file_IO4):
    #
    #      DATA SET INPUT
    #
    #      MC1 NS      DATA SET NO
    #      MC2 MT      TYPE OF MEASUREMENT
    #      MC3 NCOX    CORRELATION MATRIX GIVEN IF .NE. 0
    #      MC4 NCT     NO OF CROSS SECTIONS INVOLVED
    #      MC5 NT(1)   CROSS SECTION IDENTIFICATION
    #      MC6 NT(2)   SAME
    #      MC7 NT(3)   SAME
    #      MC8 NNCOX   DIVIDE UNCERTAINTIES BY 10.
    #
    # label .lbl2
    NS = MC1
    MT = MC2
    NCOX = MC3
    NCT = MC4
    NT[1] = MC5
    NT[2] = MC6
    NT[3] = MC7
    NNCOX = MC8
    format123 = "(16I5)"
    if NCT > 3:
        NT[4:(NCT+1)] = fort_read(file_IO3, format123)
        L = NCT + 1  # to match L value of fortran after READ loop
    ID = ID+1
    IDEN[ID,2] = N+1
    IDEN[ID,6] = NS
    IDEN[ID,7] = MT
    #
    #       identify absolute or shape data
    #
    MTTP = 1
    IDEN[ID, 8] = 1
    if MT == 2 or MT == 4:
        goto .lbl510
    if MT == 8 or MT == 9:
        goto .lbl510
    goto .lbl511

    label .lbl510

    MTTP = 2
    IDEN[ID, 8] = 2

    label .lbl511

    # VP      if(modrep .ne. 0) go to 140
    format142 = "(//, ' ***********DATASET**************************** '/)"
    fort_write(file_IO4, format142, [])
    NU = NCT
    if NCT > 4:
        NU = 4
    NCT2 = NCT - NU
    NU1 = NU + 1
    format139 = "(2X,8HDATA SET,I5,2X,A16,4(2X,2A8))"
    tmp = [[LABL.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(1,NU)]
    L = 3  # to reflect value of L after loop in Fortran
           # because L in list comprehension goes immediately out of scope
    fort_write(file_IO4, format139, [MC1, LABL.TYPE[MT],tmp])
    if NCT2 <= 0:
        goto .lbl140

    format149 = "(2X,6(2X,2A8))"
    tmp = [[LABL.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(NU1,NCT2)]
    L = 3  # to reflect value of L after loop in Fortran
           # because L in list comprehension goes immediately out of scope
    fort_write(file_IO4, format149, tmp)

    label .lbl140

    #
    #       NAME ID AND REFERENCE I/O
    #
    format131 = "(3I5,4A8,4A8)"
    format132 = "(/' YEAR',I5,' TAG',I3,' AUTHOR:  ',4A8,4A8/)"
    IDEN[ID,3:6], LABL.CLABL[1:5], LABL.BREF[1:5] = unflatten(fort_read(file_IO3, format131), [[3],[4],[4]])
    fort_write(None, format132, [IDEN[ID, 3:5], LABL.CLABL[1:5], LABL.BREF[1:5]])
    # VP      if(modrep .ne. 0) go to 183
    fort_write(file_IO4, format132, [IDEN[ID, 3:5], LABL.CLABL[1:5], LABL.BREF[1:5]])

    label .lbl183
    NCCS = IDEN[ID, 5]
    #
    #       READ(3,    ) NORMALIZATION UNCERTAINTIES
    #
    XNORU = 0.
    if MTTP == 2:
        goto .lbl200

    format201 = "(10F5.1,10I3)"
    data.ENFF[ID, 1:11], NENF[ID, 1:11] = unflatten(fort_read(file_IO3, format201), [[10],[10]])

    #
    #       CALCULATE TOTAL NORMALIZATION UNCERTAINTY
    #
    for L in fort_range(1,10):  # .lbl208
        XNORU = XNORU + (data.ENFF[ID,L])**2
    L = L + 1  # to match L value of fortran after loop

    label .lbl200
    #
    #       READ(3,    ) ENERGY DEPENDENT UNCERTAINTY PARAMETERS
    #
    format202 = "(3F5.2,I3)"
    for K in fort_range(1,11):
        data.EPAF[1:4, K, ID], NETG[K, ID] = unflatten(fort_read(file_IO3, format202), [[3], 1])
    #
    #       READ(3,    ) CORRELATIONS INFORMATION
    #
    if NCCS == 0:
        #goto .lbl203
        return (MT, NCT, NS, NCOX, NNCOX, XNORU, NCCS, MTTP, ID, IDEN)

    format841 = "(10F5.1)"
    format205 = "(I5,20I3)"
    for K in fort_range(1,NCCS):  # .lbl204
        NCSST[K], tmp = unflatten(fort_read(file_IO3, format205), [1,[20]])
        NEC[1:3, 1:11, K] = np.reshape(tmp, (2,10), order='F')
        #NCSST[K], NEC[0:2, 0:10, K] = unflatten(fort_read(file_IO3, format205), [1,[20]])
        data.FCFC[1:11, K] = fort_read(file_IO3, format841)
    return (MT, NCT, NS, NCOX, NNCOX, XNORU, NCCS, MTTP, ID, IDEN)



@with_goto
def accounting(data, APR, MT, NT, NCT,
        KAS, NS, NADD, LDA, NNCOX, MOD2, XNORU, file_IO3):
    #
    #      ACCOUNTING
    #
    #      N,NADD      NO OF TOTAL DATA POINTS SO FAR IN BLOCK
    #      ID          NO OF EXPERIMENTAL DATA SETS
    #      NP          NO OF DATA POINTS IN THIS SET
    #
    NALT = NADD

    for KS in fort_range(1,LDA):  # .lbl21

        format109 = "(2E10.4,12F5.1)"
        data.E[NADD], data.CSS[NADD], data.CO[1:13, NADD] = unflatten(fort_read(file_IO3, format109), [2,[12]])
        L = 13  # to reflect fortran value after READ loop
        if data.E[NADD] == 0:
            return (NALT, L, NADD)
            #goto .lbl95

        #
        #      SORT EXP ENERGIES  TO FIND CORRESPONDING INDEX OF EVALUATION EN
        #
        #      KAS(I,L)   GIVES INDEX OF EVALUATION ENERGY FOR I.TH EXP POINT
        #                 AND L.TH CROSS SECTION
        #
        if MT == 6:
            goto .lbl70

        #
        #      NCT is the number of cross sections involved
        #
        for L in fort_range(1,NCT):  # .lbl48
            JE = APR.MCS[NT[L], 2]
            JI = APR.MCS[NT[L], 3]

            for K in fort_range(JE, JI):  # .lbl12
                E1 = .999*APR.EN[K]
                E2 = 1.001*APR.EN[K]
                if data.E[NADD] > E1 and data.E[NADD] < E2:
                    goto .lbl75
            # .lbl12
            goto .lbl15
            label .lbl75
            KAS[NADD, L] = K
            #
            #      Exception for dummy data sets
            #
            if NS >= 900 and NS <= 909:
                data.CSS[NADD] = APR.CS[K]
            # .lbl48
        L = L + 1  # to match L value of fortran after loop

        #
        #      this is the Axton special (uncertainties have been multiplied by 10
        #         in order to preserve precision beyond 0.1%)
        #
        label .lbl70
        if NNCOX == 0:
            goto .lbl59
        for LZ in fort_range(1,11):  # .lbl57
            data.CO[LZ, NADD] = data.CO[LZ, NADD] / 10.

        label .lbl59

        #
        #      test option:  as set with mode control
        #
        #      changing weights of data based on year or data set tag
        #
        if MOD2 == 0:
            goto .lbl320
        if MOD2 > 1000:
            goto .lbl321
        if MOD2 == 10:
            goto .lbl322
        if MOD2 > 10:
            goto .lbl320

        # replaces computed goto
        if MOD2 == 1:
            goto .lbl331
        if MOD2 > 1 and MOD2 < 10:
            goto .lbl336

        label .lbl331

        #
        #      downweighting data sets with tags .NE. 1
        #
        if IDEN[ID,4] == 1:
            goto .lbl320

        label .lbl342
        for I in fort_range(3,11):
            data.CO[I, NADD] = AMO3*data.CO[I, NADD]

        goto .lbl320

        #
        #      downweighting based on year of measurement
        #
        label .lbl321
        if IDEN[ID, 3] < MOD2:
            goto .lbl342

        goto .lbl320

        label .lbl322
        #
        #      downweighting of specified data sets
        #
        for IST in fort_range(1,NELI): # .lbl391
            if IDEN[ID, 6] == NRED[IST]:
                goto .lbl342

        label .lbl391
        goto .lbl320

        label .lbl336
        format339 = "('  WEIGHTING OPTION NOT IMPLEMENTED, DATA SET  ',I5/)" 
        fort_write(file_IO4, format339, NS)

        label .lbl320
        #
        #      CALCULATE TOTAL UNCERTAINTY  DCS
        #
        RELU = 0.
        for L in fort_range(3,11):  # .lbl207
            RELU += data.CO[L, NADD]**2
        L = L + 1  # to match L value of fortran after loop

        data.DCS[NADD] = np.sqrt(XNORU + RELU) 
        NADD += 1

        label .lbl15
    label .lbl21
    return (NALT, L, NADD)



@with_goto
def should_exclude_dataset(NS, IELIM, NELIM, MTTP, ID, NADD, NALT, file_IO4): 

    NP = NADD - NALT
    if IELIM == 0:
        goto .lbl173
    #
    #      data set excluded ?
    #
    for K in fort_range(1,IELIM):
        if NS == NELIM[K]:
            goto .lbl517

    label .lbl172
    #
    #      NO VALID DATA POINTS OR SET REQUESTED TO BE EXCLUDED
    #
    label .lbl173
    if NP == 1 and MTTP == 2:
        goto .lbl517

    if NP != 0:
        return (False, NP, ID, NADD)

    label .lbl517
    format168 = "(' SET ',I5,' W/O VALID POINTS OR ELIMINATED'/)"
    fort_write(file_IO4, format168, [NS])
    ID = ID - 1
    NADD = NALT
    return (True, NP, ID, NADD)



@with_goto
def construct_Ecor(data, NETG, IDEN, NCSST, NEC,
        L, MODC, NCOX, NALT, NP, NADD1, ID,
        XNORU, NCCS, MTTP, NS, file_IO3, file_IO4):
    #
    #      CONSTRUCT ECOR
    #
    #         MODE  1   INPUT OF ECOR
    #               2   UNCORRELATED
    #               3   ALL CORRELATED ERRORS GIVEN
    #
    if NCOX == 0:
        goto .lbl5001
    MODAL = MODC
    MODC = 1

    label .lbl5001
    if MODC == 1:
        goto .lbl54
    elif MODC == 2:
        goto .lbl1779
    elif MODC >= 3 and MODC <= 6:
        goto .lbl56
    #
    #      INPUT OF ECOR
    #
    label .lbl54
    format161 = "(10F8.5)"
    for KS in fort_range(1,NCOX):  # .lbl61
        num_el_read = 0
        num_el_desired = KS
        res = []
        while num_el_read < num_el_desired:
            tmp = fort_read(file_IO3, format161)
            tmp = [x for x in tmp if x is not None]
            res += tmp
            num_el_read += len(tmp)

        data.ECOR[KS,1:(KS+1)] = res
    KS = KS + 1  # to match the value of KS after Fortran loop
    L = KS  # to match the value of L after READ (label 61 in Fortran code)


    MODC = MODAL
    goto .lbl79


    #
    #       CONSTRUCT ECOR FROM UNCERTAINTY COMPONENTS
    #
    label .lbl56
    data.ECOR[NALT, NALT] = 1.
    if NP == 1:
        goto .lbl1789
    NALT1 = NALT + 1
    for KS in fort_range(NALT1, NADD1):  # .lbl62
        C1 = data.DCS[KS]
        KS1 = KS - 1

        for KT in fort_range(NALT, KS1):  # .lbl162
            Q1 = 0.
            C2 = data.DCS[KT]
            for L in fort_range(3,11):  # .lbl215
                if NETG[L, ID] == 9:
                    goto .lbl214
                if NETG[L, ID] == 0:
                    goto .lbl214

                FKS = data.EPAF[1,L,ID] + data.EPAF[2,L,ID]
                XYY = data.EPAF[2,L,ID] - (data.E[KS]-data.E[KT])/(data.EPAF[3,L,ID]*data.E[KS])
                if XYY < 0.:
                    XYY = 0.
                FKT = data.EPAF[1, L, ID] + XYY
                Q1=Q1+data.CO[L,KS]*data.CO[L,KT]*FKS*FKT

                label .lbl214
                label .lbl215
            L = L + 1  # to match L value of fortran after loop
            CERR = (Q1 + XNORU) / (C1*C2)

            if CERR > .99:
                CERR = .99
            # limit accuracy of comparison to reflect
            # Fortran behavior

            label .lbl162
            data.ECOR[KS,KT] = CERR

        data.ECOR[KS, KS] = 1.
    label .lbl62

    #
    #   ADD CROSS CORRELATIONS OF EXPERIMENTAL DATA BLOCK
    #
    label .lbl1789
    if ID == 1:
        goto .lbl79
    if NCCS == 0:
        goto .lbl79
    ID1 = ID - 1
    
    for I in fort_range(1,NCCS):  # .lbl271

        NSET = NCSST[I]
        for II in fort_range(1,ID1):  # .lbl272
            if IDEN[II,6] == NSET:
                goto .lbl273
        #
        #   CORRELATED DATA SET NOT FOUND AHEAD OF PRESENT DATA
        #   SET WITHIN DATA BLOCK
        #
        format274 = "('CORRELATED DATA SET  ',I5,' NOT FOUND FOR SET ',I5)" 
        fort_write(file_IO4, format274, [NSET, NS])
        goto .lbl275

        label .lbl273
        NCPP = IDEN[II, 1]
        #
        #      cross correlation
        #
        MTT = IDEN[II, 8]
        if MTT == 2 and NP == 1:
            goto .lbl275
        if MTTP == 2 and NCPP == 1:
            goto .lbl275

        label .lbl469
        NCST = IDEN[II, 2]
        NCED = NCPP + NCST - 1
        for K in fort_range(NALT, NADD1):  # .lbl278
            C1 = data.DCS[K]
            for KK in fort_range(NCST, NCED):  # .lbl279
                C2 = data.DCS[KK]
                Q1 = 0.
                for KKK in fort_range(1,10):  # .lbl281
                    NC1 = NEC[1, KKK, I]
                    NC2 = NEC[2, KKK, I]
                    if NC1 > 21 or NC2 > 21:
                        goto .lbl2811
                    if NC1 == 0:
                        goto .lbl2753
                    if NC2 == 0:
                        goto .lbl2753
                    AMUFA = data.FCFC[KKK, I]
                    if NC1 > 10:
                        goto .lbl310
                    C11 = data.ENFF[ID, NC1]
                    goto .lbl311

                    label .lbl310
                    NC1 = NC1 - 10
                    if NETG[NC1, ID] == 9:
                        goto .lbl2800
                    FKT = data.EPAF[1, NC1, ID] + data.EPAF[2, NC1, ID]
                    goto .lbl2801

                    label .lbl2800
                    FKT = 1.
                    label .lbl2801

                    C11 = FKT*data.CO[NC1, K]

                    label .lbl311
                    if NC2 > 10:
                        goto .lbl312
                    C22 = data.ENFF[II, NC2]
                    goto .lbl313

                    label .lbl312
                    NC2 = NC2 - 10

                    if NETG[NC2, II] == 9:
                        goto .lbl2802

                    XYY = data.EPAF[2,NC2,II] - np.abs(data.E[K]-data.E[KK])/ (data.EPAF[3,NC2,II]*data.E[KK])
                    if XYY < 0.:
                        XYY = 0.
                    FKS = data.EPAF[1, NC2, II] + XYY
                    goto .lbl2803

                    label .lbl2802
                    FKS = 1.

                    label .lbl2803
                    C22 = FKS * data.CO[NC2, KK]

                    label .lbl313
                    Q1 = Q1 + AMUFA*C11*C22

                    label .lbl2811
                label .lbl281
                label .lbl2753
                data.ECOR[K,KK] = Q1/(C1*C2)

            label .lbl279
        label .lbl278
        label .lbl275
    label .lbl271

    goto .lbl79

    #
    #       UNCORRELATED OR SINGLE VALUE
    #
    label .lbl1779
    for KLK in fort_range(NALT, NADD1):  # .lbl74
        data.ECOR[KLK,KLK] = 1.

    label .lbl79
    return (MODC, L)

