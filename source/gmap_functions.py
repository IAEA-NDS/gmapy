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
