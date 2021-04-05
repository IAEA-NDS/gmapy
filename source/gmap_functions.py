from goto import with_goto
from fortran_utils import fort_range, fort_read, fort_write

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


