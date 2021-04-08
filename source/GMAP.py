# helpful for fortran to python conversion
from goto import with_goto
from generic_utils import Bunch, unflatten
from fortran_utils import fort_range, fort_read, fort_write

import linpack_slim
from linpack_utils import (pack_symmetric_matrix, unpack_symmetric_matrix,
                           unpack_utriang_matrix)

# other python packages
import os
import numpy as np

# refactoring
from gmap_functions import (read_prior, read_block_input,
        read_dataset_input, accounting, should_exclude_dataset,
        construct_Ecor)


#################################################
#   START OF GMAP PROGRAM
##################################################

@with_goto
def main():

    # IMPLICIT definitions in original version
    # IMPLICIT REAL*8 (A-H,O-Z)

    #
    #      KAS        indexes of experimental cross sections
    #      NT         id of cross sections involved in measured quantity
    #      IDEN       data set info (see below)
    #      NSETN      shape data set numbers
    #      IPP        i/o choices
    #      NENF       tags of norm. uncertainty components
    #      NETG       tags of energy dependent uncertainty components
    #      NCSST      data set Nr.s for cross correlations
    #      NEC        error component pairs for cross correlations
    #      NRED       data set Nr.s for downweighting
    #      KA         indexes
    #      NELIM      data set Nr.s to exclude from evaluation
    #

    #   INTEGER*2 KAS(250,5),NT(5),IDEN(30,8),NSETN(200),IPP(8),
    #  1 NENF(40,10),NETG(11,40),NCSST(10),NEC(2,10,10),NRED(160)
    #  2 ,KA(1200,250),NELIM(40)
    KAS = np.zeros((250+1, 5+1), dtype=int)
    NT = np.zeros(5+1, dtype=int)
    IDEN = np.zeros((30+1, 8+1), dtype=int)
    NSETN = np.zeros(200+1, dtype=int)
    IPP = np.zeros(8+1, dtype=int)
    NENF = np.zeros((40+1, 10+1), dtype=int)
    NETG = np.zeros((11+1,40+1), dtype=int)
    NCSST = np.zeros(10+1, dtype=int)
    NEC = np.zeros((2+1,10+1,10+1), dtype=int)
    NRED = np.zeros(160+1, dtype=int)
    KA = np.zeros((1200+1, 250+1), dtype=int)
    NELIM = np.zeros(40+1, dtype=int)

    # TODO: otherwise error thrown 'variable accessed before assignment'
    #       is this an issue in the original Fortran code?
    EAVR = 0.
    K1 = 0

    basedir = '.'
    file_IO3 = open('data.gma', 'r')
    file_IO4 = open('gma.res', 'w')
    file_IO5 = open('plot.dta', 'w')

    #   Parameters/apriori
    #
    #      EN    ENERGY GRID
    #      CS    APRIORI CROSS SECTIONS
    #      MCS   INDEXES
    #
    APR = Bunch({
        'EN': np.zeros(1200+1, dtype=float),
        'CS': np.zeros(1200+1, dtype=float),
        'MCS': np.zeros((35+1,3+1), dtype=int)
        })

    #
    #   Data block / data set   
    #
    #      E       ENERGIES OF EXPERIMENTAL DATA SET
    #      CSS     MEASUREMENT VALUES OF EXPERIMENTAL DATA SET
    #      DCS     TOTAL UNCERTAINTIES OF EXPERIMENTAL VALUES
    #      CO      ENERGY DEPENDENT UNCERTAINTY COMPONENTS
    #      ECOR    CORRELATION MATRIX OF EXPERIMENTS IN DATA BLOCK
    #      ENFIS   ENERGIES OF FISSION SPECTRUM
    #      FIS     FISSION SPECTRUM*BINWIDTH
    #      ENFF    NORMALIZATION UNCERTAINTIES COMPONENTS
    #      EPAF    UNCERTAINTY COMPONENT PARAMETERS
    #      FCFC    CROSS CORRELATION FACTORS
    #
    data = Bunch({
        'E': np.zeros(250+1, dtype=float),
        'CSS': np.zeros(250+1, dtype=float),
        'DCS': np.zeros(250+1, dtype=float),
        'FIS': np.zeros(250+1, dtype=float),
        'ECOR': np.zeros((250+1, 250+1), dtype=float),
        'ENFIS': np.zeros(250+1, dtype=float),
        'CO': np.zeros((12+1, 250+1), dtype=float),
        'ENFF': np.zeros((30+1, 10+1), dtype=float),
        'EPAF': np.zeros((3+1, 11+1, 30+1), dtype=float),
        'FCFC': np.zeros((10+1, 10+1), dtype=float)
        })

    #
    #       AA      COEFFICIENT MATRIX
    #       AM      MEASUREMENT VECTOR
    # VP    ECOR    inverse of correlation matrix of measurement vector AM
    # VP            or relative weight matrix
    # VP    BM      vector accumulating (in data block cycle) sum of 
    # VP            AA(transpose)*ECOR*AM
    # VP    B       matrix accumulating (in data block cycle) sum of 
    # VP            AA(transpose)*ECOR*AA
    #       DE      ADJUSTMENT VECTOR
    # VP            equals B(inverse)*BM
    #
    gauss = Bunch({
            'AA': np.zeros((1200+1, 250+1), dtype=float),
            'AM': np.zeros(250+1, dtype=float),
            'DE': np.zeros(1200+1, dtype=float),
            'BM': np.zeros(1200+1, dtype=float),
            'B': np.zeros(720600+1, dtype=float)
    })

    #
    #  IDEN(K,I)  K DATA SET SEQUENCE
    #
    #             I=1 NO OF DATA POINTS IN SET
    #             I=2 INDEX OF FIRST VALUE IN ECOR
    #             I=3 YEAR
    #             I=4 DATA SET TAG
    #             I=5 NO OF OTHER SETS WITH WHICH CORRELATIONS ARE GIVEN
    #             I=6 DATA SET NR
    #             I=7 MT-TYPE OF DATA
    #             I=8 1 ABSOLUTE, 2 SHAPE
    #
    #  CONTROLS/LABELS
    #
    LABL = Bunch({
        'AKON':  [None for i in range(12)],
        'CLAB': np.empty((35+1,2+1), dtype=object),
        'CLABL': np.empty(4+1, dtype=object),
        'TYPE': np.zeros(10+1, dtype=object),
        'BREF': np.zeros(4+1, dtype=object)

        })
    #
    #      CONTROL
    #      MODE    MODE OF EXP. CORRELATION MATRIX CONSTRUCTION
    #      I/OC    I/O CONTROL
    #      APRI    INPUT OF APRIORI CROSS SECTIONS
    #      FIS*    FISSION FLUX FACTORS
    #      BLCK    START OF DATA BLOCK
    #      DATA    INPUT OF DATA
    #      END*    END OF EXP  DATA, START OF EVALUATION
    #      EDBL    END OF EXP. DATA SET BLOCK
    #      ELIM    DATA SETS TO BE ELIMINATED
    #

    LABL.AKON[1] = 'APRI'
    LABL.AKON[2] = 'DATA'
    LABL.AKON[3] = 'END*'
    LABL.AKON[4] = 'BLCK'
    LABL.AKON[5] = 'I/OC'
    LABL.AKON[7] = 'EDBL'
    LABL.AKON[8] = 'FIS*'
    LABL.AKON[9] = 'MODE'
    LABL.AKON[10] = 'STOP'
    LABL.AKON[11] = 'ELIM'
    #
    #      MEASUREMENT TYPE IDENTIFICATION
    #
    LABL.TYPE[1] = 'CROSS SECTION   '
    LABL.TYPE[2] = 'CS-SHAPE        '
    LABL.TYPE[3] = 'RATIO           '
    LABL.TYPE[4] = 'RATIO SHAPE     '
    LABL.TYPE[5] = 'TOTAL  CS       '
    LABL.TYPE[6] = 'FISSION AVERAGE '
    LABL.TYPE[7] = 'ABS. S1/(S2+S3) '
    LABL.TYPE[8] = 'SHAPE OF SUM    '
    LABL.TYPE[9] = 'SHP. S1/(S2+S3) '
    #
    #      INITIALIZE PARAMETERS
    #
    #      NTOT TOTAL NO OF DATA POINTS
    #      LDA   MAX NO IN DATA BLOCK
    #      LDB NO OF UNKNOWNS
    #
    #      DATA DE/1200*0.D0/,BM/1200*0.D0/,B/720600*0.D0/,NRED/160*0/
    #
    MODREP = 0
    NTOT = 0
    SIGMA2 = 0.
    LDF = 200
    LDA = 250
    LDB = 1200
    LDBB2 = LDB*(LDB+1)//2
    MODC = 3
    NSHP = 0
    NFIS = 0
    IELIM = 0
    LLL = 0
    # VP nullning of the vector BM and vector (matrix) B which keep
    # VP accumilate sum in cycle on experimental data
    # VPBEG********************************************************
    gauss.BM.fill(0.0)
    gauss.B.fill(0.0)

    #
    #      CONTROL
    #
    #      AKON     CONTROL WORD  A4
    #      MC1-8    PARAMETERS   8I5
    #
    format110 = (r"1H1,' PROGRAM  GMA'//," +
                 r"'    GAUSS-MARKOV-AITKEN LEAST SQUARES NUCLEAR DATA EVALUATION'//," +
                 r"'    W.P.POENITZ,ANL'///")
    fort_write(file_IO4, format110, [])
    #
    #      Control parameter input
    #
    format100 = "(A4,1X,8I5)"
    label .lbl50
    ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8 = fort_read(file_IO3,  format100)
    #
    for K in fort_range(1, 11):  # .lbl10
        if ACON == LABL.AKON[K]:
            if K == 1:
                # INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
                NC, NR = read_prior(MC1, MC2, APR, LABL, IPP, file_IO3, file_IO4)
                goto .lbl50
            if K == 2:
                goto .lbl2
            if K == 3:
                goto .lbl3
            if K == 4:
                ID, N, NADD = read_block_input(data, gauss, LDA, LDB, KA, KAS, MODREP, file_IO4)
                goto .lbl50
            if K == 5:
                goto .lbl5
            if K == 6:
                format104 = "(A4,2X,'  CONTROL CODE UNKNOWN')"
                fort_write(file_IO4, format104, [ACON])
                exit()
            if K == 7:
                goto .lbl7
            if K == 8:
                goto .lbl8
            if K == 9:
                goto .lbl9
            if K == 10:
                goto .lbl51
            if K == 11:
                # input:  data set numbers which are to be excluded from the evaluation
                IELIM = MC1
                format171 = r"(16I5)"
                NELIM = fort_read(file_IO3, format171)
                goto .lbl50

    # end loop: .lbl10

    #
    #      test option:  forced stop for testing purpose
    #
    label .lbl51
    format107 = "( '  REQUESTED STOP ' )"
    fort_write(file_IO4, format107)
    exit()
    #
    #      I/O CONTROL
    #
    label .lbl5
    IPP = [None for i in range(9)]
    IPP[1] = MC1
    IPP[2] = MC2
    IPP[3] = MC3
    IPP[4] = MC4
    IPP[5] = MC5
    IPP[6] = MC6
    IPP[7] = MC7
    IPP[8] = MC8
    goto .lbl50


    label .lbl2
    MT, NCT, NS, NCOX, NNCOX, XNORU, NCCS, MTTP, ID, IDEN = \
    read_dataset_input(
            MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
            data, LABL, IDEN, NENF, NETG, NCSST, NEC, NT,
            ID, N, file_IO3, file_IO4
    )

    NALT, L, NADD = \
    accounting(
            data, APR, MT, NT, NCT,
            KAS, NS, NADD, LDA, NNCOX, MOD2, XNORU, file_IO3
    )


    exclflag, NP, ID, NADD = \
    should_exclude_dataset(
            NS, IELIM, NELIM, MTTP, ID, NADD, NALT, file_IO4
    )

    if exclflag:
        goto .lbl50

    #
    #      continue for valid data
    #
    IDEN[ID, 1] = NP
    NADD1 = NADD - 1


    MODC, L = \
    construct_Ecor(
            data, NETG, IDEN, NCSST, NEC,
            L, MODC, NCOX, NALT, NP, NADD1, ID,
            XNORU, NCCS, MTTP, NS, file_IO3, file_IO4
    )

    if MT == 6:
        goto .lbl28
    #
    #   output of KAS for checking
    #
    if IPP[7] == 0:
        goto .lbl2309

    format702 = "(20I5)"
    for K in fort_range(1,NCT):
        fort_write(file_IO4, format702, [KAS[NALT:(NADD1+1)], K])
    #
    #      DETERMINE APRIORI NORMALIZATION FOR SHAPE MEASUREMENTS
    #
    label .lbl2309
    if MTTP == 1:
        goto .lbl2828

    NSHP = NSHP + 1
    NSETN[NSHP] = NS
    L = NR + NSHP
    if L > LDB:
        goto .lbl76

    label .lbl2828
    #VP   PRIOR/EXP column is added
    format5173 = "(/'  ENERGY/MEV   VALUE    ABS. UNCERT. " + \
                 " PRIOR/EXP UNCERT./%    DIFF./%" + \
                 "  VAL.*SQRT(E)'/)"
    fort_write(file_IO4, format5173, [])
    AP = 0.
    WWT = 0.
    for K in fort_range(NALT, NADD1):  # .lbl29
        CSSK = data.CSS[K]
        DCSK = data.DCS[K]
        WXX = 1./(DCSK*DCSK)
        WWT = WWT + WXX

        label .lbl97
        KX = KAS[K, 1]
        KY=KAS[K,2]
        KZ=KAS[K, 3]
        AX = APR.CS[KX]
        if MT == 4 or MT == 3:
            AX = AX / APR.CS[KY]
        if MT == 8 or MT == 5:
            AX = AX + APR.CS[KY]
        if MT == 5 and NCT == 3:
            AX = AX + APR.CS[KZ]
        if MT == 8 and NCT == 3:
            AX = AX + APR.CS[KZ]
        if MT == 9 or MT == 7:
            AX = AX/(APR.CS[KY]+APR.CS[KZ])

        AZ = AX / CSSK

        #VPBEG Assigning uncertainties as % error relative the prior
        if MPPP == 1:
            data.DCS[K] = AZ*data.DCS[K]
        #VPEND 
        #
        #      DATA OUTPUT
        #
        if IPP[2] == 0:
            goto .lbl99

        SECS = np.sqrt(data.E[K])*CSSK
        FDQ = DCSK * CSSK/100.
        DIFF = (CSSK-AX)*100./AX
        #VP   AZ print out was added
        format133 = "(2X,E10.4,2X,E10.4,2X,E10.4,3X,F6.4,3X,F6.2," + \
                    " 3X,F10.2,3X,F10.4)"
        fort_write(file_IO4, format133, [data.E[K], CSSK, FDQ, AZ, DCSK, DIFF, SECS])
        #VP   Print out for Ratio of pior/exp value is added
        label .lbl99
        #
        AP=AP+AZ*WXX
    label .lbl29  # end for loop


    AP=AP/WWT
    # VP      if(modrep .ne. 0) go to 2627
    format111 = "(/' APRIORI NORM ',I4,F10.4,I5,2X,4A8)"
    fort_write(file_IO4, format111, [L, AP, NS, LABL.CLABL[1:5]])
    label .lbl2627
    if MTTP == 2:
        goto .lbl2826

    goto .lbl28

    label .lbl2826
    if MODREP != 0:
        goto .lbl63

    AP = 1.0 / AP
    APR.CS[L] = AP
    goto .lbl28

    label .lbl63
    AP=APR.CS[L]

    label .lbl28
    #
    #      FILL AA,AM,AND COV
    #
    for KS in fort_range(NALT, NADD1):  # .lbl18
        DQQQ = data.DCS[KS]*data.CSS[KS]*0.01
        J = KAS[KS,1]
        I = KAS[KS,2]
        I8 = KAS[KS,3]
        if J == 0 and MT != 6:
            goto .lbl89
        if I != 0:
            goto .lbl148
        if MT == 3:
            goto .lbl89
        if MT == 5:
            goto .lbl89
        if MT == 4:
            goto .lbl89
        if MT == 7:
            goto .lbl89
        if MT == 8:
            goto .lbl89
        if MT == 9:
            goto .lbl89

        label .lbl148
        if I8 != 0:
            goto .lbl147
        if MT == 7:
            goto .lbl89
        if MT == 9:
            goto .lbl89

        label .lbl147
        N = N + 1

        if MT == 6:
            goto .lbl46
        if MT == 5:
            goto .lbl45
        if MT == 8:
            goto .lbl248

        KA[J,1] = KA[J,1] + 1
        KR = KA[J,1]
        KA[J,KR+1] = N

        if MT == 1:
            goto .lbl41
        if MT == 2:
            goto .lbl42
        if MT == 3:
            goto .lbl43
        if MT == 4:
            goto .lbl44
        if MT == 5:
            goto .lbl446
        if MT == 6:
            goto .lbl446
        if MT == 7:
            goto .lbl247
        if MT == 8:
            goto .lbl446
        if MT == 9:
            goto .lbl249

        label .lbl446
        format447 = "(10H ERROR 446)"
        fort_write(file_IO4, format447, [])
        exit()
        #
        #      CROSS SECTION
        #
        label .lbl41
        CX = APR.CS[J]
        gauss.AA[J,KR] = CX / DQQQ
        goto .lbl36

        #
        #      CROSS SECTION SHAPE    L is shape data norm. const. index
        #
        label .lbl42
        CX = APR.CS[J]*AP
        CXX = CX/DQQQ
        gauss.AA[J,KR] = CXX
        KA[L,1] = KA[L,1]+1
        KR = KA[L,1]
        KA[L,KR+1] = N
        gauss.AA[L,KR] =  CXX
        goto .lbl36
        #
        #      RATIO
        #
        label .lbl43
        CX = APR.CS[J]/APR.CS[I]
        CCX = CX/DQQQ
        gauss.AA[J,KR] = CCX
        KA[I,1] = KA[I,1]+1
        KR = KA[I,1]
        KA[I,KR+1] = N
        gauss.AA[I,KR] = -CCX
        goto .lbl36
        #
        #      RATIO SHAPE
        #
        label .lbl44
        CX = APR.CS[J]*AP/APR.CS[I]
        CXX = CX/DQQQ
        gauss.AA[J,KR] = CXX
        KA[I,1] = KA[I,1]+1
        KR = KA[I,1]
        KA[I,KR+1] = N
        gauss.AA[I,KR] = -CXX
        KA[L,1] = KA[L,1]+1
        KR = KA[L,1]
        KA[L,KR+1] = N
        gauss.AA[L,KR] =  CXX
        goto .lbl36
        #
        #      TOTAL CROSS SECTION
        #
        label .lbl45
        CX = 0.
        for I in fort_range(1,NCT):  # .lbl49
            II = KAS[KS,I]
            CX = CX+APR.CS[II]


        for I in fort_range(1,NCT):  # .lbl60
            J  = KAS[KS,I]
            KA[J,1] = KA[J,1]+1
            KR = KA[J,1]
            KA[J,KR+1] = N
            gauss.AA[J,KR] = APR.CS[J]/DQQQ

        goto .lbl36

        #
        #      FISSION AVERAGE
        #
        label .lbl46
        K = 0
        if NT[1] == 9:
            K = 1
        JA = APR.MCS[NT[1],2]
        JE = APR.MCS[NT[1],3]
        NW1=1
        if NT[1] == 9:
            NW1 = 2
        NW=NW1
        FL=0.
        SFL=0.
        J1=JA+1
        J2=JE-1
        
        for LI in fort_range(J1, J2):  # .lbl53
            NW=NW+1
            FL=FL+data.FIS[NW]
            EL1=(APR.EN[LI]+APR.EN[LI-1])*0.5
            EL2=(APR.EN[LI]+APR.EN[LI+1])*0.5
            DE1=(APR.EN[LI]-EL1)*0.5
            DE2=(EL2-APR.EN[LI])*0.5
            SS1=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI-1]))
            SS2=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI+1]))
            CSSLI=(SS1*DE1+SS2*DE2)/(DE1+DE2)
            SFL=SFL+CSSLI*data.FIS[NW]

        FL=FL+data.FIS[1]+data.FIS[NW+1]
        SFL=SFL+data.FIS[1]*APR.CS[JA]+data.FIS[NW+1]*APR.CS[JE]
        SFIS=SFL/FL
        format156 = "( 'AP FISSION AVERAGE ',3F10.4,'  EXP. VAL. ',2F10.4)"
        fort_write(file_IO4, format156, [EAVR, SFIS, FL, data.CSS[KS], data.DCS[KS]])
        CX=SFIS
        for J in fort_range(JA, JE):  # .lbl39
            K=K+1
            KA[J,1]=KA[J,1]+1
            KR=KA[J,1]
            KA[J,KR+1]=N
            if J == JA or J == JE:
                goto .lbl195
            EL1=(APR.EN[J]+APR.EN[J-1])*0.5
            EL2=(APR.EN[J]+APR.EN[J+1])*0.5
            DE1=(APR.EN[J]-EL1)*0.5
            DE2=(EL2-APR.EN[J])*0.5
            SS1=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J-1]))
            SS2=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J+1]))
            CSSJ=(SS1*DE1+SS2*DE2)/(DE1+DE2)
            goto .lbl196
            label .lbl195
            CSSJ = APR.CS[J]
            label .lbl196

            gauss.AA[J,KR]=CSSJ*data.FIS[K]/DQQQ

        goto .lbl36

        #
        #   ABSOLUTE RATIO S1/(S2+S3)
        #
        label .lbl247
        CX=APR.CS[J]/(APR.CS[I]+APR.CS[I8])
        if I == J:
            goto .lbl251
        CBX=CX/DQQQ
        gauss.AA[J,KR]=CBX
        KA[I,1]=KA[I,1]+1
        KR=KA[I,1]
        KA[I,KR+1]=N
        CBX2=CBX*CBX*DQQQ/APR.CS[J]
        CCX=CBX2*APR.CS[I]
        gauss.AA[I,KR]=-CCX
        KA[I8,1]=KA[I8,1]+1
        KR=KA[I8,1]
        KA[I8,KR+1]=N
        CCX=CBX2*APR.CS[I8]
        gauss.AA[I8,KR]=-CCX
        goto .lbl36

        label .lbl251
        CBX=CX*CX*APR.CS[I8]/(APR.CS[J]*DQQQ)
        gauss.AA[J,KR]=CBX
        KA[I8,1]=KA[I8,1]+1
        KR=KA[I8,1]
        KA[I8,KR+1]=N
        gauss.AA[I8,KR]=-CBX
        goto .lbl36
        #
        #   SHAPE OF SUM
        #
        label .lbl248
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
            gauss.AA[J,KR]=APR.CS[J]*APDQ

        KA[L,1]=KA[L,1]+1
        KR=KA[L,1]
        KA[L,KR+1]=N
        gauss.AA[L,KR]=CX/DQQQ
        goto .lbl36

        #
        #   SHAPE OF RATIO S1/(S2+S3)
        #
        label .lbl249
        CII8=APR.CS[I]+APR.CS[I8]
        CX=AP*APR.CS[J]/CII8
        CBX=CX/DQQQ
        if I == J:
            goto .lbl390

        gauss.AA[J,KR]=CBX
        KA[I,1]=KA[I,1]+1
        KR=KA[I,1]
        KA[I,KR+1]=N
        CDX=CBX*APR.CS[I]/CII8
        gauss.AA[I,KR]=-CDX
        KA[I8,1]=KA[I8,1]+1
        KR=KA[I8,1]
        KA[I8,KR+1]=N
        CDX=CBX*APR.CS[I8]/CII8
        gauss.AA[I8,KR]=-CDX
        KA[L,1]=KA[L,1]+1
        KR=KA[L,1]
        KA[L,KR+1]=N
        gauss.AA[L,KR]=CBX
        goto .lbl36

        label .lbl390
        CCX=CBX*APR.CS[I8]/CII8
        gauss.AA[J,KR]=CCX
        KA[I8,1]=KA[I8,1]+1
        KR=KA[I8,1]
        KA[I8,KR+1]=N
        gauss.AA[I8,KR]=-CCX
        KA[L,1]=KA[L,1]+1
        KR=KA[L,1]
        KA[L,KR+1]=N
        gauss.AA[L,KR]=CBX
        goto .lbl36

        label .lbl36
        gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
        goto .lbl667

        label .lbl89
        format704 = "( '  DATA POINT BUT NOT AN AP FOR SET ',I5,' NO ',I4)"
        fort_write(file_IO4, format704, [MC1, KS])

        label .lbl667
    label .lbl18  # end of for loop

    goto .lbl50

    #
    #    Data BLOCK complete
    #
    #      FILL IN SYMMETRIC TERM
    #
    label .lbl7
    N1=N-1
    if ID == 0:
        goto .lbl50

    format2830 = "(80X,4HN = ,I5)"
    fort_write(file_IO4, format2830, [N])
    IREP = 0
    for K in fort_range(1,N1):  # .lbl25
        K1 = K + 1
        for L in fort_range(K1, N):  # .lbl25
            if MODC == 2:
                data.ECOR[L, K] = 0.
            data.ECOR[K, L] = data.ECOR[L, K]
            # label .lbl25
        L = L + 1  # to match L value of fortran after loop

    #
    #      output of correlation matrix of data block
    #
    if IPP[3] == 0:
        goto .lbl67
    if N == 1:
        goto .lbl67
    if MODC == 2:
        goto .lbl67

    format101 = "(1H*//,'   CORRELATION MATRIX OF DATA BLOCK'/)"
    fort_write(file_IO4, format101, [])
    format151 = "(1X,24F7.4)"
    for K in fort_range(1,N):
        fort_write(file_IO4, format151, [data.ECOR[K,1:(K+1)]])

    label .lbl67
    #
    #      INVERT ECOR
    #

    if MODC == 2:
        goto .lbl19
    if N == 1:
        goto .lbl19

    # cholesky decomposition
    #CALL DPOFA(ECOR,LDA,N,INFO)
    INFO = np.array(0)
    tmp = np.array(data.ECOR[1:(N+1),1:(N+1)], dtype='float64', order='F')
    linpack_slim.dpofa(a=tmp, info=INFO) 
    data.ECOR[1:(N+1),1:(N+1)] = tmp

    # ALTERNATIVE USING NUMPY FUNCTION cholesky
    # INFO = 0
    # try:
    #     data.ECOR[1:(N+1),1:(N+1)] = cholesky(data.ECOR[1:(N+1), 1:(N+1)]).T 
    # except np.linalg.LinAlgError:
    #     INFO = 1

    if INFO != 0:
        goto .lbl31

    JOB=1
    # CALL DPODI(ECOR,LDA,N,DET,JOB)
    tmp = np.array(data.ECOR[1:(N+1),1:(N+1)], dtype='float64', order='F')
    tmp_det = np.array([0., 0.], dtype='float64', order='F') 
    linpack_slim.dpodi(tmp, det=tmp_det, job=JOB)
    data.ECOR[1:(N+1),1:(N+1)] = tmp

    # ALTERNATIVE USING NUMPY inv function
    # tmp = inv(data.ECOR[1:(N+1),1:(N+1)])
    # data.ECOR[1:(N+1),1:(N+1)] = np.matmul(tmp.T, tmp)


    for K in fort_range(2,N):  # .lbl17
        L1=K-1
        for L in fort_range(1,L1):
            data.ECOR[K,L] = data.ECOR[L,K]
        L = L + 1  # to match L value of fortran after loop
    #
    #      output of inverted correlation matrix of data block
    #
    if IPP[5] == 0:
        goto .lbl19
    for K in fort_range(1,N):
        fort_write(file_IO4, format151, [data.ECOR[K,1:(K+1)]])

    label .lbl19

    #
    #      GET MATRIX PRODUCTS
    #
    NRS=NR+NSHP
    for I in fort_range(1,NRS):  # .lbl90
        NI=KA[I,1]
        if NI == 0:
            goto .lbl92

        for J in fort_range(I, NRS):  # .lbl83
            NJ=KA[J,1]
            if NJ == 0:
                goto .lbl84
            IJ=J*(J-1)//2+I

            for MI in fort_range(1,NI):  # .lbl85
                MIX=KA[I,MI+1]
                for MJ in fort_range(1,NJ):  # .lbl85
                    MJX=KA[J,MJ+1]
                    gauss.B[IJ]=gauss.B[IJ]+gauss.AA[I,MI]*gauss.AA[J,MJ]*data.ECOR[MIX,MJX]
                    
                label .lbl85

            label .lbl84
        label .lbl83
        label .lbl92
    label .lbl90

    for I in fort_range(1,NRS):  # .lbl91
        NI=KA[I,1]
        if NI == 0:
            goto .lbl93

        for MI in fort_range(1,NI):  # .lbl86
            MIX=KA[I,MI+1]
            for MJ in fort_range(1,N):  #.lbl86
                gauss.BM[I]=gauss.BM[I]+gauss.AA[I,MI]*data.ECOR[MIX,MJ]*gauss.AM[MJ]

        label .lbl93
    label .lbl91

    for I in fort_range(1,N):  # .lbl26
        SUX=0.
        for J in fort_range(1,N):  # .lbl52
            SUX=SUX+data.ECOR[I,J]*gauss.AM[J]
        
        SIGMA2=SIGMA2+gauss.AM[I]*SUX
    label .lbl26

    NTOT=NTOT+N
    SIGL=SIGMA2/NTOT
    format476 = "(/' ADDED ',I5,' TO GIVE ',I5,' TOTAL',2I5,F10.2/)"
    fort_write(None, format476, [N, NTOT, NSHP, NRS, SIGL])
    if N > LDA:
        exit()
    if NRS > LDB:
        exit()
    if MODREP != 0:
        goto .lbl50

    fort_write(file_IO4, format476, [N, NTOT, NSHP, NRS, SIGL])
    goto .lbl50

    #
    #      GETTING THE RESULT
    #
    label .lbl3

    file_IO3.seek(0,0)
    format6919 = "(' start getting the result ')"
    fort_write(None, format6919, [])
    SIGMAA=SIGMA2/float(NTOT-NRS)
    format9679 = "(/' UNCERTENTY SCALING   ',E12.4/)"
    fort_write(file_IO4, format9679, [SIGMAA])
    NRST=NRS*(NRS+1)/2
    if IPP[8] ==  0:
        goto .lbl51
    if IPP[4] == 0:
        goto .lbl68
    format116 = "(1H*//,'  MATRIX PRODUCT'//)"
    fort_write(file_IO4, format116, [])
    format152 = "(2X,10E10.4)"
    fort_write(file_IO4, format152, gauss.B[1:(NRST+1)])
    label .lbl68
    format2840 = "(80X,9HLDB,NRS= ,2I6,6H  NTOT,I8)"
    fort_write(file_IO4, format2840, [LDB, NRS, NTOT])
    format7103 = "(2E16.8)"
    format6918 = "(' start on matrix inversion ')"
    fort_write(None, format6918, [])


    # CALL DPPFA(B,NRS,INFO)
    NUMEL = NRS*(NRS+1)//2
    INFO = 0.
    tmp = np.array(gauss.B[1:(NUMEL+1)], dtype='float64', order='F')
    linpack_slim.dppfa(ap=tmp, n=NRS, info=INFO)
    gauss.B[1:(NUMEL+1)] = tmp

    # ALTERNATIVE: numpy does not know about the packed storaged format
    #              of symmetric matrices used by DPPFA, so we need to
    #              unpack the matrix first

    # INFO = 0
    # try:
    #     tmp[1:NRS+1,1:NRS+1] = cholesky(tmp[1:NRS+1,1:NRS+1]).T 
    # except np.linalg.LinAlgError:
    #     INFO = 1

    if INFO != 0:
        goto .lbl40

    format9171 = "(' INVERT SOLUTION MATRIX')"
    fort_write(file_IO4, format9171, [])
    fort_write(None, format9171, [])

    JOB = 1
    # CALL DPPDI(gauss.B,NRS,DET,JOB)
    NUMEL = NRS*(NRS+1)//2
    tmp_det = np.array([0., 0.], dtype='float64', order='F')
    tmp = np.array(gauss.B[1:(NUMEL+1)], dtype='float64', order='F')
    linpack_slim.dppdi(ap=tmp, n=NRS, det=tmp_det, job=JOB)
    gauss.B[1:(NUMEL+1)] = tmp

    # ALTERNATIVE: using numpy/scipy functions instead of LINPACK functions
    # invert the matrix, we do it a bit complicated to use the cholesky factor
    # we invert the cholesky factor and then mutliply its inverse by its transposed inverse
    # tmp = unpack_utriang_matrix(tmp)
    # tmp = inv(tmp)
    # tmp = np.matmul(tmp, tmp.T)
    # # pack the result again
    # gauss.B[1:(NUMEL+1)] = pack_symmetric_matrix(tmp)


    format6917 = "(' completed inversion of matrix')"
    fort_write(None, format6917, [])

    for I in fort_range(1,NRS):  # .lbl13
        gauss.DE[I]=0.
        for K in fort_range(1,NRS):  # .lbl13
            IK=K*(K-1)//2+I
            if K < I:
                IK = I*(I-1)//2 + K
            gauss.DE[I]=gauss.DE[I]+gauss.B[IK]*gauss.BM[K]
        label .lbl13

    #
    #      output of the result
    #
    for L in fort_range(1,NC):  # .lbl14
        format117 = "(1H1,'   RESULT',5X,2A8//)" 
        fort_write(file_IO4, format117, [LABL.CLAB[L,1:3]])
        fort_write(file_IO5, format117, [LABL.CLAB[L,1:3]])
        format112 = "( '   E/MEV         CS/B            DCS/B       DCS/%" + \
                    "     DIF/%    CS*SQRT(E)'/)"
        fort_write(file_IO4, format112, [])
        JA=APR.MCS[L,2]
        JI=APR.MCS[L,3]
        FLX=0.

        for K in fort_range(JA, JI):  # .lbl77
            KBK=K*(K-1)//2+K
            DDX=APR.CS[K]*np.sqrt(gauss.B[KBK])
            CXX=APR.CS[K]*(1.+gauss.DE[K])
            CXXD=100.*(CXX-APR.CS[K])/CXX
            for KK in fort_range(1,NFIS):  # .lbl705
                if data.ENFIS[KK] > .999*APR.EN[K] and data.ENFIS[KK] < 1.001*APR.EN[K]:
                    goto .lbl703
            label .lbl705

            goto .lbl706
            label .lbl703
            if K == JA or K == JI:
                goto .lbl295

            EL1=(APR.EN[K]+APR.EN[K-1])*0.5
            EL2=(APR.EN[K]+APR.EN[K+1])*0.5
            DE1=(APR.EN[K]-EL1)*0.5
            DE2=(EL2-APR.EN[K])*0.5
            SS1=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K-1])*APR.CS[K-1]))
            SS2=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K+1])*APR.CS[K+1]))
            CSSK=(SS1*DE1+SS2*DE2)/(DE1+DE2)
            goto .lbl296
            label .lbl295
            CSSK=CXX
            label .lbl296
            FLX=FLX+data.FIS[KK]*CSSK
            label .lbl706
            FQW=DDX*100./CXX
            SECS=np.sqrt(APR.EN[K])*CXX
            format153 = "(1X,E10.4,2F15.8,2X,F6.1,3X,F7.2,3X,F10.5)" 
            fort_write(file_IO4, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            fort_write(file_IO5, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            if MODAP == 0:
                goto .lbl58
            if MODAP == 2 and K <= APR.MCS[5,3]:
                goto .lbl58
            APR.CS[K]=CXX
            label .lbl58
        label .lbl77
        format158 = "(1H*//,'  FISSION AVERAGE ' ,F8.4//)" 
        fort_write(file_IO4, format158, [FLX])
        # label .lbl14  # end of for loop
    L = L + 1  # to match L value of fortran after loop

    #
    #   OUTPUT OF NORM. FACTORS FOR SHAPE DATA
    #
    format114 = "(1H*///, '  NORMALIZATION  OF SHAPE DATA '///)"
    fort_write(file_IO4, format114, [])
    NR1=NR+1
    LLX=0
    if NSHP == 0:
        goto .lbl292

    for K in fort_range(NR1, NRS):  # .lbl82
        LLX=LLX+1
        KK=K*(K-1)//2+K
        ZCS=APR.CS[K]
        DDX=APR.CS[K]*np.sqrt(gauss.B[KK])
        CXX=APR.CS[K]*(1.+gauss.DE[K])
        DDXD=DDX*100./CXX
        format115 = "(2I6,4F10.4)"
        fort_write(file_IO4, format115, [K,NSETN[LLX],CXX,DDX,DDXD,ZCS])
        APR.CS[K]=CXX
    label .lbl82  # end for loop

    format5115 = "(2I6,2D20.12,F10.4)"
    label .lbl292
    #
    #     reset for repeat of fit with replaced apriori from first fit
    #
    if MODAP == 0 or MODREP == MODAP:
        goto .lbl64
    MODREP=MODREP+1
    NTOT=0
    SIGMA2=0.
    NSHP=0
    for L in fort_range(1,LDB):  # .lbl72
        gauss.DE[L]=0.
        gauss.BM[L]=0.

    for L in fort_range(1,LDBB2):  # .lbl73
        gauss.B[L]=0.
    
    format130 = "(A4)"
    for L in fort_range(1,2000):  # .lbl69
        DUM = fort_read(file_IO3, format130)[0]
        if DUM == LABL.AKON[4]:
            ID, N, NADD = read_block_input(data, gauss, LDA, LDB, KA, KAS, MODREP, file_IO4)
            goto .lbl50
        label .lbl69
    label .lbl64

    #
    #   OUTPUT OF CORRELATION MATRIX OF THE RESULT
    #
    if IPP[6] == 0:
        goto .lbl184

    for K in fort_range(1,NC):  # .lbl78
        J1=APR.MCS[K,2]
        J2=APR.MCS[K,3]

        for L in fort_range(1,K):  # .lbl80
            format122 = "(1H1, '  CORRELATION MATRIX OF THE RESULT   ',2A8,2A8///)"
            fort_write(file_IO4, format122, [LABL.CLAB[K,1:3], LABL.CLAB[L,1:3]])
            J3=APR.MCS[L,2]
            J4=APR.MCS[L,3]
            if K == L:
                goto .lbl87

            for I in fort_range(J1, J2):  # .lbl88
                II=I*(I-1)//2+I
                for J in fort_range(J3, J4):  # .lbl16
                    IJ=I*(I-1)//2+J
                    JJ=J*(J-1)//2+J
                    gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                label .lbl16  # end loop
                fort_write(file_IO4, format151, [gauss.BM[J3:(J4+1)]]) 
            label .lbl88  # end loop
            goto .lbl300

            label .lbl87
            for I in fort_range(J1, J2):  # .lbl55
                II=I*(I-1)//2+I
                for J in fort_range(J1,I):  # .lbl27
                    IJ=I*(I-1)//2+J
                    JJ=J*(J-1)//2+J
                    gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                label .lbl27
                label .lbl55
                fort_write(file_IO4, format151, [gauss.BM[J1:(I+1)]])

            label .lbl300
            label .lbl80
        L = L + 1  # to match L value of fortran after loop
    label .lbl78
    label .lbl184
    exit()

    #
    #      INPUT OF FISSION SPECTRUM
    #
    label .lbl8
    if MC1 == 0:
        goto .lbl692
    format119 = "(2E13.5)"
    for K in fort_range(1,LDF):  # .lbl690
        data.ENFIS[K], data.FIS[K] = fort_read(file_IO3, format119)
        if data.ENFIS[K] == 0:
            goto .lbl691
    K = K + 1  # to match K value of fortran after loop
    label .lbl690
    label .lbl691
    NFIS = K - 1
    goto .lbl38

    # 
    #       MAXWELLIAN SPECTRUM
    # 
    label .lbl692
    EAVR=MC2/1000.
    NFIS=APR.MCS[MC3,1]
    JA=APR.MCS[MC3, 2]
    JE=APR.MCS[MC3, 3]
    LL=0
    for K in fort_range(JA, JE):  # .lbl693
        LL=LL+1
    label .lbl693
    data.ENFIS[LL] = APR.EN[K]
    NFIS1=NFIS-1
    FISUM=0.
    for K in fort_range(1, NFIS1):  # .lbl695
        E1=(data.ENFIS[K-1]+data.ENFIS[K])/2.
        E2=(data.ENFIS[K+1]+data.ENFIS[K])/2.
        DE12=E2-E1
        F1=np.sqrt(E1)*np.exp(-1.5*E1/EAVR)
        F2=np.sqrt(E2)*np.exp(-1.5*E2/EAVR)
        E12=(E1+E2)/2.
        F3=np.sqrt(E12)*np.exp(-1.5*E12/EAVR)
        data.FIS[K]=((F1+F2)*.5+F3)*.5
        FISUM=FISUM+DE12*data.FIS[K]
    label .lbl695

    data.FIS[1]=np.sqrt(APR.EN[JA])*np.exp(-1.5*APR.EN[JA]/EAVR)
    data.FIS[NFIS]=np.sqrt(APR.EN[JE])*np.exp(-1.5*APR.EN[JE]/EAVR)
    DE12=(data.ENFIS[2]-data.ENFIS[1])/2.
    DE13=data.ENFIS[1]+DE12
    FISUM=FISUM+data.FIS[1]*DE13
    DE14=(data.ENFIS(NFIS)-data.ENFIS[NFIS1])/2.
    FISUM=FISUM+data.FIS[NFIS]*2.*DE14
    label .lbl38
    format800 = "(/' FISSION SPECTRUM * BIN WIDTH'/)"
    fort_write(file_IO4, format800, [])
    if MC1 != 0:
        goto .lbl189
    for K in fort_range(2, NFIS1):  # .lbl696
        E1=(data.ENFIS[K-1]+data.ENFIS[K])/2.
        E2=(data.ENFIS[K+1]+data.ENFIS[K])/2.
        DE12=E2-E1
    label .lbl696
    data.FIS[K]=data.FIS[K]*DE12/FISUM
    data.FIS[NFIS]=data.FIS[NFIS]*DE14/FISUM
    data.FIS[1]=data.FIS[1]*DE13/FISUM
    label .lbl189
    format157 = "(2F10.6)"
    for KQ in fort_range(1,NFIS):  # .lbl694
        fort_write(file_IO4, format157, [data.ENFIS[KQ], data.FIS[KQ]])

    goto .lbl50

    #
    #   MODE DEFINITION
    #
    label .lbl9
    MODC = MC1
    MOD2 = MC2
    #VPBEG MPPP=1 allows to use anti-PPP option, when errors of exp data 
    #VP   are taken as % uncertainties from true (posterior) evaluation 
    MPPP = MC5
    #VPEND
    AMO3=MC3/10.
    MODAP=MC4
    if MC2 != 10:
        goto .lbl50
    #
    #      test option:  input of data set numbers which are to be downweighted
    #
    K1=1
    K2=16
    format677 = "(16I5)"
    for K in fort_range(1,10):  # .lbl678
        fort_write(file_IO3, format677, [NRED[K1:(K2+1)]])
        if NRED[K2] == 0:
            goto .lbl679
        K1=K1+16
        K2=K2+16
    label .lbl678
    label .lbl679
    for K in fort_range(K1,K2):  # .lbl680
        if NRED[K] == 0:
            goto .lbl681
    label .lbl680
    label .lbl681
    NELI=K-1
    goto .lbl50
    #
    #      MESSAGE
    #
    label .lbl76
    format701 = "( '   OVERFLOW OF UNKNOWN-VECTOR SPACE WITH SET  ',I3)"
    fort_write(file_IO4, format701, [MC1])
    exit()
    #
    #      ATTEMPT TO MAKE CORR. MATRIX POSITIVE DEFINITE
    #
    label .lbl31
    format105 = "(/' EXP BLOCK CORREL. MATRIX NOT PD',20X,'***** WARNING *')" 
    fort_write(file_IO4, format105, [])
    IREP=IREP+1
    N1=N-1
    for K in fort_range(1,N1):  # .lbl2211
        K1=K+1
        for L in fort_range(K1, N):  # .lbl2211
            if MODC == 2:
                data.ECOR[L,K] = 0.
            data.ECOR[K,L] = data.ECOR[L,K]
    label .lbl2211
    for K in fort_range(1,N):  # .lbl2212
        data.ECOR[K,K] = 1.

    CXZ=0.10
    for K in fort_range(1,N):  # .lbl37
        for L in fort_range(1,N):
            data.ECOR[K,L]=data.ECOR[K,L]/(1.+CXZ)
            if K == L:
                data.ECOR[K,L] = 1.
    label .lbl37
    if IREP < 15:
        goto .lbl67

    goto .lbl50

    #
    #      MESSAGES
    #
    label .lbl40
    format105 = "(/' EXP BLOCK CORREL. MATRIX NOT PD',20X,'***** WARNING *')" 
    format106 = "( '  SOLUTION  CORREL. MATRIX NOT PD ' )"
    fort_write(file_IO4, format106)

    file_IO3.close()
    file_IO4.close()
    file_IO5.close()
    return None


main()
