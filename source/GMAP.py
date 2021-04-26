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
from gmap_functions import (force_stop, read_prior, read_block_input,
        read_dataset_input, accounting, should_exclude_dataset,
        construct_Ecor, determine_apriori_norm_shape,
        fill_AA_AM_COV, complete_symmetric_Ecor, output_Ecor_matrix,
        invert_Ecor, get_matrix_products, get_result, output_result)


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
        'FCFC': np.zeros((10+1, 10+1), dtype=float),
        'AAA': np.zeros((250+1, 250+1), dtype=float)
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
            'B': np.zeros(720600+1, dtype=float),
            'EGR': np.zeros(199+1, dtype=float),
            'EEGR': np.zeros(400+1, dtype=float),
            'RELTRG': np.zeros((200+1, 200+1), dtype=float),
            'RELCOV': np.zeros((200+1, 200+1), dtype=float)
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
                force_stop(file_IO4)
            if K == 11:
                # input:  data set numbers which are to be excluded from the evaluation
                IELIM = MC1
                format171 = r"(16I5)"
                NELIM = fort_read(file_IO3, format171)
                goto .lbl50

    # end loop: .lbl10

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

    label .lbl2309

    (NSHP, L, AP) = \
    determine_apriori_norm_shape(data, APR, KAS, LABL, NSETN,
            MT, L, NSHP, MTTP, MPPP, IPP, NS, NR, NALT, NADD1,
            MODREP, LDB, MC1, NCT, file_IO4)

    label .lbl28

    N = fill_AA_AM_COV(data, APR, gauss, AP, KAS, KA, N, L, EAVR, NT, NCT, MT, NALT, NADD1, file_IO4)
    goto .lbl50

    #
    #    Data BLOCK complete
    #
    label .lbl7
    N1=N-1
    if ID == 0:
        goto .lbl50

    IREP = 0
    complete_symmetric_Ecor(data, MODC, N, N1, file_IO4)

    if not (IPP[3] == 0 or N == 1 or MODC == 2):
        output_Ecor_matrix(data, N, file_IO4)

    if not (MODC == 2 or N == 1):
        invertible, IREP = invert_Ecor(data, N, IPP, MODC, IREP, file_IO4)
        if not invertible:
            goto .lbl50

    
    NRS, NTOT, SIGMA2 = get_matrix_products(gauss, data, N, LDA, LDB, MODREP,
            NR, NSHP, KA, NTOT, SIGMA2, file_IO4)
    goto .lbl50

    label .lbl3
    get_result(gauss, SIGMA2, NTOT, NRS, IPP, LDB, file_IO3, file_IO4)

    JA = output_result(gauss, data, APR, MODAP, NFIS, NR, NC,
            NSHP, NRS, LABL, NSETN, file_IO4, file_IO5)


    format5115 = "(2I6,2D20.12,F10.4)"
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

    format151 = "(1X,24F7.4)"
    for K in fort_range(1,NC):  # .lbl78
        J1=APR.MCS[K,2]
        J2=APR.MCS[K,3]

        # CVP 3 lines below are added by VP, 26 July, 2004
        NROW=J2-J1+2
        for III in fort_range(1, NROW):
            gauss.EGR[III] = 1.0*III
        # CVP

        for L in fort_range(1,K):  # .lbl80
            format122 = "(1H1, '  CORRELATION MATRIX OF THE RESULT   ',2A8,2A8///)"
            fort_write(file_IO4, format122, [LABL.CLAB[K,1:3], LABL.CLAB[L,1:3]])
            J3=APR.MCS[L,2]
            J4=APR.MCS[L,3]

            # CVP 3 lines below are added by VP, 26 July 2004
            NCOL = J4-J3+2
            for III in fort_range(1, NROW+NCOL):
                gauss.EEGR[III] = 1.0*III
            # CVP

            if K == L:
                goto .lbl87

            for I in fort_range(J1, J2):  # .lbl88
                II=I*(I-1)//2+I
                for J in fort_range(J3, J4):  # .lbl16
                    IJ=I*(I-1)//2+J
                    JJ=J*(J-1)//2+J
                    gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                    # CVP three lines below are inserted by VP
                    gauss.RELCOV[I-J1+1, J-J3+1] = gauss.B[IJ]
                    data.AAA[I-J1+1, J-J3+1] = gauss.BM[J]
                    data.AAA[J-J3+1, I-J1+1] = gauss.BM[J]
                    # CVP
                label .lbl16  # end loop
                fort_write(file_IO4, format151, [gauss.BM[J3:(J4+1)]]) 
            label .lbl88  # end loop

            # CVP   Lines below are added by VP, 26 July, 2004
            format388 = '(6E11.4)'
            fort_write(file_IO4, format388,
                    [gauss.EEGR[1:(NROW+NCOL+1)],
                        gauss.RELCOV[1:NROW, 1:NCOL].flatten()])
            fort_write(file_IO4, format388,
                    [gauss.EEGR[1:(NROW+NCOL+1)],
                     gauss.RELCOV[1:NROW, 1:NCOL].flatten(order='F')])
            # CVP   print below is inserted by VP Aug2013
            IMAX = J2-J1+1
            format389 = '(2x,f7.3,1x,200(E10.4,1x))'
            for I in fort_range(1, IMAX):
                fort_write(file_IO4, format389,
                        [APR.EN[JA+I-1],
                         data.AAA[I,1:(J4-J3+2)]])

            goto .lbl300

            label .lbl87
            for I in fort_range(J1, J2):  # .lbl55
                II=I*(I-1)//2+I
                for J in fort_range(J1,I):  # .lbl27
                    IJ=I*(I-1)//2+J
                    JJ=J*(J-1)//2+J
                    gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                    # CVP lines below are added by VP, 26 July, 2004
                    gauss.RELTRG[I-J1+1,J-J1+1] = gauss.B[IJ]
                    data.AAA[I-J1+1, J-J1+1] = gauss.BM[J]
                    data.AAA[J-J1+1, I-J1+1] = gauss.BM[J]
                    # CVP end

                label .lbl27
                label .lbl55
                fort_write(file_IO4, format151, [gauss.BM[J1:(I+1)]])

            format389 = '(2x,f7.3,1x,200(E10.4,1x))'
            IMAX = J2-J1+1
            for I in fort_range(1,IMAX):
                fort_write(file_IO4, format389,
                        [APR.EN[JA+I-1], data.AAA[I,1:(J2-J1+2)]])

            # CVP   Lines below are added by VP, 26 July, 2004
            format388 = '(6E11.4)'
            tmp = [[gauss.RELTRG[III,JJJ]
                    for III in fort_range(JJJ,NROW-1)]
                    for JJJ in fort_range(1, NROW-1)]
            fort_write(file_IO4, format388,
                    [gauss.EGR[1:(NROW+1)], tmp])
            # CVP

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


main()
