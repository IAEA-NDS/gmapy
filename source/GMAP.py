# helpful for fortran to python conversion
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
        invert_Ecor, get_matrix_products, get_result, output_result,
        output_result_correlation_matrix, input_fission_spectrum)


#################################################
#   START OF GMAP PROGRAM
##################################################

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

    while True:

        #
        #      Control parameter input
        #
        format100 = "(A4,1X,8I5)"
        ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8 = fort_read(file_IO3,  format100)

        # LABL.AKON[1] == 'APRI'
        if ACON == LABL.AKON[1]:
            # INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
            NC, NR = read_prior(MC1, MC2, APR, LABL, IPP, file_IO3, file_IO4)


        # LABL.AKON[2] == 'DATA'
        elif ACON == LABL.AKON[2]:

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
                continue

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

            if MT != 6:
                #
                #   output of KAS for checking
                #
                if IPP[7] != 0:
                    format702 = "(20I5)"
                    for K in fort_range(1,NCT):
                        fort_write(file_IO4, format702, [KAS[NALT:(NADD1+1)], K])

                (NSHP, L, AP) = \
                determine_apriori_norm_shape(data, APR, KAS, LABL, NSETN,
                        MT, L, NSHP, MTTP, MPPP, IPP, NS, NR, NALT, NADD1,
                        MODREP, LDB, MC1, NCT, file_IO4)

            N = fill_AA_AM_COV(data, APR, gauss, AP, KAS, KA, N, L, EAVR, NT, NCT, MT, NALT, NADD1, file_IO4)


        # LABL.AKON[3] == 'END*'
        elif ACON == LABL.AKON[3]:
            get_result(gauss, SIGMA2, NTOT, NRS, IPP, LDB, file_IO3, file_IO4)
            JA = output_result(gauss, data, APR, MODAP, NFIS, NR, NC,
                    NSHP, NRS, LABL, NSETN, file_IO4, file_IO5)
            #
            #     reset for repeat of fit with replaced apriori from first fit
            #
            if not (MODAP == 0 or MODREP == MODAP):

                MODREP=MODREP+1
                NTOT=0
                SIGMA2=0.
                NSHP=0
                gauss.DE[0:(LDB+1)] = 0.
                gauss.BM[0:(LDB+1)] = 0.
                gauss.B[0:(LDBB2+1)] = 0.

                format130 = "(A4)"
                for L in fort_range(1,2000):  # .lbl69
                    DUM = fort_read(file_IO3, format130)[0]
                    if DUM == LABL.AKON[4]:
                        break

                if DUM == LABL.AKON[4]:
                    ID, N, NADD = read_block_input(data, gauss, LDA, LDB, KA, KAS, MODREP, file_IO4)
                    continue

            output_result_correlation_matrix(gauss, data, APR, IPP, NC,
                    LABL, JA, file_IO4)
            exit()


        # LABL.AKON[4] == 'BLCK'
        elif ACON == LABL.AKON[4]:
            ID, N, NADD = read_block_input(data, gauss, LDA, LDB, KA, KAS, MODREP, file_IO4)


        # LABL.AKON[5] == 'I/OC'
        elif ACON == LABL.AKON[5]:
            #
            #      I/O CONTROL
            #
            IPP = [None for i in range(9)]
            IPP[1] = MC1
            IPP[2] = MC2
            IPP[3] = MC3
            IPP[4] = MC4
            IPP[5] = MC5
            IPP[6] = MC6
            IPP[7] = MC7
            IPP[8] = MC8


        elif ACON == LABL.AKON[6]:
            format104 = "(A4,2X,'  CONTROL CODE UNKNOWN')"
            fort_write(file_IO4, format104, [ACON])
            exit()


        # LABL.AKON[7] == 'EDBL'
        elif ACON == LABL.AKON[7]:
            #
            #    Data BLOCK complete
            #
            N1=N-1
            if ID == 0:
                continue

            IREP = 0
            complete_symmetric_Ecor(data, MODC, N, N1, file_IO4)

            if not (IPP[3] == 0 or N == 1 or MODC == 2):
                output_Ecor_matrix(data, N, file_IO4)

            if not (MODC == 2 or N == 1):
                invertible, IREP = invert_Ecor(data, N, IPP, MODC, IREP, file_IO4)
                if not invertible:
                    continue

            NRS, NTOT, SIGMA2 = get_matrix_products(gauss, data, N, LDA, LDB, MODREP,
                    NR, NSHP, KA, NTOT, SIGMA2, file_IO4)


        # LABL.AKON[8] == 'FIS*'
        elif ACON == LABL.AKON[8]:
            NFIS = input_fission_spectrum(data, MC1, LDF, file_IO3, file_IO4)


        # LABL.AKON[9] == 'MODE'
        elif ACON == LABL.AKON[9]:
            #
            #   MODE DEFINITION
            #
            MODC = MC1
            MOD2 = MC2
            #VPBEG MPPP=1 allows to use anti-PPP option, when errors of exp data 
            #VP   are taken as % uncertainties from true (posterior) evaluation 
            MPPP = MC5
            #VPEND
            AMO3=MC3/10.
            MODAP=MC4
            if MC2 != 10:
                continue
            #
            #      test option:  input of data set numbers which are to be downweighted
            #
            K1=1
            K2=16
            format677 = "(16I5)"
            for K in fort_range(1,10):  # .lbl678
                fort_write(file_IO3, format677, [NRED[K1:(K2+1)]])
                if NRED[K2] == 0:
                    break
                K1=K1+16
                K2=K2+16

            for K in fort_range(K1,K2):  # .lbl680
                if NRED[K] == 0:
                    break

            NELI=K-1


        # LABL.AKON[10] == 'STOP'
        elif ACON == LABL.AKON[10]:
            force_stop(file_IO4)


        # LABL.AKON[11] == 'ELIM'
        elif ACON == LABL.AKON[11]:
            # input:  data set numbers which are to be excluded from the evaluation
            IELIM = MC1
            format171 = r"(16I5)"
            NELIM = fort_read(file_IO3, format171)



main()
