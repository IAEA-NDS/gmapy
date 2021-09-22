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
from gmap_functions import (force_stop, read_prior, prepare_for_datablock_input,
        complete_symmetric_Ecor, output_Ecor_matrix,
        invert_Ecor, get_matrix_products, get_result, output_result,
        output_result_correlation_matrix, input_fission_spectrum,
        deal_with_dataset)

from data_management import init_gauss, init_prior, init_labels


#################################################
#   START OF GMAP PROGRAM
##################################################

def main():

    # IMPLICIT definitions in original version
    # IMPLICIT REAL*8 (A-H,O-Z)

    # NOTE: This is part of a hack to reproduce a
    #       bug in the Fortran version of GMAP 
    #       with values in ENFF leaking into
    #       the next datablock
    data = None

    #
    #      NSETN      shape data set numbers
    #      IPP        i/o choices
    #      NRED       data set Nr.s for downweighting
    #      NELIM      data set Nr.s to exclude from evaluation
    #
    #   INTEGER*2 KAS(250,5),NT(5),IDEN(30,8),NSETN(200),IPP(8),
    #  1 NENF(40,10),NETG(11,40),NCSST(10),NEC(2,10,10),NRED(160)
    #  2 ,KA(1200,250),NELIM(40)
    NSETN = np.zeros(200+1, dtype=int)
    IPP = np.zeros(8+1, dtype=int)
    NRED = np.zeros(160+1, dtype=int)
    NELIM = np.zeros(40+1, dtype=int)

    basedir = '.'
    file_IO3 = open('data.gma', 'r')
    file_IO4 = open('gma.res', 'w')
    file_IO5 = open('plot.dta', 'w')

    APR = init_prior()
    gauss = init_gauss()
    LABL = init_labels()

    #
    #      INITIALIZE PARAMETERS
    #
    #      NTOT TOTAL NO OF DATA POINTS
    #
    #      DATA DE/1200*0.D0/,BM/1200*0.D0/,B/720600*0.D0/,NRED/160*0/
    #
    MODREP = 0
    NTOT = 0
    SIGMA2 = 0.
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


        # LABL.AKON[9] == 'MODE'
        if ACON == LABL.AKON[9]:
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
            if MC2 == 10:
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


        # LABL.AKON[1] == 'APRI'
        elif ACON == LABL.AKON[1]:
            # INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
            read_prior(MC1, MC2, APR, LABL, IPP, file_IO3, file_IO4)


        # LABL.AKON[8] == 'FIS*'
        elif ACON == LABL.AKON[8]:
            fisdata, NFIS = input_fission_spectrum(MC1, file_IO3, file_IO4)


        # LABL.AKON[4] == 'BLCK'
        elif ACON == LABL.AKON[4]:
            # NOTE: data is provided as argument to reproduce a bug
            #       in the Fortran version that causes values of ENFF
            #       leaking into the next datablock
            data, N, = prepare_for_datablock_input(data, gauss, MODREP, file_IO4)


        # LABL.AKON[2] == 'DATA'
        elif ACON == LABL.AKON[2]:
            (MODC, NSHP, N) = deal_with_dataset(MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
                    data, fisdata, gauss,
                    LABL, APR, IELIM, NELIM, NSETN,
                    MODC, MOD2, MPPP, MODREP, N, NSHP,
                    IPP, file_IO3, file_IO4)


        # LABL.AKON[7] == 'EDBL'
        elif ACON == LABL.AKON[7]:
            #
            #    Data BLOCK complete
            #
            N1=N-1
            if data.num_datasets == 0:
                continue

            IREP = 0
            complete_symmetric_Ecor(data, MODC, N, N1, file_IO4)

            if not (IPP[3] == 0 or N == 1 or MODC == 2):
                output_Ecor_matrix(data, N, file_IO4)

            if not (MODC == 2 or N == 1):
                invertible, IREP = invert_Ecor(data, N, IPP, MODC, IREP, file_IO4)
                if not invertible:
                    continue

            NRS, NTOT, SIGMA2 = get_matrix_products(gauss, data, N, MODREP,
                    APR, NSHP, NTOT, SIGMA2, file_IO4)


        # LABL.AKON[3] == 'END*'
        elif ACON == LABL.AKON[3]:
            get_result(gauss, SIGMA2, NTOT, NRS, IPP, file_IO3, file_IO4)
            JA = output_result(gauss, fisdata, APR, MODAP, NFIS,
                    NSHP, NRS, LABL, NSETN, file_IO4, file_IO5)
            #
            #     reset for repeat of fit with replaced apriori from first fit
            #
            if not (MODAP == 0 or MODREP == MODAP):

                MODREP=MODREP+1
                NTOT=0
                SIGMA2=0.
                NSHP=0
                gauss = init_gauss()

                format130 = "(A4)"
                for L in fort_range(1,2000):  # .lbl69
                    DUM = fort_read(file_IO3, format130)[0]
                    if DUM == LABL.AKON[4]:
                        break

                if DUM == LABL.AKON[4]:
                    data, N = prepare_for_datablock_input(data, gauss, MODREP, file_IO4)
                    continue

            output_result_correlation_matrix(gauss, data, APR, IPP,
                    LABL, JA, file_IO4)
            exit()


        # LABL.AKON[11] == 'ELIM'
        elif ACON == LABL.AKON[11]:
            # input:  data set numbers which are to be excluded from the evaluation
            IELIM = MC1
            format171 = r"(16I5)"
            NELIM = fort_read(file_IO3, format171)


        # LABL.AKON[10] == 'STOP'
        elif ACON == LABL.AKON[10]:
            force_stop(file_IO4)


        elif ACON == LABL.AKON[6]:
            format104 = "(A4,2X,'  CONTROL CODE UNKNOWN')"
            fort_write(file_IO4, format104, [ACON])
            exit()


main()
