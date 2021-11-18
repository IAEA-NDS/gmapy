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
        complete_symmetric_Ecor,
        invert_Ecor, get_matrix_products, get_result, output_result,
        output_result_correlation_matrix, input_fission_spectrum,
        deal_with_dataset, read_datablock, fill_AA_AM_COV,
        construct_Ecor, init_shape_prior, count_usable_datapoints,
        accounting, apply_PPP_correction, link_prior_and_datablocks,
        update_dummy_dataset, update_prior_estimates, update_prior_shape_estimates,
        add_compinfo_to_datablock)

from output_management import (output_Ecor_matrix,
        write_prior_info,
        write_datablock_header, write_dataset_info,
        write_dataset_exclusion_info, write_missing_dataset_info,
        write_KAS_check, write_overflow_message,
        write_dataset_table, write_fission_average,
        write_invalid_datapoints_info, write_added_points_info,
        write_inv_attempt_info, write_datablock_info,
        write_result_info)

from data_management import init_gauss, init_prior, init_labels, SIZE_LIMITS

from gmap_snippets import TextfileReader, get_num_shapedatasets, get_dataset_range


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
    #      IPP        i/o choices
    #      NRED       data set Nr.s for downweighting
    #      NELIM      data set Nr.s to exclude from evaluation
    #
    #   INTEGER*2 KAS(250,5),NT(5),IDEN(30,8),NSETN(200),IPP(8),
    #  1 NENF(40,10),NETG(11,40),NCSST(10),NEC(2,10,10),NRED(160)
    #  2 ,KA(1200,250),NELIM(40)
    IPP = np.zeros(8+1, dtype=int)
    NRED = np.zeros(160+1, dtype=int)
    NELIM = np.zeros(40+1, dtype=int)

    basedir = '.'
    file_IO3 = TextfileReader('data.gma')
    file_IO4 = open('gma.res', 'w')
    file_IO5 = open('plot.dta', 'w')

    APR = init_prior()
    gauss = init_gauss()
    LABL = init_labels()

    datablock_list = []

    #
    #      INITIALIZE PARAMETERS
    #
    #
    #      DATA DE/1200*0.D0/,BM/1200*0.D0/,B/720600*0.D0/,NRED/160*0/
    #
    MODREP = 0
    MODC = 3
    IELIM = 0
    LLL = 0

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
            read_prior(MC1, MC2, APR, file_IO3)


        # LABL.AKON[8] == 'FIS*'
        elif ACON == LABL.AKON[8]:
            fisdata = input_fission_spectrum(MC1, file_IO3, file_IO4)


        # LABL.AKON[4] == 'BLCK'
        elif ACON == LABL.AKON[4]:

            file_IO3.seek(file_IO3.get_line_nr()-1)

            data = read_datablock(MODC, MOD2, AMO3,
                           IELIM, NELIM, LABL, file_IO3)

            datablock_list.append(data)


        # LABL.AKON[3] == 'END*'
        elif ACON == LABL.AKON[3]:

            link_prior_and_datablocks(APR, datablock_list)

            while True:

                for datablock in datablock_list:
                    add_compinfo_to_datablock(datablock, APR, MPPP)

                gauss.NTOT=0
                gauss = init_gauss()

                for data in datablock_list:
                    invertible = invert_Ecor(data)
                    if not invertible:
                        continue

                    fill_AA_AM_COV(data, fisdata, APR)

                    get_matrix_products(gauss, data, APR)

                get_result(gauss, APR)


                if MODREP == 0:
                    write_prior_info(APR, IPP, file_IO4)

                curNSHP = 0
                totNSHP = APR.NSHP
                for data in datablock_list:
                    curNSHP += get_num_shapedatasets(data)
                    APR.NSHP = curNSHP
                    write_datablock_info(APR, data, MODREP, MPPP, IPP, LABL, file_IO4)
                    APR.NSHP = totNSHP

                write_result_info(APR, gauss, IPP, file_IO4)

                output_result(gauss, fisdata, APR, MODAP,
                              file_IO4, file_IO5)

                if MODAP != 0:
                    update_prior_estimates(APR, gauss)

                update_prior_shape_estimates(APR, gauss)

                #
                #     reset for repeat of fit with replaced apriori from first fit
                #
                if (MODAP == 0 or MODREP == MODAP):
                    break

                MODREP=MODREP+1


            output_result_correlation_matrix(gauss, data, APR, IPP, file_IO4)
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
