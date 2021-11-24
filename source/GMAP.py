# helpful for fortran to python conversion
from generic_utils import Bunch, unflatten
from fortran_utils import fort_range, fort_read, fort_write

# refactoring
from gmap_functions import (link_prior_and_datablocks,
        update_prior_estimates, update_prior_shape_estimates,
        add_compinfo_to_datablock, gls_update)

from output_management import ( write_prior_info, write_iteration_info,
        write_GMA_header, write_fission_spectrum, output_result_correlation_matrix)

from database_reading import read_gma_database


#################################################
#   START OF GMAP PROGRAM
##################################################



def run_GMA_program():

    file_IO4 = open('gma.res', 'w')
    file_IO5 = open('plot.dta', 'w')

    db_dic = read_gma_database('data.gma')

    APR = db_dic['APR']
    datablock_list = db_dic['datablock_list']
    fisdata = db_dic['fisdata']
    LABL = db_dic['LABL']
    MPPP = db_dic['MPPP']
    IPP = db_dic['IPP']
    MODAP = db_dic['MODAP']

    link_prior_and_datablocks(APR, datablock_list)

    write_GMA_header(file_IO4)

    write_fission_spectrum(fisdata, file_IO4)

    write_prior_info(APR, IPP, file_IO4)

    MODREP = 0
    while True:

        for datablock in datablock_list:
            add_compinfo_to_datablock(datablock, fisdata, APR, MPPP)

        gauss = gls_update(datablock_list, APR)

        write_iteration_info(APR, datablock_list, fisdata, gauss,
                MODREP, MODAP, MPPP, IPP, LABL, file_IO4, file_IO5)

        if MODAP != 0:
            update_prior_estimates(APR, gauss)

        update_prior_shape_estimates(APR, gauss)

        #
        #     reset for repeat of fit with replaced apriori from first fit
        #
        if (MODAP == 0 or MODREP == MODAP):
            break

        MODREP=MODREP+1


    output_result_correlation_matrix(gauss, datablock_list[-1], APR, IPP, file_IO4)



run_GMA_program()

