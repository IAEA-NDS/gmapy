from inference import (link_prior_and_datablocks, update_prior_estimates,
        update_prior_shape_estimates, add_compinfo_to_datablock)

from inference_new import new_gls_update

from output_management import ( write_prior_info, write_iteration_info,
        write_GMA_header, write_fission_spectrum, output_result_correlation_matrix,
        create_gauss_structure)

from database_reading import read_gma_database
from data_management import init_gauss
from data_extraction_functions import (extract_covariance_matrix,
        extract_prior_table, extract_experimental_table)


#################################################
#   START OF GMAP PROGRAM
##################################################


def run_GMA_program(dbfile='data.gma', resfile='gma.res', plotfile='plot.dta',
        format_dic={}):

    file_IO4 = open(resfile, 'w')
    file_IO5 = open(plotfile, 'w')

    db_dic = read_gma_database(dbfile, format_dic=format_dic)

    APR = db_dic['APR']
    datablock_list = db_dic['datablock_list']
    LABL = db_dic['LABL']
    MPPP = db_dic['MPPP']
    IPP = db_dic['IPP']
    MODAP = db_dic['MODAP']

    link_prior_and_datablocks(APR, datablock_list)

    write_GMA_header(file_IO4)
    write_fission_spectrum(APR.fisdata, file_IO4)
    write_prior_info(APR, IPP, file_IO4)

    MODREP = 0
    while True:

        for datablock in datablock_list:
            add_compinfo_to_datablock(datablock, APR, MPPP)

        priortable = extract_prior_table(APR)
        exptable = extract_experimental_table(datablock_list)
        expcovmat = extract_covariance_matrix(datablock_list)

        upd_res = new_gls_update(priortable, exptable, expcovmat, retcov=True)
        upd_vals = upd_res['upd_vals']
        upd_covmat = upd_res['upd_covmat']

        gauss = create_gauss_structure(APR, datablock_list, upd_vals, upd_covmat)

        write_iteration_info(APR, datablock_list, gauss,
                MODREP, MODAP, MPPP, IPP, LABL, file_IO4, file_IO5)

        if MODAP != 0:
            update_prior_estimates(APR, upd_vals)

        update_prior_shape_estimates(APR, upd_vals)

        if (MODAP == 0 or MODREP == MODAP):
            break

        MODREP=MODREP+1

    output_result_correlation_matrix(gauss, datablock_list[-1], APR, IPP, file_IO4)



if __name__ == '__main__':
    run_GMA_program()

