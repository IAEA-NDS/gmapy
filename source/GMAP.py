import numpy as np

from inference import (link_prior_and_datablocks, update_prior_estimates,
        update_prior_shape_estimates, add_compinfo_to_datablock)

from inference_new import new_gls_update

from output_management import ( write_prior_info, write_iteration_info,
        write_GMA_header, write_fission_spectrum, output_result_correlation_matrix,
        create_gauss_structure)

from database_reading import read_gma_database
from data_management import init_gauss
from data_extraction_functions import (extract_covariance_matrix,
        extract_prior_table, extract_experimental_table, extract_DCS_values,
        update_effDCS_values)

from mappings.priortools import (attach_shape_prior, update_dummy_datapoints,
        calculate_PPP_correction)
from mappings.compound_map import CompoundMap

from conversion_utils import (sanitize_datablock, desanitize_datablock,
        augment_datablocks_with_NTOT, sanitize_fission_spectrum_block,
        desanitize_fission_spectrum_block, sanitize_prior, desanitize_prior)



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

    compmap = CompoundMap()

    # Check to see if new JSON style datablock are really one-to-one
    # mappings to the Fortran GMAP datablock by converting from Fortran
    # datablock to Python datablock format and back
    datablock_list = [sanitize_datablock(b) for b in datablock_list]
    datablock_list = [desanitize_datablock(b) for b in datablock_list]
    augment_datablocks_with_NTOT(datablock_list)

    # Check to see if new JSON style fission spectrum block is really
    # one-to-one mapping of the Fortran GMAP fission spectrum block
    # by converting from Fortran to Python block style and back
    APR.fisdata = desanitize_fission_spectrum_block(sanitize_fission_spectrum_block(APR.fisdata))

    # Check to see if new JSON style prior is really
    # one-to-one mapping of the Fortran GMAP prior
    # by converting from Fortran to Python prior style and back
    APR = desanitize_prior(sanitize_prior(APR))

    priortable = extract_prior_table(APR)
    exptable = extract_experimental_table(datablock_list)

    refvals = priortable['PRIOR'].to_numpy()
    uncvals = extract_DCS_values(datablock_list)
    priortable = attach_shape_prior(priortable, exptable, refvals, uncvals)

    link_prior_and_datablocks(APR, datablock_list)

    write_GMA_header(file_IO4)
    write_fission_spectrum(APR.fisdata, file_IO4)
    write_prior_info(APR, IPP, file_IO4)

    MODREP = 0
    while True:

        for datablock in datablock_list:
            add_compinfo_to_datablock(datablock, APR, MPPP)

        refvals = priortable['PRIOR'].to_numpy()
        propvals = compmap.propagate(priortable, exptable, refvals)
        update_dummy_datapoints(exptable, propvals)

        uncs = extract_DCS_values(datablock_list)
        effuncs = calculate_PPP_correction(priortable, exptable, refvals, uncs)
        update_effDCS_values(datablock_list, effuncs)

        expcovmat = extract_covariance_matrix(datablock_list)

        upd_res = new_gls_update(priortable, exptable, expcovmat, retcov=True)
        upd_vals = upd_res['upd_vals']
        upd_covmat = upd_res['upd_covmat']

        fismask = priortable['NODE'] == 'fis'
        invfismask = np.logical_not(fismask)
        red_upd_covmat = upd_covmat[np.ix_(invfismask, invfismask)]
        red_upd_vals = upd_vals[invfismask]

        gauss = create_gauss_structure(APR, datablock_list,
                red_upd_vals, red_upd_covmat)

        write_iteration_info(APR, datablock_list, gauss,
                priortable, exptable,
                MODREP, MODAP, MPPP, IPP, LABL, file_IO4, file_IO5)

        priortable['PRIOR'] = upd_vals

        if MODAP != 0:
            update_prior_estimates(APR, red_upd_vals)

        update_prior_shape_estimates(APR, red_upd_vals)

        if (MODAP == 0 or MODREP == MODAP):
            break

        MODREP=MODREP+1

    output_result_correlation_matrix(gauss, datablock_list[-1], APR, IPP, file_IO4)



if __name__ == '__main__':
    run_GMA_program()

