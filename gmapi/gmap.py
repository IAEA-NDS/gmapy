import numpy as np
import argparse
import json

# BEGIN LEGACY
from .legacy.inference import (link_prior_and_datablocks, update_prior_estimates,
        update_prior_shape_estimates, add_compinfo_to_datablock)
from .legacy.output_management import (write_prior_info, write_iteration_info,
        write_GMA_header, write_fission_spectrum, output_result_correlation_matrix,
        create_gauss_structure)
from .legacy.conversion_utils import (sanitize_datablock, desanitize_datablock,
        augment_datablocks_with_NTOT, sanitize_prior, desanitize_prior)
from .legacy.database_reading import read_gma_database
from .legacy.data_management import init_labels
from .legacy.data_extraction_functions import update_effDCS_values
# END LEGACY

from .inference import gls_update
from .data_management.tablefuns import (create_prior_table, create_experiment_table)
from .data_management.uncfuns import (create_relunc_vector, create_experimental_covmat)
from .mappings.priortools import (attach_shape_prior, update_dummy_datapoints,
        calculate_PPP_correction)
from .mappings.compound_map import CompoundMap


#################################################
#   START OF GMAP PROGRAM
##################################################


def run_gmap(dbfile='data.gma', resfile='gma.res', plotfile='plot.dta',
        dbtype='legacy', num_iter=3, correct_ppp=True, format_dic={}):

    # BEGIN LEGACY
    file_IO4 = open(resfile, 'w')
    file_IO5 = open(plotfile, 'w')
    # END LEGACY

    compmap = CompoundMap()

    if dbtype == 'legacy':
        db_dic = read_gma_database(dbfile, format_dic=format_dic)

        APR = db_dic['APR']
        datablock_list = db_dic['datablock_list']
        # MPPP flag no longer read from file but
        # provided as parameter correct_ppp
        # MPPP = db_dic['MPPP']
        IPP = db_dic['IPP']
        # MODAP no longer read from file but
        # provided as parameter num_iter
        # MODAP = db_dic['MODAP']
        # calculate new structures
        new_prior_list = sanitize_prior(APR)
        new_datablock_list = [sanitize_datablock(b) for b in datablock_list]
        priortable = create_prior_table(new_prior_list)
        exptable = create_experiment_table(new_datablock_list)

    elif dbtype == 'json':
        # Hard-coded variables that would be
        # read from the legacy database format.
        # As they will not be part of the new database
        # because they merely represent output options
        # and parameters for the fitting, they are
        # hardcoded here.
        IPP = [None, 1, 1, 1, 0, 0, 1, 0, 1]  # output options
        with open(dbfile, 'r') as f:
            db_dic = json.load(f)

        new_prior_list = db_dic['prior']
        new_datablock_list = db_dic['datablocks']
        priortable = create_prior_table(new_prior_list)
        exptable = create_experiment_table(new_datablock_list)
        # convert to legacy quantities
        datablock_list = [desanitize_datablock(b) for b in new_datablock_list]
        augment_datablocks_with_NTOT(datablock_list)
        APR = desanitize_prior(new_prior_list)

    else:
        raise ValueError('dbtype must be "legacy" or "json"')

    refvals = priortable['PRIOR'].to_numpy()
    uncs = create_relunc_vector(new_datablock_list)
    priortable = attach_shape_prior(priortable, exptable, refvals, uncs)

    # NOTE: The code enclosed by LEGACY is just there
    #       to create the output as produced by
    #       Fortran GMAP for the sake of comparison
    #       with the legacy code. All calculations
    #       are performed using new routines, which
    #       do neither rely on legacy data structures
    #       (e.g., APR) nor legacy functions operating
    #       on those structures. The results of the
    #       new routines is introduced appropriately
    #       in the legacy data structures before
    #       printing to the file.

    # BEGIN LEGACY
    link_prior_and_datablocks(APR, datablock_list)
    write_GMA_header(file_IO4)
    write_fission_spectrum(APR.fisdata, file_IO4)
    write_prior_info(APR, IPP, file_IO4)
    # END LEGACY

    MODREP = 0
    while True:

        refvals = priortable['PRIOR'].to_numpy()
        propvals = compmap.propagate(priortable, exptable, refvals)
        update_dummy_datapoints(exptable, propvals)
        uncs = create_relunc_vector(new_datablock_list)
        if correct_ppp:
            effuncs = calculate_PPP_correction(priortable, exptable, refvals, uncs)
        else:
            effuncs = uncs.copy()
        expdata = exptable['DATA'].to_numpy()
        expcovmat = create_experimental_covmat(new_datablock_list, expdata, uncs, effuncs)

        upd_res = gls_update(priortable, exptable, expcovmat, retcov=True)
        upd_vals = upd_res['upd_vals']
        upd_covmat = upd_res['upd_covmat']


        # BEGIN LEGACY
        fismask = priortable['NODE'] == 'fis'
        invfismask = np.logical_not(fismask)
        red_upd_covmat = upd_covmat[np.ix_(invfismask, invfismask)]
        red_upd_vals = upd_vals[invfismask]
        MPPP = 1 if correct_ppp else 0

        for datablock in datablock_list:
            add_compinfo_to_datablock(datablock, APR, MPPP)

        update_effDCS_values(datablock_list, effuncs)
        gauss = create_gauss_structure(APR, datablock_list,
                red_upd_vals, red_upd_covmat)

        LABL = init_labels()
        write_iteration_info(APR, datablock_list, gauss,
                priortable, exptable,
                MODREP, num_iter, MPPP, IPP, LABL, file_IO4, file_IO5)

        if num_iter != 0:
            update_prior_estimates(APR, red_upd_vals)

        update_prior_shape_estimates(APR, red_upd_vals)
        # END LEGACY

        priortable['PRIOR'] = upd_vals

        if (num_iter == 0 or MODREP == num_iter):
            break

        MODREP=MODREP+1

    # BEGIN LEGACY
    output_result_correlation_matrix(gauss, datablock_list[-1], APR, IPP, file_IO4)
    # END LEGACY

