import numpy as np
import pandas as pd
import json
from scipy.sparse import coo_matrix, csr_matrix

# BEGIN LEGACY
from ..legacy.inference import (link_prior_and_datablocks, update_prior_estimates,
        update_prior_shape_estimates, add_compinfo_to_datablock)
from ..legacy.output_management import (write_prior_info, write_iteration_info,
        write_GMA_header, write_fission_spectrum, output_result_correlation_matrix,
        create_gauss_structure)
from ..legacy.conversion_utils import (sanitize_datablock, desanitize_datablock,
        augment_datablocks_with_NTOT, sanitize_prior, desanitize_prior)
from ..legacy.database_reading import read_gma_database
from ..legacy.data_management import init_labels
from ..legacy.data_extraction_functions import update_effDCS_values
# END LEGACY

from ..inference import gls_update
from ..data_management.tablefuns import (create_prior_table, create_experiment_table)
from ..data_management.uncfuns import (create_relunc_vector, create_experimental_covmat)
from ..mappings.priortools import (attach_shape_prior, update_dummy_datapoints,
        calculate_PPP_correction)
from ..mappings.compound_map import CompoundMap

# DEBUG
from ..mappings.priortools import calculate_PPP_correction2, propagate_mesh_css
from ..mappings.priortools import update_dummy_datapoints2


#################################################
#   START OF GMAP PROGRAM
##################################################

def run_gmap(dbfile='data.gma', resfile='gma.res', plotfile='plot.dta',
        dbtype='legacy', num_iter=3, correct_ppp=True, legacy_output=False,
        fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=True,
        format_dic={}):

    # BEGIN LEGACY
    if legacy_output:
        file_IO4 = open(resfile, 'w')
        file_IO5 = open(plotfile, 'w')
    # END LEGACY

    compmap = CompoundMap(fix_sacs_jacobian=fix_sacs_jacobian,
                          legacy_integration=legacy_integration)

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


    datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
    refvals = datatable['PRIOR'].to_numpy()

    uncs = np.full(len(refvals), np.nan)
    expsel = datatable['NODE'].str.match('exp_').to_numpy()
    uncs[expsel] = create_relunc_vector(new_datablock_list)
    datatable = attach_shape_prior(datatable, compmap, refvals, uncs)

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
    if legacy_output:
        link_prior_and_datablocks(APR, datablock_list)
        write_GMA_header(file_IO4)
        write_fission_spectrum(APR.fisdata, file_IO4)
        write_prior_info(APR, IPP, file_IO4)
    # END LEGACY

    MODREP = 0
    expsel = datatable['NODE'].str.match('exp_').to_numpy()
    exp_idcs = datatable.index[expsel].to_numpy()
    nonexp_idcs = datatable.index[np.logical_not(expsel)]
    uncs = np.full(len(datatable), 0.)
    uncs[exp_idcs] = create_relunc_vector(new_datablock_list)

    orig_priorvals = datatable['PRIOR'].to_numpy().copy()
    while True:

        refvals = datatable['PRIOR'].to_numpy()
        propvals = compmap.propagate(datatable, refvals)
        update_dummy_datapoints(datatable, propvals)
        # We also need to update the datablock list
        # because we are preparing the code to do the
        # PPP correction in create_experimental_covmat
        # in a better (=less convoluted) way
        update_dummy_datapoints2(new_datablock_list, propvals[expsel])

        if correct_ppp:
            effuncs, tmppropvals = calculate_PPP_correction(datatable, compmap, refvals, uncs)
        else:
            effuncs = uncs.copy()

        expdata = datatable['DATA'].to_numpy()
        # construct covariance matrix
        uncs_red = uncs[expsel]
        effuncs_red = effuncs[expsel]
        expdata_red = expdata[expsel]
        # DEBUG
        propcss = propagate_mesh_css(datatable, compmap, refvals)
        propcss_red = propcss[expsel]
        assert np.all(propcss_red == tmppropvals[expsel])

        tmpeffuncs = calculate_PPP_correction2(new_datablock_list, propcss_red, uncs_red)
        # DEBUG
        assert np.all(np.isclose(effuncs[expsel], tmpeffuncs))

        tmp = create_experimental_covmat(new_datablock_list, expdata_red, propcss_red, uncs_red,
                                         effuncs_red, fix_ppp_bug=fix_ppp_bug)
        tmp = coo_matrix(tmp)
        covmat = csr_matrix((tmp.data, (exp_idcs[tmp.row], exp_idcs[tmp.col])),
                               shape=(len(datatable), len(datatable)),
                               dtype=float)
        del(tmp)
        # add prior uncertainties
        prioruncs = datatable.loc[nonexp_idcs, 'UNC'].to_numpy()
        covmat += csr_matrix((np.square(prioruncs), (nonexp_idcs, nonexp_idcs)),
                             shape=(len(datatable), len(datatable)), dtype=float)

        upd_res = gls_update(compmap, datatable, covmat, retcov=True)
        prior_idcs = upd_res['idcs']
        upd_vals = upd_res['upd_vals']
        upd_covmat = upd_res['upd_covmat']

        # BEGIN LEGACY
        if legacy_output:
            fismask = datatable['NODE'] == 'fis'
            MPPP = 1 if correct_ppp else 0

            for datablock in datablock_list:
                add_compinfo_to_datablock(datablock, APR, MPPP)

            update_effDCS_values(datablock_list, effuncs_red)
            gauss = create_gauss_structure(APR, datablock_list,
                    upd_vals, upd_covmat)

            LABL = init_labels()
            write_iteration_info(APR, datablock_list, gauss,
                    datatable, compmap,
                    MODREP, num_iter, MPPP, IPP, LABL, file_IO4, file_IO5)

            if num_iter != 0:
                update_prior_estimates(APR, upd_vals)

            update_prior_shape_estimates(APR, upd_vals)
        # END LEGACY

        datatable.loc[prior_idcs, 'PRIOR'] = upd_vals

        if (num_iter == 0 or MODREP == num_iter):
            break

        MODREP=MODREP+1

    # BEGIN LEGACY
    if legacy_output:
        output_result_correlation_matrix(gauss, datablock_list[-1], APR, IPP, file_IO4)
        file_IO4.close()
        file_IO5.close()
    # END LEGACY

    datatable['PRIOR'] = orig_priorvals

    postvals = datatable['PRIOR'].to_numpy()
    postvals[prior_idcs] = upd_vals
    datatable['DATAUNC'] = np.sqrt(covmat.diagonal())
    datatable['POST'] = compmap.propagate(datatable, postvals)
    datatable['POSTUNC'] = np.full(len(datatable), np.NaN, dtype=float)
    datatable.loc[prior_idcs, 'POSTUNC'] = np.sqrt(np.diag(upd_covmat))
    datatable['RELPOSTUNC'] = datatable['POSTUNC'].to_numpy() / datatable['POST'].to_numpy()
    return {'table': datatable, 'postcov': upd_covmat, 'idcs': prior_idcs,
            'priorcov': covmat}

