import numpy as np
import pandas as pd
import json
from scipy.sparse import coo_matrix, csr_matrix

from .inference import gls_update
from .data_management.tablefuns import (create_prior_table, create_experiment_table)
from .data_management.uncfuns import (create_relunc_vector, create_experimental_covmat)
from .mappings.priortools import (attach_shape_prior, update_dummy_datapoints,
        calculate_PPP_correction)
from .mappings.compound_map import CompoundMap

from .data_management.database_IO import (read_legacy_gma_database,
        read_json_gma_database)


def run_gmap_simplified(prior_list=None, datablock_list=None,
        dbfile=None, dbtype='legacy', num_iter=3, correct_ppp=True):

    compmap = CompoundMap(fix_sacs_jacobian=True,
                          legacy_integration=False)

    if dbfile is not None:
        if dbtype == 'legacy':
            db_dic = read_legacy_gma_database(dbfile)
        elif dbtype == 'json':
            db_dic = read_json_gma_database(dbfile)
        else:
            raise ValueError('dbtype must be "legacy" or "json"')

        if prior_list is None:
            prior_list = db_dic['prior_list']

        if datablock_list is None:
            datablock_list = db_dic['datablock_list']

    if prior_list is None:
        raise TypeError('you must provide prior_list or a dbfile')
    if datablock_list is None:
        raise TypeError('you must provide datablock_list or a dbfile')

    priortable = create_prior_table(prior_list)
    exptable = create_experiment_table(datablock_list)

    datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
    refvals = datatable['PRIOR'].to_numpy()

    uncs = np.full(len(refvals), np.nan)
    expsel = datatable['NODE'].str.match('exp_').to_numpy()
    uncs[expsel] = create_relunc_vector(datablock_list)
    datatable = attach_shape_prior(datatable, compmap, refvals, uncs)

    MODREP = 0
    expsel = datatable['NODE'].str.match('exp_').to_numpy()
    exp_idcs = datatable.index[expsel].to_numpy()
    nonexp_idcs = datatable.index[np.logical_not(expsel)]
    uncs = np.full(len(datatable), 0.)
    uncs[exp_idcs] = create_relunc_vector(datablock_list)

    orig_priorvals = datatable['PRIOR'].to_numpy().copy()
    while True:

        refvals = datatable['PRIOR'].to_numpy()
        propvals = compmap.propagate(datatable, refvals)
        update_dummy_datapoints(datatable, propvals)
        if correct_ppp:
            effuncs = calculate_PPP_correction(datatable, compmap, refvals, uncs)
        else:
            effuncs = uncs.copy()

        expdata = datatable['DATA'].to_numpy()
        # construct covariance matrix
        uncs_red = uncs[expsel]
        effuncs_red = effuncs[expsel]
        expdata_red = expdata[expsel]
        tmp = create_experimental_covmat(datablock_list, expdata_red, uncs_red,
                                         effuncs_red)
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

        datatable.loc[prior_idcs, 'PRIOR'] = upd_vals

        if (num_iter == 0 or MODREP == num_iter):
            break

        MODREP=MODREP+1

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

