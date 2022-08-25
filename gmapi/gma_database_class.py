import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from .data_management.database_IO import read_gma_database
from .data_management.tablefuns import create_prior_table, create_experiment_table
from .data_management.uncfuns import create_relunc_vector, create_experimental_covmat
from .mappings.compound_map import CompoundMap
from .inference import lm_update
from .mappings.priortools import (propagate_mesh_css,
        attach_shape_prior, remove_dummy_datasets)


class GMADatabase:

    def __init__(self, dbfile, remove_dummy=True, mapping=None):
        db = read_gma_database(dbfile)
        if remove_dummy:
            remove_dummy_datasets(db['datablock_list'])

        priortable = create_prior_table(db['prior_list'])
        exptable = create_experiment_table(db['datablock_list'])
        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        if not mapping:
            mapping = CompoundMap()
        # initialize the normalization errors
        refvals = datatable['PRIOR']
        reluncs = np.full(len(refvals), np.nan)
        expsel = datatable['NODE'].str.match('exp_', na=False).to_numpy()
        reluncs[expsel] = create_relunc_vector(db['datablock_list'])
        datatable = attach_shape_prior(datatable, mapping, refvals, reluncs)
        # define the state variables of the instance
        self._cache = {}
        self._raw_database = db
        self._datatable = datatable
        self._covmat = None
        self._mapping = mapping
        self._initialize_uncertainty_info()


    def _initialize_uncertainty_info(self):
        datatable = self._datatable
        db = self._raw_database
        datatable.sort_index(inplace=True)
        expsel = datatable['NODE'].str.match('exp_', na=False).to_numpy()
        nonexpsel = np.logical_not(expsel)
        # assemble covariance matrix
        expcov = create_experimental_covmat(db['datablock_list'])
        expcov = coo_matrix(expcov)
        prioruncs = datatable.loc[nonexpsel, 'UNC'].to_numpy()
        priorvars = np.square(prioruncs)
        prior_idcs = datatable.index[nonexpsel]
        exp_idcs = datatable.index[expsel]
        row_idcs = np.concatenate([prior_idcs, exp_idcs[expcov.row]])
        col_idcs = np.concatenate([prior_idcs, exp_idcs[expcov.col]])
        elems = np.concatenate([priorvars, expcov.data])
        covmat = csr_matrix((elems, (row_idcs, col_idcs)),
                shape=(len(datatable), len(datatable)), dtype='d')
        # update uncertainties in datatable
        datatable.loc[expsel, 'UNC'] = np.sqrt(covmat.diagonal()[expsel])
        # update class state
        self._covmat = covmat
        # self._datatable = ... not necessary because inplace change
        self._cache['uncertainties'] = datatable.UNC.to_numpy()


    def _update_covmat(self):
        db = self._raw_database
        datatable = self._datatable
        covmat = self._covmat
        datatable.sort_index(inplace=True)
        uncs = datatable.UNC.to_numpy()
        olduncs = self._cache['uncertainties']
        unc_not_changed = np.isclose(uncs, olduncs, equal_nan=True)
        unc_changed = np.logical_not(unc_not_changed)
        static_idcs = datatable.index[unc_not_changed]
        changed_idcs = datatable.index[unc_changed]
        # preserve the static covariance matrix block
        static_cov = covmat[static_idcs,:][:,static_idcs]
        static_cov = coo_matrix(static_cov)
        static_row_idcs = static_idcs[static_cov.row]
        static_col_idcs = static_idcs[static_cov.col]
        # replace the covmat block where uncertainties changed
        # by a diagonal matrix with the new squared uncertainties
        new_priorvars = np.square(uncs[unc_changed])
        row_idcs = np.concatenate([static_row_idcs, changed_idcs])
        col_idcs = np.concatenate([static_col_idcs, changed_idcs])
        elems = np.concatenate([static_cov.data, new_priorvars])
        covmat = csr_matrix((elems, (row_idcs, col_idcs)),
                shape=(len(datatable), len(datatable)), dtype='d')
        self._covmat = covmat
        self._cache['uncertainties'] = uncs


    def evaluate(self, remove_idcs=None, **kwargs):
        mapping = self._mapping
        if remove_idcs is None:
            datatable = self._datatable
            covmat = self._covmat
        if remove_idcs is not None:
            keep_mask = np.full(len(self._datatable), True)
            keep_mask[remove_idcs] = False
            datatable, covmat, orig_idcs = \
                    self._remove_data_internal(self._datatable,
                            self._covmat, remove_idcs)
            startvals = kwargs.get('startvals', None)
            startvals = startvals[keep_mask] if startvals is not None else None
            kwargs['startvals'] = startvals

        lmres = lm_update(mapping, datatable, covmat, **kwargs)
        self._cache['lmb'] = lmres['lmb']
        self._cache['last_rejected'] = lmres['last_rejected']
        self._cache['converged'] = lmres['converged']

        if remove_idcs is None:
            adj_idcs = lmres['idcs']
        else:
            datatable = self._datatable
            adj_idcs = orig_idcs[lmres['idcs']]

        refvals = np.full(len(datatable), np.nan)
        refvals[adj_idcs] = lmres['upd_vals']

        propvals = propagate_mesh_css(datatable, mapping, refvals,
                                      prop_normfact=False, prop_usu_errors=False)
        self._datatable['POST'] = propvals
        return propvals


    def _remove_data_internal(self, datatable, covmat, idcs):
        datatable = datatable.sort_index()
        remove_mask = np.full(len(datatable), False)
        remove_mask[idcs] = True
        inv_remove_mask = np.logical_not(remove_mask)
        # reduce table
        datatable.drop(idcs, inplace=True)
        orig_idcs = datatable.index
        datatable.reset_index(drop=True, inplace=True)
        # reduce covariance matrix
        covmat = covmat[:, inv_remove_mask]
        covmat = covmat[inv_remove_mask,:]
        return datatable, covmat, orig_idcs


    def remove_data(self, idcs):
        res = self._remove_data_internal(self._datatable, self._covmat, idcs)
        self._datatable, self._covmat, _  = res
        return None


    def get_datatable(self):
        return self._datatable.copy()


    def set_datatable(self, datatable):
        if len(datatable) != self._covmat.shape[0]:
            raise IndexError('number of rows of new datatable ' +
                    'does not match dimension of covariance matrix')
        self._datatable = datatable.copy()
        self._datatable.sort_index(inplace=True)
        self._update_covmat()


    def get_covmat(self):
        return self._covmat.copy()


    def set_covmat(self, covmat):
        if len(covmat.shape) != 2:
            raise IndexError('covariance matrix must be 2d array')
        if covmat.shape[0] != covmat.shape[1]:
            raise IndexError('covariance matrix must be square matrix')
        if covmat.shape[0] != len(self._datatable):
            raise IndexError('dimensions of new covariance matrix are ' +
                    'incompatible with the number of rows in datatable')
        # update the uncertainties in table
        covmat = covmat.copy()
        datatable = self._datatable
        datatable.sort_index(inplace=True)
        datatable.UNC = np.sqrt(covmat.diagonal())
        self._covmat = covmat

    def get_mapping(self):
        return self._mapping

