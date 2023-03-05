import pandas as pd
import numpy as np
from .mappings.usu_error_map import USUErrorMap
from .gma_database_class import GMADatabase
import scipy.sparse as spsp
from scipy.optimize import minimize
from warnings import warn


class GMADatabaseUSU(GMADatabase):

    def __init__(self, dbfile, remove_dummy=True, mapping=None):
        super().__init__(dbfile, remove_dummy=remove_dummy, mapping=mapping)
        self._base_mapping = self._mapping
        self._usu_coupling_column = None
        self._usu_coupling_mapping = None


    def set_usu_components(self, feature_columns, usu_names=None,
                         NA_values=('NA', np.nan)):
        if usu_names is not None and len(usu_names) != len(feature_columns):
            raise IndexError('length of names of USU components ' +
                    'must match the number of given features')
        # check that all feature columns exist in datatable
        dt = self.get_datatable()
        if not np.all(np.isin(feature_columns, dt.columns)):
            raise IndexError('not all specified column names exist')
        # we delete all previously defined USU components
        usu_idcs = dt.index[dt.NODE.str.match('usu_')]
        if len(usu_idcs) > 0:
            self.remove_data(usu_idcs)
            dt = self.get_datatable()

        # construct the extra dataframes
        dt.sort_index(inplace=True)
        dt_list = [dt]
        for i, feat_col in enumerate(feature_columns):
            is_not_na = np.logical_not(dt[feat_col].isin(NA_values))
            is_exp = dt.NODE.str.match('exp_')
            is_good = np.logical_and(is_not_na, is_exp)
            uniq_feat_attrs = np.sort(pd.unique(dt.loc[is_good, feat_col]))
            if usu_names is None:
                cur_usu_name = 'usu_' + str(feat_col)
            else:
                cur_usu_name = 'usu_' + usu_names[i]
            new_usu_dt = pd.DataFrame.from_dict({
                'NODE': cur_usu_name,
                'PRIOR': 0.,
                'UNC': 0.01,
                feat_col: uniq_feat_attrs
                })
            dt_list.append(new_usu_dt)
        new_dt = pd.concat(dt_list, ignore_index=True)
        self._datatable = new_dt

        # extend the covariance matrix
        usuuncs = new_dt.UNC[new_dt.NODE.str.match('usu_')]
        usuuncs = usuuncs.to_numpy()
        usucov = spsp.diags(np.square(usuuncs))
        covmat = self._covmat
        self._covmat = spsp.block_diag([covmat, usucov],
                format='csr')
        self._cache['uncertainties'] =  np.concatenate(
                [self._cache['uncertainties'], usuuncs])

        # update the mapping
        self._mapping = USUErrorMap(self._base_mapping, feature_columns,
                                    NA_values=NA_values)

    def set_usu_couplings(self, coupling_column):
        dt = self._datatable
        if coupling_column not in dt.columns:
            raise IndexError(f'column {coupling_column} not found in datatable')
        usu_sel = dt.NODE.str.match('usu_', na=False)
        if np.sum(usu_sel) == 0:
            raise IndexError('no USU components defined! ' +
                    'Please call set_usu_components to do so.')
        usu_couplings = dt.loc[usu_sel, coupling_column]
        if np.any(usu_couplings.isna()):
            raise ValueError('NA values not allowed for USU couplings')
        usu_unc_names = pd.unique(usu_couplings)
        usu_couplings = usu_couplings.to_numpy()
        col_idcs_list = []
        row_idcs_list = []
        val_list = []
        for row_idx, cur_unc_name in enumerate(usu_unc_names):
            cur_row_idcs = np.where(cur_unc_name == usu_couplings)[0]
            cur_col_idcs = np.full(len(cur_row_idcs), row_idx)
            cur_vals = np.full(len(cur_row_idcs), 1.)
            col_idcs_list.append(cur_col_idcs)
            row_idcs_list.append(cur_row_idcs)
            val_list.append(cur_vals)

        row_idcs = np.concatenate(row_idcs_list)
        col_idcs = np.concatenate(col_idcs_list)
        vals = np.concatenate(val_list)
        S = spsp.csr_matrix((vals, (row_idcs, col_idcs)))
        self._usu_coupling_mapping = S
        self._usu_coupling_column = coupling_column

    def evaluate(self, remove_idcs=None, adjust_usu=True,
            outer_iter=50, inner_iter=1, atol=1e-6, rtol=1e-6, print_status=False, **lm_options):
        if adjust_usu and remove_idcs is not None:
            raise ValueError('dynamic removal of experimental data ' +
                            'and USU adjustment not allowed at the same time')
        dt = self._datatable

        if not adjust_usu:
            inner_iter = outer_iter * inner_iter
            outer_iter = 1

        lm_options['maxiter'] = inner_iter
        lm_options['atol'] = atol
        lm_options['rtol'] = rtol
        lm_options['print_status'] = print_status

        if 'POST' in dt.columns:
            old_postvals = dt['POST'].to_numpy()
        else:
            old_postvals = dt['PRIOR'].to_numpy()
        lm_options['startvals'] = old_postvals
        lm_options['must_converge'] = False

        converged = False
        for i in range(outer_iter):
            if print_status:
                print(f'##############################')
                print(f'Outer iteration nr. {i}')
                print('Estimate posterior values and associated uncertainties...')
            super().evaluate(remove_idcs, **lm_options)
            # for convergence diagnostics
            new_postvals = self._datatable['POST'].to_numpy()
            # keep track of the step size control parameter
            # and the use as new startvals in next iteration
            # the posterior values of the current iteration
            lm_options['lmb'] = self._cache['lmb']
            lm_options['startvals'] = new_postvals
            if adjust_usu:
                if print_status:
                    print('Estimate USU uncertainties...')
                old_usu_uncs = dt.UNC[dt.NODE.str.match('usu_')]
                self.determine_usu_uncertainties()
                new_usu_uncs = dt.UNC[dt.NODE.str.match('usu_')]

            if print_status:
                absdiff_postvals = np.abs(new_postvals - old_postvals)
                reldiff_postvals = absdiff_postvals / (atol + old_postvals)
                max_reldiff_postvals = np.nanmax(reldiff_postvals)
                print( 'Maximum relative change in:')
                print(f'    posterior estimates: {max_reldiff_postvals}')
                if adjust_usu:
                    absdiff_usu_uncs = np.abs(new_usu_uncs - old_usu_uncs)
                    reldiff_usu_uncs = absdiff_usu_uncs / (atol + old_usu_uncs)
                    max_reldiff_usu_uncs = np.nanmax(reldiff_usu_uncs)
                    print(f'    USU uncertainties: {max_reldiff_usu_uncs}\n')

            last_rejected = self._cache['last_rejected']
            # we access the convergence flag of the LM algorithm
            if self._cache['converged']:
                if not adjust_usu or np.allclose(new_usu_uncs, old_usu_uncs,
                                                 atol=atol, rtol=rtol, equal_nan=True):
                    converged = True
                    break

            old_postvals = new_postvals.copy()
            if adjust_usu:
                old_usu_uncs = new_usu_uncs.copy()

        if not converged:
            warn(f'The estimates did not converge within the desired accuracy. '
                     'You may want to rerun this function with relaxed '
                     'numbers for atol and rtol or more iterations. '
                     'It may also suffice to rerun this function with '
                     'the same parameter specifications as the results '
                     'of this run will be used as starting values for '
                     'the next run.')

    def determine_usu_uncertainties(self, refvals=None):
        if self._usu_coupling_column is None:
            raise IndexError('no uncertainty coupling specified! ' +
                   'Please call set_usu_uncertainty_coupling first')

        dt = self._datatable
        covmat = self._covmat
        mapping = self._mapping
        expvals = dt['DATA'].to_numpy()
        Scoup = self._usu_coupling_mapping
        usu_idcs = dt.index[dt.NODE.str.match('usu_', na=False)]
        usu_unc_assoc = dt.loc[usu_idcs, self._usu_coupling_column]
        usu_unc_groups = usu_unc_assoc.unique()

        if refvals is None:
            if 'POST' in dt.columns:
                refvals = dt.POST.to_numpy()
            else:
                refvals = dt.PRIOR.to_numpy()

        for usu_unc_group in usu_unc_groups:
            err_sel = (usu_unc_assoc == usu_unc_group)
            numel = err_sel.sum()
            cur_usu_idcs = usu_idcs[err_sel]
            cur_errs = refvals[cur_usu_idcs]
            cur_var = 1/numel * np.sum(np.square(cur_errs))
            dt.loc[cur_usu_idcs, 'UNC'] = np.sqrt(cur_var)
            covmat[cur_usu_idcs, cur_usu_idcs] = cur_var
            self._cache['uncertainties'] = dt['UNC'].to_numpy()
