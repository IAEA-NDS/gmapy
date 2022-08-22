import pandas as pd
import numpy as np
from .mappings.usu_error_map import USUErrorMap
from .gma_database_class import GMADatabase
import scipy.sparse as spsp


class GMADatabaseUSU(GMADatabase):

    def __init__(self, dbfile, remove_dummy=True, mapping=None):
        super().__init__(dbfile, remove_dummy, mapping)
        self._base_mapping = self._mapping 


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
                'UNC': np.inf,
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

