import numpy as np
from .basic_maps import basic_propagate, get_basic_sensmat
from .helperfuns import return_matrix



class USUMap:

    def __init__(self, compmap, feature_column, NA_values=('NA', np.isnan)):
        self.compmap = compmap
        self.feature_column = feature_column
        self.NA_values = NA_values


    def is_responsible(self, datatable):
        feat_col = self.feature_column
        na_vals = self.NA_values
        tar_feat = datatable[feat_col].to_numpy()
        is_featured_point = np.logical_not(np.isin(tar_feat, na_vals))
        expmask = np.logical_and(datatable['NODE'].str.match('exp_', na=False),
                is_featured_point)
        return np.array(expmask, dtype=bool)


    def propagate(self, datatable, refvals):
        propvals = self.__compute(datatable, refvals, 'propagate')
        return propvals


    def jacobian(self, datatable, refvals, ret_mat=False):
        num_points = datatable.shape[0]
        idcs1, idcs2, coeffs = self.__compute(datatable, refvals, 'jacobian')
        return return_matrix(idcs1, idcs2, coeffs,
                  dims = (num_points, num_points),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, datatable, refvals, what):
        compmap = self.compmap
        feat_col = self.feature_column
        na_vals = self.NA_values

        priormask = datatable['NODE'].str.match('usu_', na=False)
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]

        usu_feat = priortable[feat_col].to_numpy()
        exp_feat = exptable[feat_col].to_numpy()
        if np.any(np.isin(na_vals, usu_feat)):
            raise ValueError(f'NA values ({na_vals}) not allowed for USU components')
        if np.any(np.isin(na_vals, exp_feat)):
            raise ValueError(f'NA values ({na_vals}) not allowed in experimental feature columns')

        sort_idcs = usu_feat.argsort()
        src_idcs = sort_idcs[np.searchsorted(usu_feat, exp_feat, sorter=sort_idcs)]
        glob_src_idcs = priortable.index[src_idcs]
        glob_tar_idcs = exptable.index

        base_propvals = compmap.propagate(datatable, refvals)

        if what == 'propagate':
            propvals = base_propvals.copy()
            propvals[glob_tar_idcs] = base_propvals[glob_tar_idcs] * (1 + refvals[glob_src_idcs])
            return propvals
        elif what == 'jacobian':
            glob_sensvals = base_propvals[glob_tar_idcs]  
            base_S = compmap.jacobian(datatable, refvals, ret_mat=False)
            idcs1 = np.concatenate([base_S['idcs1'], glob_src_idcs])
            idcs2 = np.concatenate([base_S['idcs2'], glob_tar_idcs])
            data = np.concatenate([base_S['x'], glob_sensvals])
            return idcs1, idcs2, data
        else:
            raise ValueError('what must be either "propagate" or "jacobian"')

