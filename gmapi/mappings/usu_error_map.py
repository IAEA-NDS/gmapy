import numpy as np
from .basic_maps import basic_propagate, get_basic_sensmat
from .helperfuns import return_matrix



class USUErrorMap:

    def __init__(self, compmap, feature_columns, NA_values=('NA', np.isnan)):
        self.compmap = compmap
        self.feature_columns = feature_columns
        self.NA_values = NA_values


    def is_responsible(self, datatable):
        feat_columns = self.feature_columns
        na_vals = self.NA_values
        is_featured_point = np.full(len(datatable), False)
        for feat_col in feat_columns:
            tar_feat = datatable[feat_col].to_numpy()
            cur_feat_mask = np.isin(tar_feat, na_vals, invert=True)
            is_featured_point = np.logical_or(is_featured_point, cur_feat_mask)

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
        feat_columns = self.feature_columns
        na_vals = self.NA_values

        priormask = datatable['NODE'].str.match('usu_', na=False)
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]

        base_propvals = compmap.propagate(datatable, refvals)
        if what == 'propagate':
            propvals = base_propvals.copy()
        elif what == 'jacobian':
            base_S = compmap.jacobian(datatable, refvals, ret_mat=False)
            idcs1_list = [base_S['idcs1']]
            idcs2_list = [base_S['idcs2']]
            coeffs_list = [base_S['x']]
        else:
            raise ValueError('what must be either "propagate" or "jacobian"')

        for feat_col in feat_columns:
            usu_feat = priortable[feat_col].to_numpy()
            exp_feat = exptable[feat_col].to_numpy()

            # np.isin does not handle np.nan in the intended way
            is_nonna_src = np.array([x not in na_vals for x in usu_feat])
            is_nonna_tar = np.array([x not in na_vals for x in exp_feat])
            usu_nonna_feat = usu_feat[is_nonna_src]
            exp_nonna_feat = exp_feat[is_nonna_tar]
            sort_idcs = usu_nonna_feat.argsort()
            src_idcs = sort_idcs[np.searchsorted(usu_nonna_feat, exp_nonna_feat, sorter=sort_idcs)]
            glob_src_idcs = priortable.index[is_nonna_src][src_idcs]
            glob_tar_idcs = exptable.index[is_nonna_tar]

            if what == 'propagate':
                propvals[glob_tar_idcs] += base_propvals[glob_tar_idcs] * refvals[glob_src_idcs]
            elif what == 'jacobian':
                glob_sensvals = base_propvals[glob_tar_idcs]
                idcs1_list.append(glob_src_idcs)
                idcs2_list.append(glob_tar_idcs)
                coeffs_list.append(glob_sensvals)

        if what == 'propagate':
            return propvals
        elif what == 'jacobian':
            idcs1 = np.concatenate(idcs1_list)
            idcs2 = np.concatenate(idcs2_list)
            coeffs = np.concatenate(coeffs_list)
            return idcs1, idcs2, coeffs

