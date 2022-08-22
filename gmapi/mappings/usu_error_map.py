import numpy as np
import pandas as pd
from sksparse.cholmod import cholesky
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


    def propagate(self, datatable, refvals, only_usu=False):
        propvals = self.__compute(datatable, refvals, 'propagate', only_usu)
        return propvals


    def jacobian(self, datatable, refvals, ret_mat=False, only_usu=False):
        num_points = datatable.shape[0]
        idcs1, idcs2, coeffs = self.__compute(datatable, refvals, 'jacobian', only_usu)
        return return_matrix(idcs1, idcs2, coeffs,
                  dims = (num_points, num_points),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, datatable, refvals, what, only_usu=False):
        compmap = self.compmap
        feat_columns = self.feature_columns
        na_vals = self.NA_values

        priormask = datatable['NODE'].str.match('usu_', na=False)
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]

        # if only_usu is false, we also include the mapping component
        # associated with contribution of other (i.e. non-USU)
        # quantities to the prediction of experimental datasets
        base_propvals = compmap.propagate(datatable, refvals)
        if what == 'propagate':
            if not only_usu:
                propvals = base_propvals.copy()
            else:
                propvals = np.full(len(base_propvals), 0., dtype='d')
        elif what == 'jacobian':
            if not only_usu:
                base_S = compmap.jacobian(datatable, refvals, ret_mat=False)
                idcs1_list = [base_S['idcs1']]
                idcs2_list = [base_S['idcs2']]
                coeffs_list = [base_S['x']]
            else:
                idcs1_list = []
                idcs2_list = []
                coeffs_list = []
        else:
            raise ValueError('what must be either "propagate" or "jacobian"')

        for feat_col in feat_columns:
            usu_feat = priortable[feat_col]
            exp_feat = exptable[feat_col]
            is_nonna_src = np.logical_not(usu_feat.isin(na_vals))
            is_nonna_tar = np.logical_not(exp_feat.isin(na_vals))
            # cast usu_feat and exp_usu feat to avoid surprises with indexing
            usu_nonna_feat = usu_feat[is_nonna_src].to_numpy()
            exp_nonna_feat = exp_feat[is_nonna_tar].to_numpy()
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

    # additional functions to obtain gradients with respect
    # to the USU uncertainties

    # mapings interface can rely on datatable (but not on 'data' or 'covmat'
    # which quantities do I need? datatable, refvals, expvals, covmat

    def _prepare_auxiliary_usu_info(self, datatable, refvals, covmat):
        usu_idcs = datatable.index[datatable.NODE.str.match('usu_')]
        exp_idcs = datatable.index[datatable.NODE.str.match('exp_')]
        Susu = self.jacobian(datatable, refvals, ret_mat=True, only_usu=True)
        Susu = Susu[exp_idcs,:][:,usu_idcs].tocsc()
        # the .tocsc() addition to avoid an efficiency warning
        # during the Cholesky decomposition
        expcov = covmat[exp_idcs,:][:,exp_idcs].tocsc()
        usucov = covmat[usu_idcs,:][:,usu_idcs].tocsc()
        expcov_fact = cholesky(expcov)
        usucov_fact = cholesky(usucov)
        usu_aux = {
            'refvals': refvals,
            'usu_idcs': usu_idcs,
            'exp_idcs': exp_idcs,
            'Susu': Susu,
            'expcov_fact': expcov_fact,
            'usucov_fact': usucov_fact,
                }
        return usu_aux


    def logdet(self, datatable, refvals, covmat):
        usu_aux = self._prepare_auxiliary_usu_info(datatable, refvals, covmat)
        Susu = usu_aux['Susu']
        usucov_fact = usu_aux['usucov_fact']
        expcov_fact = usu_aux['expcov_fact']
        Z = usucov_fact.inv() + Susu.T @ expcov_fact(Susu)
        Z_fact = cholesky(Z)
        res = expcov_fact.logdet() + usucov_fact.logdet() + Z_fact.logdet()
        return res


    def chisquare(self, datatable, refvals, expvals, covmat):
        usu_aux = self._prepare_auxiliary_usu_info(datatable, refvals, covmat)
        usu_idcs = usu_aux['usu_idcs']
        exp_idcs = usu_aux['exp_idcs']
        Susu = usu_aux['Susu']
        usucov_fact = usu_aux['usucov_fact']
        expcov_fact = usu_aux['expcov_fact']
        refvals = np.copy(refvals)
        # calculate the difference between predictions and experiments
        # (we force the USU error to be zero; however, should this be a user choice?)
        refvals[usu_idcs] = 0.
        preds = self.propagate(datatable, refvals)
        preds = preds[exp_idcs]
        real_expvals = expvals[exp_idcs]
        d = real_expvals - preds
        z0 = expcov_fact(d)
        first_term = d.T @ z0
        z1 = Susu.T @ z0
        zc = usucov_fact.inv() + Susu.T @ expcov_fact(Susu)
        zc_fact = cholesky(zc)
        z2 = zc_fact(z1)
        second_term = z1.T @ z2
        return first_term - second_term

