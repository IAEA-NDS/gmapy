import numpy as np
import pandas as pd
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class EnergyDependentAbsoluteUSUMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            (datatable['NODE'].str.match('exp_', na=False)).any() &
            (datatable['NODE'].str.match('endep_abs_usu_', na=False)).any()
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = priortable['NODE'].str.match('endep_abs_usu_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]
        # establish the mapping for each USU component
        # associated with an experimental dataset
        rerr_expids = priortable['NODE'].str.extract(r'endep_abs_usu_([0-9]+)$')
        rerr_expids = pd.unique(rerr_expids.iloc[:, 0])
        for expid in rerr_expids:
            srcnode = 'endep_abs_usu_' + expid
            srcdt = priortable[priortable['NODE'] == srcnode]
            tarnode = 'exp_' + expid
            tardt = exptable[exptable['NODE'] == tarnode]
            src_idcs = np.array(srcdt.index)
            tar_idcs = np.array(tardt.index)
            src_ens = srcdt['ENERGY'].to_numpy()
            tar_ens = tardt['ENERGY'].to_numpy()
            propfun = self._generate_atomic_propagate(src_ens, tar_ens)
            self._add_lists(
                [src_idcs], tar_idcs, propfun
            )

    def _generate_atomic_propagate(self, src_ens, tar_ens):
        def _atomic_propagate(inpvar):
            abserrs = PiecewiseLinearInterpolation(src_ens, tar_ens)(inpvar)
            return abserrs
        return _atomic_propagate


def create_endep_abs_usu_df(datatable, reacs, energies, uncs):
    dt = datatable
    red_dt = dt[dt.NODE.str.match('exp_[0-9]+') & dt.REAC.isin(reacs)]
    expids = pd.unique(red_dt.NODE.str.extract(r'exp_([0-9]+)$').iloc[:,0])
    dt_list = []
    for expid in expids:
        expname = 'exp_' + expid
        curdt = red_dt[red_dt.NODE == expname]
        curnode = 'endep_abs_usu_' + expid
        curreac = curdt.REAC.iat[0]
        new_usu_dt = pd.DataFrame.from_dict({
            'NODE': curnode,
            'REAC': curreac,
            'ENERGY': energies,
            'PRIOR': 0.0,
            'DATA': np.nan,
            'UNC': uncs
        })
        dt_list.append(new_usu_dt)
    res_dt = pd.concat(dt_list, axis=0, ignore_index=True)
    return res_dt


def attach_endep_abs_usu_df(datatable, reacs, energies, uncs):
    dt = datatable.copy()
    dt.sort_index(inplace=True, axis=0)
    usu_dt = create_endep_abs_usu_df(dt, reacs, energies, uncs)
    res_dt = pd.concat([dt, usu_dt], axis=0, ignore_index=True)
    return res_dt
