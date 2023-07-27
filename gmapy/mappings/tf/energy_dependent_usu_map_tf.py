import numpy as np
import pandas as pd
from .cross_section_modifier_base_map_tf import CrossSectionModifierBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class EnergyDependentUSUMap(CrossSectionModifierBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            (datatable['NODE'].str.match('exp_', na=False)).any() &
            (datatable['NODE'].str.match('endep_usu_', na=False)).any()
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = priortable['NODE'].str.match('endep_usu_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]
        # establish the mapping for each USU component
        # associated with an experimental dataset
        rerr_expids = priortable['NODE'].str.extract(r'endep_usu_([0-9]+)$')
        rerr_expids = pd.unique(rerr_expids.iloc[:, 0])
        for expid in rerr_expids:
            srcnode = 'endep_usu_' + expid
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
        def _atomic_propagate(propvals, inpvar):
            relerrs = PiecewiseLinearInterpolation(src_ens, tar_ens)(inpvar)
            abserrs = relerrs * propvals
            return abserrs
        return _atomic_propagate
