import numpy as np
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class CrossSectionMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            datatable['REAC'].str.match('MT:1-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priortable = priortable[priormask]
        expmask = (exptable['REAC'].str.match('MT:1-R1:', na=False) &
                   exptable['NODE'].str.match('exp_', na=False))

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[
                priortable['REAC'].str.fullmatch(curreac, na=False)
            ]
            exptable_red = exptable[
                exptable['REAC'].str.fullmatch(curreac, na=False)
            ]
            # abbreviate some variables
            ens1 = np.array(priortable_red['ENERGY'])
            idcs1red = np.array(priortable_red.index)
            ens2 = np.array(exptable_red['ENERGY'])
            idcs2red = np.array(exptable_red.index)
            propfun = self._generate_atomic_propagate(ens1, ens2)
            self._add_lists((idcs1red,), idcs2red, propfun)

    def _generate_atomic_propagate(self, ens1, ens2):
        def _atomic_propagate(inpvar):
            intres = PiecewiseLinearInterpolation(ens1, ens2)(inpvar)
            return intres
        return _atomic_propagate
