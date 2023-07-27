import numpy as np
from .cross_section_base_map import CrossSectionBaseMap
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
)


class CrossSectionMap(CrossSectionBaseMap):

    def _prepare(self, priortable, exptable, selcol):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priortable = priortable[priormask]
        expmask = (exptable['REAC'].str.match('MT:1-R1:', na=False) &
                   exptable['NODE'].str.match('exp_', na=False))

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if not np.any(expmask):
            return inp, out

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'].str.fullmatch(curreac, na=False)]
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index

            inpvar = selcol.define_selector(idcs1red, self._src_len)
            intres = LinearInterpolation(inpvar, ens1, ens2, zero_outside=True)
            outvar = Distributor(intres, idcs2red, self._tar_len)
            inp.add_selector(inpvar)
            out.add_distributor(outvar)

        return inp, out
