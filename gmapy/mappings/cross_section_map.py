import numpy as np
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
    reuse_or_create_input_selector
)


class CrossSectionMap:

    def __init__(self, datatable, selcol=None):
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        selcol = selcol.get_selectors()
        inp, out = self.__prepare(datatable, selcol)
        self.__input = inp
        self.__output = out

    def is_responsible(self):
        ret = np.full(self.__numrows, False)
        if self.__output is not None:
            idcs = self.__output.get_indices()
            ret[idcs] = True
        return ret

    def propagate(self, refvals):
        self.__input.assign(refvals)
        return self.__output.evaluate()

    def jacobian(self, refvals):
        self.__input.assign(refvals)
        return self.__output.jacobian()

    def get_selectors(self):
        if self.__input is not None:
            return self.__input.get_selectors()
        else:
            return []

    def get_distributors(self):
        if self.__output is not None:
            return self.__output.get_distributors()
        else:
            return []

    def __prepare(self, datatable, selcol):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priortable = datatable[priormask]
        expmask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        if not np.any(expmask):
            return None, None
        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        inpvars = []
        outvars = []
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'].str.fullmatch(curreac, na=False)]
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index

            inpvar = reuse_or_create_input_selector(
                idcs1red, len(datatable), selcol
            )
            intres = LinearInterpolation(inpvar, ens1, ens2, zero_outside=True)
            outvar = Distributor(intres, idcs2red, len(datatable))
            inpvars.append(inpvar)
            outvars.append(outvar)

        inp = InputSelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
