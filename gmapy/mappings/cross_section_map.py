import numpy as np
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
)


class CrossSectionMap:

    def __init__(self, datatable, selcol=None, distsum=None):
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        inp, out = self.__prepare(datatable, selcol)
        self.__input = inp
        self.__output = out
        if distsum is not None:
            distsum.add_distributors(self.__output.get_distributors())

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
        return self.__input.get_selectors()

    def get_distributors(self):
        return self.__output.get_distributors()

    def __prepare(self, datatable, selcol):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priortable = datatable[priormask]
        expmask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if not np.any(expmask):
            return inp, out

        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'].str.fullmatch(curreac, na=False)]
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index

            inpvar = selcol.define_selector(idcs1red, len(datatable))
            intres = LinearInterpolation(inpvar, ens1, ens2, zero_outside=True)
            outvar = Distributor(intres, idcs2red, len(datatable))
            inp.add_selector(inpvar)
            out.add_distributor(outvar)

        return inp, out
