import numpy as np
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Distributor,
    LinearInterpolation
)
from .helperfuns import return_matrix_new


class CrossSectionMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)

    def propagate(self, datatable, refvals):
        return self.__compute(datatable, refvals, 'propagate')

    def jacobian(self, datatable, refvals, ret_mat=False):
        S = self.__compute(datatable, refvals, 'jacobian')
        return return_matrix_new(S, how='csr' if ret_mat else 'dic')

    def __compute(self, datatable, refvals, what):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
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

            inpvar = Selector(idcs1red, len(datatable))
            intres = LinearInterpolation(inpvar, ens1, ens2, zero_outside=True)
            outvar = Distributor(intres, idcs2red, len(datatable))
            inpvars.append(inpvar)
            outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = sum(outvars)
        inp.assign(refvals)

        if what == 'propagate':
            return out.evaluate()
        elif what == 'jacobian':
            return out.jacobian()
        else:
            raise ValueError(f'what "{what}" not implemented"')
