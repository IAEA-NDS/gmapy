import numpy as np
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Distributor,
    LinearInterpolation
)
from .helperfuns import return_matrix_new


class CrossSectionAbsoluteRatioMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:7-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+', na=False) &
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
            # obtian the involved reactions
            string_groups = curreac.split('-')
            reac1id = int(string_groups[1].split(':')[1])
            reac2id = int(string_groups[2].split(':')[1])
            reac3id = int(string_groups[3].split(':')[1])
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            reac3str = 'MT:1-R1:' + str(reac3id)
            if (reac1str == reac2str or
                reac2str == reac3str or
                reac3str == reac1str):
                   raise IndexError('all three reactions in a/(b+c) must be different')
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(reac1str, na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(reac2str, na=False)]
            priortable_red3 = priortable[priortable['REAC'].str.fullmatch(reac3str, na=False)]
            # and in the exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_idcs3 = priortable_red3.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            src_en3 = priortable_red3['ENERGY']
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']

            inpvar1 = Selector(src_idcs1, len(datatable))
            inpvar2 = Selector(src_idcs2, len(datatable))
            inpvar3 = Selector(src_idcs3, len(datatable))
            inpvar1_int = LinearInterpolation(inpvar1, src_en1, tar_en)
            inpvar2_int = LinearInterpolation(inpvar2, src_en2, tar_en)
            inpvar3_int = LinearInterpolation(inpvar3, src_en3, tar_en)
            tmpres = inpvar1_int / (inpvar2_int + inpvar3_int)
            outvar = Distributor(tmpres, tar_idcs, len(datatable))

            inpvars.extend([inpvar1, inpvar2, inpvar3])
            outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = sum(outvars)
        inp.assign(refvals)

        if what == 'jacobian':
            return out.jacobian()
        elif what == 'propagate':
            return out.evaluate()