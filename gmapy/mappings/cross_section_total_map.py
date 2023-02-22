import numpy as np
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation
)
from .helperfuns import return_matrix_new


class CrossSectionTotalMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+', na=False) &
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
            reac_groups = curreac.split('-')[1:]
            reacids = [int(x.split(':')[1]) for x in reac_groups]
            reacstrs = ['MT:1-R1:' + str(rid) for rid in reacids]
            if len(np.unique(reacstrs)) < len(reacstrs):
                   raise IndexError('Each reaction must occur only once in reaction string')
            # retrieve the relevant reactions in the prior
            priortable_reds = [priortable[priortable['REAC'].str.fullmatch(r, na=False)] for r in reacstrs]
            # retrieve relevant rows in exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # some abbreviations
            src_idcs_list = [pt.index for pt in priortable_reds]
            src_en_list = [pt['ENERGY'] for pt in priortable_reds]
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']

            cvars = [Selector(idcs, len(datatable)) for idcs in src_idcs_list]
            inpvars.extend(cvars)
            cvars_int = []
            for cv, en in zip(cvars, src_en_list):
                cvars_int.append(LinearInterpolation(cv, en, tar_en))

            tmpres = sum(cvars_int)
            outvar = Distributor(tmpres, tar_idcs, len(datatable))
            outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        inp.assign(refvals)

        if what == 'propagate':
            return out.evaluate()
        elif what == 'jacobian':
            return out.jacobian()
        else:
            raise ValueError(f'what "{what}" not implemented"')
