import numpy as np
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
    reuse_or_create_input_selector
)


class CrossSectionTotalMap:

    def __init__(self, datatable, selcol=None):
        self.__numrows = len(datatable)
        self.__input, self.__output = self.__prepare(datatable, selcol)

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
        expmask = np.array(
            datatable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        )
        if not np.any(expmask):
            return None, None
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

            cvars = [
                reuse_or_create_input_selector(idcs, len(datatable), selcol)
                for idcs in src_idcs_list
            ]
            inpvars.extend(cvars)
            cvars_int = []
            for cv, en in zip(cvars, src_en_list):
                cvars_int.append(LinearInterpolation(cv, en, tar_en))

            tmpres = sum(cvars_int)
            outvar = Distributor(tmpres, tar_idcs, len(datatable))
            outvars.append(outvar)

        inp = InputSelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
