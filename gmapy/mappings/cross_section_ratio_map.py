import numpy as np
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
    reuse_or_create_input_selector
)


class CrossSectionRatioMap:

    def __init__(self, datatable, selector_list=None):
        self.__numrows = len(datatable)
        self.__input, self.__output = self.__prepare(datatable, selector_list)

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

    def __prepare(self, datatable, selector_list):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))

        priortable = datatable[priormask]
        expmask = np.array(
            datatable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
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
            string_groups = curreac.split('-')
            reac1id = int(string_groups[1].split(':')[1])
            reac2id = int(string_groups[2].split(':')[1])
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(reac1str, na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(reac2str, na=False)]
            # and in the exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']

            inpvar1 = reuse_or_create_input_selector(
                src_idcs1, len(datatable), selector_list
            )
            inpvar2 = reuse_or_create_input_selector(
                src_idcs2, len(datatable), selector_list
            )
            inpvar1_int = LinearInterpolation(inpvar1, src_en1, tar_en)
            inpvar2_int = LinearInterpolation(inpvar2, src_en2, tar_en)
            tmpres = inpvar1_int / inpvar2_int
            outvar = Distributor(tmpres, tar_idcs, len(datatable))
            inpvars.extend([inpvar1, inpvar2])
            outvars.append(outvar)

        inp = InputSelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
