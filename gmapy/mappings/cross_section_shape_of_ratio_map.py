import numpy as np
from .mapping_elements import (
    InputSelectorCollection,
    Replicator,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
    reuse_or_create_input_selector
)


class CrossSectionShapeOfRatioMap:

    def __init__(self, datatable, selcol=None):
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        selcol = selcol.get_selectors()
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
        priormask = np.logical_or(priormask, datatable['NODE'].str.match('norm_', na=False))
        priortable = datatable[priormask]
        expmask = np.array(
            datatable['REAC'].str.match('MT:9-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        )
        if not np.any(expmask):
            return None, None
        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        inpvars = []
        outvars = []
        for curreac in reacs:
            # obtain the involved reactions
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
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_idcs3 = priortable_red3.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            src_en3 = priortable_red3['ENERGY']

            inpvar1 = reuse_or_create_input_selector(
                src_idcs1, len(datatable), selcol
            )
            inpvar2 = reuse_or_create_input_selector(
                src_idcs2, len(datatable), selcol
            )
            inpvar3 = reuse_or_create_input_selector(
                src_idcs3, len(datatable), selcol
            )
            inpvars.extend([inpvar1, inpvar2, inpvar3])

            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            datasets = exptable_red['NODE'].unique()
            for ds in datasets:
                # subset another time exptable to get dataset info
                tar_idcs = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)].index
                tar_en = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)]['ENERGY']
                # obtain normalization and position in priortable
                normstr = ds.replace('exp_', 'norm_')
                norm_index = priortable[priortable['NODE'].str.fullmatch(normstr, na=False)].index
                if len(norm_index) != 1:
                    raise IndexError('Exactly one normalization factor must be present for a dataset')

                norm_fact = reuse_or_create_input_selector(
                    norm_index, len(datatable), selcol
                )
                inpvars.append(norm_fact)
                norm_fact_rep = Replicator(norm_fact, len(tar_idcs))

                inpvar1_int = LinearInterpolation(inpvar1, src_en1, tar_en)
                inpvar2_int = LinearInterpolation(inpvar2, src_en2, tar_en)
                inpvar3_int = LinearInterpolation(inpvar3, src_en3, tar_en)
                tmpres = norm_fact_rep * inpvar1_int / (inpvar2_int + inpvar3_int)
                outvar = Distributor(tmpres, tar_idcs, len(datatable))
                outvars.append(outvar)

        inp = InputSelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
