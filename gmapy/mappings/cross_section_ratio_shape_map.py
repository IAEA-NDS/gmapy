import numpy as np
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Replicator,
    Distributor,
    SumOfDistributors,
    LinearInterpolation
)
from .helperfuns import return_matrix_new


class CrossSectionRatioShapeMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:4-R1:[0-9]+-R2:[0-9]+', na=False) &
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
        priormask = np.logical_or(priormask, datatable['NODE'].str.match('norm_', na=False))
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
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(reac1str, na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(reac2str, na=False)]
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            # cycle over the datasets as each of those has different normalization constant
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            datasets = exptable_red['NODE'].unique()

            inpvar1 = Selector(src_idcs1, len(datatable))
            inpvar2 = Selector(src_idcs2, len(datatable))
            inpvars.extend([inpvar1, inpvar2])

            for ds in datasets:
                tar_idcs = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)].index
                tar_en = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)]['ENERGY']

                inpvar1_int = LinearInterpolation(inpvar1, src_en1, tar_en)
                inpvar2_int = LinearInterpolation(inpvar2, src_en2, tar_en)
                ratio = inpvar1_int / inpvar2_int

                # obtain normalization and position in priortable
                normstr = ds.replace('exp_', 'norm_')
                norm_index = priortable[priortable['NODE'].str.fullmatch(normstr, na=False)].index
                if len(norm_index) != 1:
                    raise IndexError('Exactly one normalization factor must be present for a dataset')

                norm_fact = Selector(norm_index, len(datatable))
                norm_fact_rep = Replicator(norm_fact, len(tar_idcs))
                mult_res = ratio * norm_fact_rep
                outvar = Distributor(mult_res, tar_idcs, len(datatable))
                inpvars.append(norm_fact)
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
