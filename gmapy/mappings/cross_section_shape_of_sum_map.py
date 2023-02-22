import numpy as np
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Replicator,
    Distributor,
    SumOfDistributors,
    LinearInterpolation
)


class CrossSectionShapeOfSumMap:

    def __init__(self, datatable):
        self.__numrows = len(datatable)
        self.__input, self.__output = self.__prepare(datatable)

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

    def __prepare(self, datatable):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priormask = np.logical_or(priormask, datatable['NODE'].str.match('norm_', na=False))
        priortable = datatable[priormask]
        expmask = np.array(
            datatable['REAC'].str.match('MT:8(-R[0-9]+:[0-9]+)+', na=False) &
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
            priortable_reds = [priortable[priortable['REAC'].str.fullmatch(r, na=False)]
                                    for r in reacstrs]
            # some abbreviations
            src_idcs_list = [pt.index for pt in priortable_reds]
            src_en_list = [pt['ENERGY'] for pt in priortable_reds]

            cvars = [Selector(idcs, len(datatable)) for idcs in src_idcs_list]
            inpvars.extend(cvars)

            # retrieve relevant rows in exptable
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

                norm_fact = Selector(norm_index, len(datatable))
                inpvars.append(norm_fact)
                norm_fact_rep = Replicator(norm_fact, len(tar_idcs))

                cvars_int = []
                for cv, src_en in zip(cvars, src_en_list):
                    cvars_int.append(LinearInterpolation(cv, src_en, tar_en))

                tmpres = sum(cvars_int) * norm_fact_rep
                outvar = Distributor(tmpres, tar_idcs, len(datatable))
                outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
