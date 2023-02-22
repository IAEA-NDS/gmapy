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


class CrossSectionShapeMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:2-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)

    def propagate(self, datatable, refvals):
        return self.__compute(datatable, refvals, 'propagate')

    def jacobian(self, datatable, refvals, ret_mat=False):
        S = self.__compute(datatable, refvals, 'jacobian')
        return return_matrix_new(S, how='csr' if ret_mat else 'dic')

    def __compute(self, datatable, refvals, what):
        isresp = self.is_responsible(datatable)
        reacs = datatable.loc[isresp, 'REAC'].unique()

        inpvars = []
        outvars = []
        for curreac in reacs:
            priormask = ((datatable['REAC'].str.fullmatch(curreac.replace('MT:2','MT:1'), na=False)) &
                         datatable['NODE'].str.match('xsid_', na=False))
            priortable_red = datatable[priormask]
            exptable_red = datatable[(datatable['REAC'].str.fullmatch(curreac, na=False) &
                                      datatable['NODE'].str.match('exp_'))]
            ens1 = priortable_red['ENERGY']
            idcs1red = priortable_red.index

            inpvar = Selector(idcs1red, len(datatable))
            inpvars.append(inpvar)
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'].str.fullmatch(dataset_id, na=False)]
                # get the respective normalization factor from prior
                mask = datatable['NODE'].str.fullmatch(dataset_id.replace('exp_', 'norm_'), na=False)
                norm_index = datatable[mask].index
                if (len(norm_index) != 1):
                    raise IndexError('There are ' + str(len(norm_index)) +
                        ' normalization factors in prior for dataset ' + str(dataset_id))

                norm_fact = Selector(norm_index, len(datatable))
                inpvars.append(norm_fact)
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index

                norm_fact_rep = Replicator(norm_fact, len(idcs2red))
                inpvar_int = LinearInterpolation(inpvar, ens1, ens2)
                prod = norm_fact_rep * inpvar_int
                outvar = Distributor(prod, idcs2red, len(datatable))
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
