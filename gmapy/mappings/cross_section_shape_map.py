import numpy as np
from .cross_section_base_map import CrossSectionBaseMap
from .mapping_elements import (
    InputSelectorCollection,
    Replicator,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
)


class CrossSectionShapeMap(CrossSectionBaseMap):

    def _prepare(self, priortable, exptable, selcol):
        isresp = np.array(exptable['REAC'].str.match('MT:2-R1:', na=False) &
                          exptable['NODE'].str.match('exp_', na=False))

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if not np.any(isresp):
            return inp, out
        reacs = exptable.loc[isresp, 'REAC'].unique()

        for curreac in reacs:
            priormask = ((priortable['REAC'].str.fullmatch(curreac.replace('MT:2','MT:1'), na=False)) &
                         priortable['NODE'].str.match('xsid_', na=False))
            priortable_red = priortable[priormask]
            exptable_red = exptable[(exptable['REAC'].str.fullmatch(curreac, na=False) &
                                     exptable['NODE'].str.match('exp_'))]
            ens1 = priortable_red['ENERGY']
            idcs1red = priortable_red.index

            inpvar = selcol.define_selector(idcs1red, self._src_len)
            inp.add_selector(inpvar)
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'].str.fullmatch(dataset_id, na=False)]
                # get the respective normalization factor from prior
                mask = priortable['NODE'].str.fullmatch(dataset_id.replace('exp_', 'norm_'), na=False)
                norm_index = priortable[mask].index
                if (len(norm_index) != 1):
                    raise IndexError('There are ' + str(len(norm_index)) +
                        ' normalization factors in prior for dataset ' + str(dataset_id))

                norm_fact = selcol.define_selector(norm_index, self._src_len)
                inp.add_selector(norm_fact)
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index

                norm_fact_rep = Replicator(norm_fact, len(idcs2red))
                inpvar_int = LinearInterpolation(inpvar, ens1, ens2)
                prod = norm_fact_rep * inpvar_int
                outvar = Distributor(prod, idcs2red, self._tar_len)
                out.add_distributor(outvar)

        return inp, out
