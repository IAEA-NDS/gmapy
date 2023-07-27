import numpy as np
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class CrossSectionShapeMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            datatable['REAC'].str.match('MT:2-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        isresp = np.array(exptable['REAC'].str.match('MT:2-R1:', na=False) &
                          exptable['NODE'].str.match('exp_', na=False))

        reacs = exptable.loc[isresp, 'REAC'].unique()
        for curreac in reacs:
            priormask = ((priortable['REAC'].str.fullmatch(curreac.replace('MT:2','MT:1'), na=False)) &
                         priortable['NODE'].str.match('xsid_', na=False))
            priortable_red = priortable[priormask]
            exptable_red = exptable[(exptable['REAC'].str.fullmatch(curreac, na=False) &
                                     exptable['NODE'].str.match('exp_'))]
            ens1 = np.array(priortable_red['ENERGY'])
            idcs1red = np.array(priortable_red.index)
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'].str.fullmatch(dataset_id, na=False)]
                # get the respective normalization factor from prior
                mask = priortable['NODE'].str.fullmatch(dataset_id.replace('exp_', 'norm_'), na=False)
                norm_index = np.array(priortable[mask].index)
                if (len(norm_index) != 1):
                    raise IndexError('There are ' + str(len(norm_index)) +
                        ' normalization factors in prior for dataset ' + str(dataset_id))
                # abbreviate some variables
                ens2 = np.array(exptable_ds['ENERGY'])
                idcs2red = np.array(exptable_ds.index)
                propfun = self._generate_atomic_propagate(ens1, ens2)
                self._add_lists((idcs1red, norm_index), idcs2red, propfun)

    def _generate_atomic_propagate(self, ens1, ens2):
        def _atomic_propagate(inpvar, norm_fact):
            inpvar_int = PiecewiseLinearInterpolation(ens1, ens2)(inpvar)
            return norm_fact * inpvar_int
        return _atomic_propagate
