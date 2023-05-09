import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionShapeMap(tf.Module):

    def __init__(self, datatable, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionMap not applicable')
        self._datatable = datatable
        self._reduce = reduce
        if selcol is None:
            selcol = InputSelectorCollection()
        self._selcol = selcol

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:2-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    @tf.function
    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        isresp = np.array(exptable['REAC'].str.match('MT:2-R1:', na=False) &
                          exptable['NODE'].str.match('exp_', na=False))

        reacs = exptable.loc[isresp, 'REAC'].unique()

        selcol = self._selcol
        out_list = []
        for curreac in reacs:
            priormask = ((priortable['REAC'].str.fullmatch(curreac.replace('MT:2','MT:1'), na=False)) &
                         priortable['NODE'].str.match('xsid_', na=False))
            priortable_red = priortable[priormask]
            exptable_red = exptable[(exptable['REAC'].str.fullmatch(curreac, na=False) &
                                     exptable['NODE'].str.match('exp_'))]
            ens1 = np.array(priortable_red['ENERGY'])
            idcs1red = np.array(priortable_red.index)

            inpvar = selcol.define_selector(idcs1red)(inputs)
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

                norm_fact = selcol.define_selector(norm_index)(inputs)
                # abbreviate some variables
                ens2 = np.array(exptable_ds['ENERGY'])
                idcs2red = np.array(exptable_ds.index)

                inpvar_int = PiecewiseLinearInterpolation(ens1, ens2)(inpvar)
                prod = norm_fact * inpvar_int
                outvar = Distributor(idcs2red, tar_len)(prod)
                out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
