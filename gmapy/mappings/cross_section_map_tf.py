import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionMap(tf.Module):

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
            datatable['REAC'].str.match('MT:1-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    @tf.function
    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priortable = priortable[priormask]
        expmask = (exptable['REAC'].str.match('MT:1-R1:', na=False) &
                   exptable['NODE'].str.match('exp_', na=False))

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        selcol = self._selcol
        out_list = []
        for curreac in reacs:
            priortable_red = priortable[
                priortable['REAC'].str.fullmatch(curreac, na=False)
            ]
            exptable_red = exptable[
                exptable['REAC'].str.fullmatch(curreac, na=False)
            ]
            # abbreviate some variables
            ens1 = np.array(priortable_red['ENERGY'])
            idcs1red = np.array(priortable_red.index)
            ens2 = np.array(exptable_red['ENERGY'])
            idcs2red = np.array(exptable_red.index)

            inpvar = selcol.define_selector(idcs1red)(inputs)
            intres = PiecewiseLinearInterpolation(ens1, ens2)(inpvar)

            outvar = Distributor(idcs2red, tar_len)(intres)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
