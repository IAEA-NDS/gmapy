from .priortools import prepare_prior_and_exptable
import tensorflow as tf
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelector,
    Distributor
)


class CrossSectionMap(tf.keras.layers.Layer):

    def __init__(self, datatable, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionMap not applicable')
        self._datatable = datatable
        self._reduce = reduce

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:1-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def call(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priortable = priortable[priormask]
        expmask = (exptable['REAC'].str.match('MT:1-R1:', na=False) &
                   exptable['NODE'].str.match('exp_', na=False))

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        out_list = []
        for curreac in reacs:
            priortable_red = priortable[
                priortable['REAC'].str.fullmatch(curreac, na=False)
            ]
            exptable_red = exptable[
                exptable['REAC'].str.fullmatch(curreac, na=False)
            ]
            # abbreviate some variables
            ens1 = tf.constant(priortable_red['ENERGY'],)
            idcs1red = tf.constant(priortable_red.index, dtype=tf.int32)
            ens2 = tf.constant(exptable_red['ENERGY'])
            idcs2red = tf.constant(exptable_red.index, dtype=tf.int32)

            inpvar = InputSelector(idcs1red)(inputs)
            intres = PiecewiseLinearInterpolation(ens1, ens2)(inpvar)
            outvar = Distributor(idcs2red, tar_len)(intres)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
