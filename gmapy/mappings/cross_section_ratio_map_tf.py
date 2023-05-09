import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionRatioMap(tf.Module):

    def __init__(self, datatable, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionRatioMap not applicable')
        self._datatable = datatable
        self._reduce = reduce
        if selcol is None:
            selcol = InputSelectorCollection()
        self._selcol = selcol

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    @tf.function
    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))

        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()
        selcol = self._selcol
        out_list = []
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
            src_idcs1 = np.array(priortable_red1.index)
            src_idcs2 = np.array(priortable_red2.index)
            src_en1 = np.array(priortable_red1['ENERGY'])
            src_en2 = np.array(priortable_red2['ENERGY'])
            tar_idcs = np.array(exptable_red.index)
            tar_en = np.array(exptable_red['ENERGY'])

            inpvar1 = selcol.define_selector(src_idcs1)(inputs)
            inpvar2 = selcol.define_selector(src_idcs2)(inputs)
            inpvar1_int = PiecewiseLinearInterpolation(src_en1, tar_en)(inpvar1)
            inpvar2_int = PiecewiseLinearInterpolation(src_en2, tar_en)(inpvar2)
            tmpres = inpvar1_int / inpvar2_int
            outvar = Distributor(tar_idcs, tar_len)(tmpres)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
