import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionShapeOfRatioMap(tf.Module):

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
            datatable['REAC'].str.match('MT:9-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    @tf.function
    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priormask = np.logical_or(priormask, priortable['NODE'].str.match('norm_', na=False))
        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:9-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        selcol = self._selcol
        out_list = []
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
            src_idcs1 = np.array(priortable_red1.index)
            src_idcs2 = np.array(priortable_red2.index)
            src_idcs3 = np.array(priortable_red3.index)
            src_en1 = np.array(priortable_red1['ENERGY'])
            src_en2 = np.array(priortable_red2['ENERGY'])
            src_en3 = np.array(priortable_red3['ENERGY'])

            inpvar1 = selcol.define_selector(src_idcs1)(inputs)
            inpvar2 = selcol.define_selector(src_idcs2)(inputs)
            inpvar3 = selcol.define_selector(src_idcs3)(inputs)

            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            datasets = exptable_red['NODE'].unique()
            for ds in datasets:
                # subset another time exptable to get dataset info
                tar_idcs = np.array(
                    exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)].index
                )
                tar_en = np.array(
                    exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)]['ENERGY']
                )
                # obtain normalization and position in priortable
                normstr = ds.replace('exp_', 'norm_')
                norm_index = np.array(
                    priortable[priortable['NODE'].str.fullmatch(normstr, na=False)].index
                )
                if len(norm_index) != 1:
                    raise IndexError('Exactly one normalization factor must be present for a dataset')

                norm_fact = selcol.define_selector(norm_index)(inputs)

                inpvar1_int = PiecewiseLinearInterpolation(src_en1, tar_en)(inpvar1)
                inpvar2_int = PiecewiseLinearInterpolation(src_en2, tar_en)(inpvar2)
                inpvar3_int = PiecewiseLinearInterpolation(src_en3, tar_en)(inpvar3)
                tmpres = norm_fact * inpvar1_int / (inpvar2_int + inpvar3_int)
                outvar = Distributor(tar_idcs, tar_len)(tmpres)
                out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
