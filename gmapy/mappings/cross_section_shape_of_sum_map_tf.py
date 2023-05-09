import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionShapeOfSumMap(tf.Module):

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
            datatable['REAC'].str.match('MT:8(-R[0-9]+:[0-9]+)+', na=False) &
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
            exptable['REAC'].str.match('MT:8(-R[0-9]+:[0-9]+)+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        selcol = self._selcol
        out_list = []
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
            src_idcs_list = [np.array(pt.index) for pt in priortable_reds]
            src_en_list = [np.array(pt['ENERGY']) for pt in priortable_reds]

            cvars = [
                selcol.define_selector(idcs)(inputs)
                for idcs in src_idcs_list
            ]

            # retrieve relevant rows in exptable
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
                cvars_int = []
                for cv, src_en in zip(cvars, src_en_list):
                    cvars_int.append(PiecewiseLinearInterpolation(src_en, tar_en)(cv))

                tmpres = tf.add_n(cvars_int) * norm_fact
                outvar = Distributor(tar_idcs, tar_len)(tmpres)
                out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
