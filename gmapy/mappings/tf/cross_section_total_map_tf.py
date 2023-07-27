import numpy as np
import tensorflow as tf
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class CrossSectionTotalMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            datatable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )
        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            # obtian the involved reactions
            reac_groups = curreac.split('-')[1:]
            reacids = [int(x.split(':')[1]) for x in reac_groups]
            reacstrs = ['MT:1-R1:' + str(rid) for rid in reacids]
            if len(np.unique(reacstrs)) < len(reacstrs):
                   raise IndexError('Each reaction must occur only once in reaction string')
            # retrieve the relevant reactions in the prior
            priortable_reds = [priortable[priortable['REAC'].str.fullmatch(r, na=False)] for r in reacstrs]
            # retrieve relevant rows in exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # some abbreviations
            src_idcs_list = [np.array(pt.index) for pt in priortable_reds]
            src_en_list = [np.array(pt['ENERGY']) for pt in priortable_reds]
            tar_idcs = np.array(exptable_red.index)
            tar_en = np.array(exptable_red['ENERGY'])
            propfun = self._generate_atomic_propagate(src_en_list, tar_en)
            self._add_lists(src_idcs_list, tar_idcs, propfun)

    def _generate_atomic_propagate(self, src_en_list, tar_en):
        def _atomic_propagate(*cvars):
            cvars_int = []
            for cv, en in zip(cvars, src_en_list):
                cvars_int.append(PiecewiseLinearInterpolation(en, tar_en)(cv))

            return tf.add_n(cvars_int)
        return _atomic_propagate
