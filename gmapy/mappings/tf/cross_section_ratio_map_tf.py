import numpy as np
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation
)


class CrossSectionRatioMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        datatable = cls._concat_datatable(datatable)
        return (
            datatable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))

        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()
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
            propfun = self._generate_atomic_propagate(src_en1, src_en2, tar_en)
            self._add_lists(
                (src_idcs1, src_idcs2), tar_idcs, propfun,
                (src_en1, src_en2, tar_en)
            )

    def _generate_atomic_propagate(self, src_en1, src_en2, tar_en):
        def _atomic_propagate(inpvar1, inpvar2):
            inpvar1_int = PiecewiseLinearInterpolation(src_en1, tar_en)(inpvar1)
            inpvar2_int = PiecewiseLinearInterpolation(src_en2, tar_en)(inpvar2)
            tmpres = inpvar1_int / inpvar2_int
            return tmpres
        return _atomic_propagate
