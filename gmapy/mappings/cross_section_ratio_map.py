import numpy as np
from .cross_section_base_map import CrossSectionBaseMap
from .mapping_elements import (
    InputSelectorCollection,
    Distributor,
    SumOfDistributors,
    LinearInterpolation,
)


class CrossSectionRatioMap(CrossSectionBaseMap):

    def _prepare(self, priortable, exptable, selcol):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))

        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if not np.any(expmask):
            return inp, out

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
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']

            inpvar1 = selcol.define_selector(src_idcs1, self._src_len)
            inpvar2 = selcol.define_selector(src_idcs2, self._src_len)
            inpvar1_int = LinearInterpolation(inpvar1, src_en1, tar_en)
            inpvar2_int = LinearInterpolation(inpvar2, src_en2, tar_en)
            tmpres = inpvar1_int / inpvar2_int
            outvar = Distributor(tmpres, tar_idcs, self._tar_len)
            inp.add_selectors([inpvar1, inpvar2])
            out.add_distributor(outvar)

        return inp, out
