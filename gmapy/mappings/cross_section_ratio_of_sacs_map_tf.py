import numpy as np
import tensorflow as tf
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelectorCollection,
    Distributor,
    IntegralOfProductLinLin
)


class CrossSectionRatioOfSacsMap:

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
            datatable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        is_fis_row = priortable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing in prior')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = priortable[priormask]
        expmask = exptable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False)

        selcol = self._selcol

        exptable = exptable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()

        raw_fisobj = selcol.define_selector(np.array(fistable.index))(inputs)

        scl = get_legacy_to_pointwise_fis_factors(ensfis)
        scl = tf.constant(scl, dtype=tf.float64)

        unnorm_fisobj = raw_fisobj * scl
        out_list = []
        for curexp in expids:
            exptable_red = exptable[exptable['NODE'].str.fullmatch(curexp, na=False)]
            if len(exptable_red) != 1:
                raise IndexError('None or more than one rows associated with a ' +
                        'ratio of SACS measurement, which must not happen!')
            curreac = exptable_red['REAC'].values[0]

            # locate the reactions relevant reactions in the prior
            curreac_split = curreac.split('-')
            curreac_split2 = [f.split(':') for f in curreac_split]
            reac1_id = int(curreac_split2[1][1])
            reac2_id = int(curreac_split2[2][1])

            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(f'MT:1-R1:{reac1_id}', na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(f'MT:1-R1:{reac2_id}', na=False)]
            # abbreviate some variables
            # first the ones associated with the SACS in the numerator
            ens1 = priortable_red1['ENERGY'].to_numpy()
            idcs1red = np.array(priortable_red1.index)
            # then the ones associated with the SACS in the denominator
            ens2 = priortable_red2['ENERGY'].to_numpy()
            idcs2red = np.array(priortable_red2.index)

            # finally we need the indices of experimental measurements
            idcs_exp_red = exptable_red.index

            xsobj1 = selcol.define_selector(idcs1red)(inputs)
            xsobj2 = selcol.define_selector(idcs2red)(inputs)

            fisavg1 = IntegralOfProductLinLin(ens1, ensfis)(xsobj1, unnorm_fisobj)
            fisavg2 = IntegralOfProductLinLin(ens2, ensfis)(xsobj2, unnorm_fisobj)

            fisavg_ratio = fisavg1 / fisavg2

            outvar = Distributor(idcs_exp_red, tar_len)(fisavg_ratio)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
