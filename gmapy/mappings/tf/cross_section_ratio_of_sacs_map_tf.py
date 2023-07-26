import numpy as np
import tensorflow as tf
from ..helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    IntegralOfProductLinLin
)


class CrossSectionRatioOfSacsMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))
        is_fis_row = priortable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing in prior')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = priortable[priormask]
        expmask = exptable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False)

        exptable = exptable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        fis_idcs = np.array(fistable.index)
        ensfis = fistable['ENERGY'].to_numpy()

        norm_fact = get_legacy_to_pointwise_fis_factors(ensfis)
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
            propfun = self._generate_atomic_propagate(
                ens1, ens2, ensfis, norm_fact
            )
            self._add_lists(
                (idcs1red, idcs2red, fis_idcs), idcs_exp_red, propfun
            )

    def _generate_atomic_propagate(self, ens1, ens2, ensfis, norm_fact):
        def _atomic_propagate(xsobj1, xsobj2, raw_fisobj):
            scl = tf.constant(norm_fact, dtype=tf.float64)
            unnorm_fisobj = raw_fisobj * scl
            fisavg1 = IntegralOfProductLinLin(ens1, ensfis)(xsobj1, unnorm_fisobj)
            fisavg2 = IntegralOfProductLinLin(ens2, ensfis)(xsobj2, unnorm_fisobj)
            fisavg_ratio = fisavg1 / fisavg2
            return fisavg_ratio
        return _atomic_propagate
