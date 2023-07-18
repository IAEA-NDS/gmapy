import numpy as np
import tensorflow as tf
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    IntegralLinLin,
    IntegralOfProductLinLin
)


class CrossSectionFissionAverageMap(CrossSectionBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:6-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        expmask = np.array(
            exptable['REAC'].str.match('MT:6-R1:', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )
        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))

        is_fis_row = priortable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = priortable[priormask]
        exptable = exptable[expmask]

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        fis_idcs = np.array(fistable.index, copy=True)
        ensfis = np.array(fistable['ENERGY'].to_numpy())

        # NOTE: Propagation of uncertainties in the energies of the
        #       fission spectrum would required porting get_legacy_to_pointwise_fis_factors
        #       to a function using tensorflow operations
        norm_fact_fis = get_legacy_to_pointwise_fis_factors(ensfis)

        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_red = priortable[priortable['REAC'].str.fullmatch(preac, na=False)]
            # abbreviate some variables
            ens1 = np.array(priortable_red['ENERGY'].to_numpy())
            idcs1red = np.array(priortable_red.index)

            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            idcs2red = np.array(exptable_red.index)

            num_reac_points = len(exptable_red)
            propfun = self._generate_atomic_propagate(
                ensfis, norm_fact_fis, ens1, num_reac_points
            )
            self._add_lists((idcs1red, fis_idcs), idcs2red, propfun)

    def _generate_atomic_propagate(
        self, ensfis, norm_fact_fis, ens1, num_reac_points
    ):
        def _atomic_propagate(xsobj, raw_fisobj):
            scl = tf.constant(norm_fact_fis, dtype=tf.float64)
            unnorm_fisobj = raw_fisobj * scl
            fisint = IntegralLinLin(ensfis)(unnorm_fisobj)
            fisobj = unnorm_fisobj / fisint
            curfisavg = IntegralOfProductLinLin(ens1, ensfis)(xsobj, fisobj)
            rep_curfisavg = tf.ones((num_reac_points,), dtype=tf.float64) * curfisavg
            return rep_curfisavg
        return _atomic_propagate
