import numpy as np
import tensorflow as tf
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor,
    IntegralLinLin,
    IntegralOfProductLinLin
)

class CrossSectionFissionAverageMap(tf.Module):

    def __init__(self, datatable, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionMap not applicable')
        self._datatable = datatable
        self._reduce = reduce
        if selcol is None:
            selcol = InputSelectorCollection()
        self._selcol = selcol

    @classmethod
    def is_applicable(cls, datatable):
        return (
            datatable['REAC'].str.match('MT:6-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    @tf.function
    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

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
        ensfis = np.array(fistable['ENERGY'].to_numpy())

        selcol = self._selcol

        raw_fisobj = selcol.define_selector(np.array(fistable.index))(inputs)
        # NOTE: Propagation of uncertainties in the energies of the
        #       fission spectrum would required porting get_legacy_to_pointwise_fis_factors
        #       to a function using tensorflow operations
        scl = get_legacy_to_pointwise_fis_factors(ensfis)
        scl = tf.constant(scl, dtype=tf.float64)

        unnorm_fisobj = raw_fisobj * scl
        fisint = IntegralLinLin(ensfis)(unnorm_fisobj)
        fisobj = unnorm_fisobj / fisint

        reacs = exptable['REAC'].unique()
        out_list = []
        for curreac in reacs:
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_red = priortable[priortable['REAC'].str.fullmatch(preac, na=False)]
            # abbreviate some variables
            ens1 = np.array(priortable_red['ENERGY'].to_numpy())
            idcs1red = np.array(priortable_red.index)

            xsobj = selcol.define_selector(idcs1red)(inputs)
            curfisavg = IntegralOfProductLinLin(ens1, ensfis)(xsobj, fisobj)

            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            rep_curfisavg = tf.ones((len(exptable_red),), dtype=tf.float64) * curfisavg
            idcs2red = np.array(exptable_red.index)
            outvar = Distributor(idcs2red, tar_len)(rep_curfisavg)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res
