import numpy as np
from .cross_section_base_map import CrossSectionBaseMap
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .mapping_elements import (
    InputSelectorCollection,
    Const,
    Integral,
    FissionAverage,
    Replicator,
    Distributor,
    SumOfDistributors,
)


class CrossSectionFissionAverageMap(CrossSectionBaseMap):

    def __init__(self, datatable, fix_jacobian=True,
                 legacy_integration=True,
                 atol=1e-6, rtol=1e-6, maxord=16,
                 selcol=None, distsum=None, reduce=False):
        self._fix_jacobian = fix_jacobian
        self._legacy_integration = legacy_integration
        self._atol = atol
        self._rtol = rtol
        self._maxord = maxord
        super().__init__(datatable, selcol, distsum, reduce)

    def _prepare(self, priortable, exptable, selcol):
        legacy_integration = self._legacy_integration
        fix_jacobian = self._fix_jacobian
        expmask = np.array(
            exptable['REAC'].str.match('MT:6-R1:', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if not np.any(expmask):
            return inp, out

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
        ensfis = fistable['ENERGY'].to_numpy()

        raw_fisobj = selcol.define_selector(fistable.index, self._src_len)
        inp.add_selector(raw_fisobj)
        if legacy_integration:
            fisobj = raw_fisobj
        if not legacy_integration:
            scl = get_legacy_to_pointwise_fis_factors(ensfis)
            unnorm_fisobj = raw_fisobj * Const(scl)
            fisint = Integral(
                unnorm_fisobj, ensfis, 'lin-lin',
                atol=self._atol, rtol=self._rtol, maxord=self._maxord
            )
            fisobj = unnorm_fisobj / Replicator(fisint, len(unnorm_fisobj))

        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_red = priortable[priortable['REAC'].str.fullmatch(preac, na=False)]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY'].to_numpy()
            idcs1red = priortable_red.index

            xsobj = selcol.define_selector(idcs1red, self._src_len)
            inp.add_selector(xsobj)

            curfisavg = FissionAverage(ens1, xsobj, ensfis, fisobj,
                                       check_norm=False,
                                       legacy=legacy_integration,
                                       fix_jacobian=fix_jacobian,
                                       atol=self._atol, rtol=self._rtol,
                                       maxord=self._maxord)

            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            rep_curfisavg = Replicator(curfisavg, len(exptable_red))
            idcs2red = exptable_red.index
            outvar = Distributor(rep_curfisavg, idcs2red, self._tar_len)
            out.add_distributor(outvar)

        return inp, out
