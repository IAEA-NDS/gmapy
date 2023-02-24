import numpy as np
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Const,
    Integral,
    FissionAverage,
    Replicator,
    Distributor,
    SumOfDistributors
)


class CrossSectionFissionAverageMap:

    def __init__(self, datatable, fix_jacobian=True,
                 legacy_integration=True,
                 atol=1e-6, rtol=1e-6, maxord=16):
        self._fix_jacobian = fix_jacobian
        self._legacy_integration = legacy_integration
        self._atol = atol
        self._rtol = rtol
        self._maxord = maxord
        self.__numrows = len(datatable)
        self.__input, self.__output = self.__prepare(datatable)

    def is_responsible(self):
        ret = np.full(self.__numrows, False)
        if self.__output is not None:
            idcs = self.__output.get_indices()
            ret[idcs] = True
        return ret

    def propagate(self, refvals):
        self.__input.assign(refvals)
        return self.__output.evaluate()

    def jacobian(self, refvals):
        self.__input.assign(refvals)
        return self.__output.jacobian()

    def __prepare(self, datatable):
        legacy_integration = self._legacy_integration
        fix_jacobian = self._fix_jacobian
        expmask = np.array(
            datatable['REAC'].str.match('MT:6-R1:', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        )
        if not np.any(expmask):
            return None, None
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))

        is_fis_row = datatable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = datatable[priormask]
        exptable = datatable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()

        inpvars = []
        outvars = []
        raw_fisobj = Selector(fistable.index, len(datatable))
        inpvars.append(raw_fisobj)
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

        for curexp in expids:
            exptable_red = exptable[exptable['NODE'].str.fullmatch(curexp, na=False)]
            if len(exptable_red) != 1:
                raise IndexError('None or more than one rows associated with a ' +
                        'fission average, which must not happen!')
            curreac = exptable_red['REAC'].values[0]
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_red = priortable[priortable['REAC'].str.fullmatch(preac, na=False)]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY'].to_numpy()
            idcs1red = priortable_red.index
            idcs2red = exptable_red.index

            xsobj = Selector(idcs1red, len(datatable))
            inpvars.append(xsobj)

            curfisavg = FissionAverage(ens1, xsobj, ensfis, fisobj,
                                       check_norm=False,
                                       legacy=legacy_integration,
                                       fix_jacobian=fix_jacobian,
                                       atol=self._atol, rtol=self._rtol,
                                       maxord=self._maxord)
            outvar = Distributor(curfisavg, idcs2red, len(datatable))
            outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
