import numpy as np
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
from .priortools import prepare_prior_and_exptable


class CrossSectionFissionAverageMap:

    def __init__(self, datatable, fix_jacobian=True,
                 legacy_integration=True,
                 atol=1e-6, rtol=1e-6, maxord=16,
                 selcol=None, distsum=None, reduce=False):
        self._fix_jacobian = fix_jacobian
        self._legacy_integration = legacy_integration
        self._atol = atol
        self._rtol = rtol
        self._maxord = maxord
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        self.__input, self.__output = self.__prepare(datatable, selcol, reduce)
        if distsum is not None:
            distsum.add_distributors(self.__output.get_distributors())

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

    def get_selectors(self):
        return self.__input.get_selectors()

    def get_distributors(self):
        return self.__output.get_distributors()

    def __prepare(self, datatable, selcol, reduce):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(datatable, reduce)

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
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()

        raw_fisobj = selcol.define_selector(fistable.index, src_len)
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

            xsobj = selcol.define_selector(idcs1red, src_len)
            inp.add_selector(xsobj)

            curfisavg = FissionAverage(ens1, xsobj, ensfis, fisobj,
                                       check_norm=False,
                                       legacy=legacy_integration,
                                       fix_jacobian=fix_jacobian,
                                       atol=self._atol, rtol=self._rtol,
                                       maxord=self._maxord)
            outvar = Distributor(curfisavg, idcs2red, tar_len)
            out.add_distributor(outvar)

        return inp, out
