import numpy as np
from .helperfuns import return_matrix_new
from .mapping_elements import (
    Selector,
    SelectorCollection,
    Const,
    Integral,
    FissionAverage,
    Replicator,
    Distributor
)


class CrossSectionFissionAverageMap:

    def __init__(self, fix_jacobian=True, legacy_integration=True):
        self._fix_jacobian = fix_jacobian
        self._legacy_integration = legacy_integration

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:6-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)

    def propagate(self, datatable, refvals):
        return self.__compute(datatable, refvals, what='propagate')

    def jacobian(self, datatable, refvals, ret_mat=False):
        Smat = self.__compute(datatable, refvals, what='jacobian')
        return return_matrix_new(Smat, how='csr' if ret_mat else 'dic')

    def __compute(self, datatable, refvals, what):
        assert what in ('propagate', 'jacobian')
        legacy_integration = self._legacy_integration
        fix_jacobian = self._fix_jacobian

        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))

        is_fis_row = datatable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = datatable[priormask]

        expmask = self.is_responsible(datatable)
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
            # The fission spectrum values in the legacy GMA database
            # are given as a histogram (piecewise rectangular function)
            # where the spectrum value in each bin is divided by the
            # energy bin size. For the new routine, where we interpret
            # the spectrum point-wise, we therefore need to multiply
            # by the energy bin size
            sort_idcs = ensfis.argsort()
            sorted_ensfis = ensfis[sort_idcs]
            xdiff = np.diff(sorted_ensfis)
            xmid = sorted_ensfis[:-1] + xdiff/2
            sorted_scl = np.full(len(sorted_ensfis), 1.)
            sorted_scl[1:-1] /= np.diff(xmid)
            sorted_scl[0] /= (xdiff[0]/2)
            sorted_scl[-1] /= (xdiff[-1]/2)
            scl = np.empty(len(sorted_scl), dtype=float)
            scl[sort_idcs] = sorted_scl
            # here come the autodiff augmented expressions
            unnorm_fisobj = raw_fisobj * Const(scl)
            fisint = Integral(
                unnorm_fisobj, ensfis, 'lin-lin', maxord=16, rtol=1e-6
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
                                       fix_jacobian=fix_jacobian)
            outvar = Distributor(curfisavg, idcs2red, len(datatable))
            outvars.append(outvar)

        inp = SelectorCollection(inpvars)
        out = sum(outvars)
        inp.assign(refvals)
        if what == 'propagate':
            return out.evaluate()
        elif what == 'jacobian':
            return out.jacobian()
        else:
            raise ValueError(f'what "{what}" not implemented"')
