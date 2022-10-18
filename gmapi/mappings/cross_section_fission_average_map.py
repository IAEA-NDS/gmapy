import numpy as np
from scipy.sparse import csr_matrix
from .basic_integral_maps import (basic_integral_propagate,
        basic_integral_of_product_propagate, get_basic_integral_of_product_sensmats)
from .helperfuns import return_matrix

from ..legacy.legacy_maps import (propagate_fisavg, get_sensmat_fisavg,
        get_sensmat_fisavg_corrected)



class CrossSectionFissionAverageMap:

    def __init__(self, fix_jacobian=True, legacy_integration=True):
        self._fix_jacobian = fix_jacobian
        self._legacy_integration = legacy_integration


    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:6-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)


    def propagate(self, datatable, refvals):
        preds = np.full(len(datatable), 0., dtype=float)
        mapdic = self.__compute(datatable, refvals, what='propagate')
        preds[mapdic['idcs2']] = mapdic['propvals']
        return preds


    def jacobian(self, datatable, refvals, ret_mat=False):
        num_points = datatable.shape[0]
        Sdic = self.__compute(datatable, refvals, what='jacobian')
        return return_matrix(Sdic['idcs1'], Sdic['idcs2'], Sdic['coeff'],
                  dims = (num_points, num_points),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, datatable, refvals, what):
        has_legacy_fis = datatable.NODE.str.fullmatch('fis', na=False).any()
        has_modern_fis = datatable.NODE.str.fullmatch('fis_modern', na=False).any()
        if has_legacy_fis and has_modern_fis:
            raise ValueError('either legacy or modern fission spectrum, not both!')
        if has_legacy_fis:
            return self.__legacy_compute(datatable, refvals, what)
        elif has_modern_fis:
            return self.__modern_compute(datatable, refvals, what)


    def __legacy_compute(self, datatable, refvals, what):
        legacy_integration = self._legacy_integration

        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priormask = np.logical_or(priormask, datatable['NODE'].str.fullmatch('fis', na=False))
        priortable = datatable[priormask]

        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()
        valsfis = fistable['PRIOR'].to_numpy()

        if not legacy_integration:
                # The fission spectrum values in the legacy GMA database
                # are given as a histogram (piecewise rectangular function)
                # where the spectrum value in each bin is divided by the
                # energy bin size. For the new routine, where we interpret
                # the spectrum point-wise, we therefore need to multiply
                # by the energy bin size
                sort_idcs = ensfis.argsort()
                sorted_ensfis = ensfis[sort_idcs]
                sorted_valsfis = valsfis[sort_idcs]

                xdiff = np.diff(sorted_ensfis)
                xmid = sorted_ensfis[:-1] + xdiff/2
                scl_valsfis = np.full(len(sorted_ensfis), 0.)
                scl_valsfis[1:-1] = sorted_valsfis[1:-1] / np.diff(xmid)
                scl_valsfis[0] = sorted_valsfis[0] / (xdiff[0]/2)
                scl_valsfis[-1] = sorted_valsfis[-1] / (xdiff[-1]/2)
                valsfis[sort_idcs] = scl_valsfis
                # evaluate the normalization of the fission spectrum
                normfact = 1./float(basic_integral_propagate(ensfis, valsfis,
                                                            'lin-lin', maxord=16,
                                                            rtol=1e-6))

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
            vals1 = refvals[priortable_red.index]
            idcs1red = priortable_red.index
            idcs2red = exptable_red.index
            if what == 'propagate':

                if legacy_integration:
                    curval = propagate_fisavg(ens1, vals1, ensfis, valsfis)

                else:
                    curval = basic_integral_of_product_propagate(
                            [ens1, ensfis], [vals1, valsfis],
                            ['lin-lin', 'lin-lin'], zero_outside=True,
                            maxord=16, rtol=1e-6)
                    curval = float(curval) * normfact

                idcs2 = concat([idcs2, idcs2red])
                propvals = concat([propvals, [curval]])

            elif what == 'jacobian':

                if legacy_integration:

                    if self._fix_jacobian:
                        sensvec = get_sensmat_fisavg_corrected(ens1, vals1, ensfis, valsfis)
                    else:
                        sensvec = get_sensmat_fisavg(ens1, vals1, ensfis, valsfis)

                else:
                    sensvecs = get_basic_integral_of_product_sensmats(
                            [ens1, ensfis], [vals1, valsfis],
                            ['lin-lin', 'lin-lin'], zero_outside=True,
                            maxord=16, rtol=1e-6)
                    # because we assume that the fission spectrum is constant
                    # we can ignore the sensitivity to it
                    sensvec = np.ravel(sensvecs[0])
                    sensvec *= normfact

                idcs1 = concat([idcs1, idcs1red])
                idcs2 = concat([idcs2, np.full(len(idcs1red), idcs2red, dtype=int)])
                coeff = concat([coeff, sensvec])

            else:
                raise ValueError('what must be either "propagate" or "jacobian"')

        retdic = {}
        if what == 'jacobian':
            retdic['idcs1'] = idcs1
            retdic['idcs2'] = idcs2
            retdic['coeff'] = coeff
        elif what == 'propagate':
            retdic['idcs2'] = idcs2
            retdic['propvals'] = propvals

        return retdic


    def __modern_compute(self, datatable, refvals, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priormask = np.logical_or(priormask, datatable['NODE'].str.fullmatch('fis_modern|fis_errors', na=False))
        priortable = datatable[priormask]

        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis_modern', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()
        valsfis = fistable['PRIOR'].to_numpy()
        normfact = 1./float(basic_integral_propagate(ensfis, valsfis,
                                                            'lin-lin', maxord=16,
                                                            rtol=1e-6))
        if not np.isclose(normfact, 1., rtol=1e-4):
            raise ValueError('fission spectrum not normalized')

        priortable_fiserror = priortable[priortable['NODE'].str.fullmatch(
                                 'fis_errors', na=False)]
        ensfis_error = priortable_fiserror['ENERGY'].to_numpy()
        S = self.__make_covmapping(ensfis, ensfis_error, valsfis)

        for curexp in expids:
            exptable_red = exptable[exptable['NODE'].str.fullmatch(curexp, na=False)]
            if len(exptable_red) != 1:
                raise IndexError('None or more than one rows associated with a ' +
                        'fission average, which must not happen!')
            curreac = exptable_red['REAC'].values[0]
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_reac = priortable[priortable['REAC'].str.fullmatch(preac, na=False)]
            # abbreviate some variables
            ens1 = priortable_reac['ENERGY'].to_numpy()
            vals1_a = refvals[priortable_reac.index]
            vals1_b = S @ refvals[priortable_fiserror.index]
            idcs1red_a = priortable_reac.index
            idcs1red_b = priortable_fiserror.index
            idcs2red = exptable_red.index
            if what == 'propagate':

                curval = basic_integral_of_product_propagate(
                        [ens1, ensfis], [vals1_a, valsfis+vals1_b],
                        ['lin-lin', 'lin-lin'], zero_outside=True,
                        maxord=16, rtol=1e-6)
                curval = float(curval) * normfact

                idcs2 = concat([idcs2, idcs2red])
                propvals = concat([propvals, [curval]])

            elif what == 'jacobian':
                sensvecs = get_basic_integral_of_product_sensmats(
                        [ens1, ensfis], [vals1_a, valsfis+vals1_b],
                        ['lin-lin', 'lin-lin'], zero_outside=True,
                        maxord=16, rtol=1e-6)

                sensvec_a = np.ravel(sensvecs[0])
                sensvec_a *= normfact

                tmp = np.ravel(sensvecs[1])
                tmp *= normfact
                sensvec_b = np.ravel(tmp.reshape(1,-1) @ S)

                # forward propagation

                idcs1 = concat([idcs1, idcs1red_a, idcs1red_b])
                idcs2 = concat([idcs2, np.full(len(idcs1red_a)+len(idcs1red_b),
                                               idcs2red, dtype=int)])
                coeff = concat([coeff, sensvec_a, sensvec_b])

            else:
                raise ValueError('what must be either "propagate" or "jacobian"')

        retdic = {}
        if what == 'jacobian':
            retdic['idcs1'] = idcs1
            retdic['idcs2'] = idcs2
            retdic['coeff'] = coeff
        elif what == 'propagate':
            retdic['idcs2'] = idcs2
            retdic['propvals'] = propvals

        return retdic


    def __make_covmapping(self, ens, cov_ens, scl=None):
        if not np.all(np.sort(cov_ens) == cov_ens):
            raise ValueError('cov_ens must be sorted')
        if scl is None:
            scl = np.full(len(ens), 1.)
        en_idcs = np.searchsorted(cov_ens, ens)
        assert en_idcs[0] == 0
        en_idcs[0] = 1
        en_idcs -= 1
        i_list = []
        j_list = []
        v_list = []
        n = len(ens)
        m = len(cov_ens)
        for i in range(n):
            i_list.append(i)
            j_list.append(en_idcs[i])
            v_list.append(scl[i])
        S = csr_matrix((v_list, (i_list, j_list)), shape=(n, m), dtype='d')
        return S
