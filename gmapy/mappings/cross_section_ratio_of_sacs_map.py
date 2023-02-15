import numpy as np
from .basic_integral_maps import (basic_integral_propagate,
        basic_integral_of_product_propagate, get_basic_integral_of_product_sensmats)
from .helperfuns import return_matrix

from ..legacy.legacy_maps import (propagate_fisavg, get_sensmat_fisavg,
        get_sensmat_fisavg_corrected)


class CrossSectionRatioOfSacsMap:

    def __init__(self, atol=1e-05, rtol=1e-05, maxord=16):
        self.__atol = atol
        self.__rtol = rtol
        self.__maxord = maxord

    def is_responsible(self, datatable):
        expmask = datatable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False)
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

        if what not in ('propagate', 'jacobian'):
            raise ValueError('what must be either "propagate" or "jacobian"')

        idcs1 = np.empty(0, dtype=int)
        idcs_exp = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        is_fis_row = datatable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing in prior')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = datatable[priormask]

        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()
        valsfis = fistable['PRIOR'].to_numpy()

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
            vals1 = refvals[priortable_red1.index]
            idcs1red = priortable_red1.index
            # then the ones associated with the SACS in the denominator
            ens2 = priortable_red2['ENERGY'].to_numpy()
            vals2 = refvals[priortable_red2.index]
            idcs2red = priortable_red2.index

            # finally we need the indices of experimental measurements
            idcs_exp_red = exptable_red.index

            # we must evaluate the integrals in any case,
            # doesn't matter if Jacobian or propagation is desired
            # because the results of the integrals appear
            # also in the Jacobian due to the chain rule
            curval1 = basic_integral_of_product_propagate(
                    [ens1, ensfis], [vals1, valsfis],
                    ['lin-lin', 'lin-lin'], zero_outside=True,
                    maxord=self.__maxord, rtol=self.__rtol, atol=self.__atol)
            curval1 = float(curval1) * normfact

            curval2 = basic_integral_of_product_propagate(
                    [ens2, ensfis], [vals2, valsfis],
                    ['lin-lin', 'lin-lin'], zero_outside=True,
                    maxord=self.__maxord, rtol=self.__rtol, atol=self.__atol)
            curval2 = float(curval2) * normfact

            if what == 'propagate':

                idcs_exp = concat([idcs_exp, idcs_exp_red])
                propvals = concat([propvals, [curval1/curval2]])

            if what == 'jacobian':

                # first evaluate the sensitivity with respect to
                # the SACS value in the numerator
                sensvecs1 = get_basic_integral_of_product_sensmats(
                        [ens1, ensfis], [vals1, valsfis],
                        ['lin-lin', 'lin-lin'], zero_outside=True,
                        maxord=self.__maxord, rtol=self.__rtol, atol=self.__atol)
                # because we assume that the fission spectrum is constant
                # we can ignore the sensitivity to it
                sensvec1 = np.ravel(sensvecs1[0])
                sensvec1 *= normfact

                # now we evaluate the sensitivity with respect to
                # the SACS value in the denominator
                sensvecs2 = get_basic_integral_of_product_sensmats(
                        [ens2, ensfis], [vals2, valsfis],
                        ['lin-lin', 'lin-lin'], zero_outside=True,
                        maxord=self.__maxord, rtol=self.__rtol, atol=self.__atol)
                sensvec2 = np.ravel(sensvecs2[0])
                sensvec2 *= normfact

                # apply chain rule to obtain Jacobian (d_i x1/x2 = 1/x2*d_i x1 - x1/x2^2 * d_i x2)
                sensvec1 = sensvec1 / curval2
                sensvec2 = -curval1 / (curval2*curval2) * sensvec2

                idcs1 = concat([idcs1, idcs1red, idcs2red])
                idcs_exp = concat([idcs_exp, np.full(len(idcs1red)+len(idcs2red), idcs_exp_red, dtype=int)])
                coeff = concat([coeff, sensvec1, sensvec2])

        retdic = {}
        if what == 'jacobian':
            retdic['idcs1'] = idcs1
            retdic['idcs2'] = idcs_exp
            retdic['coeff'] = coeff
        elif what == 'propagate':
            retdic['idcs2'] = idcs_exp
            retdic['propvals'] = propvals

        return retdic
