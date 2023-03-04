import numpy as np
from .helperfuns import (
    get_legacy_to_pointwise_fis_factors
)
from .mapping_elements import (
    InputSelectorCollection,
    Const,
    FissionAverage,
    Distributor,
    SumOfDistributors,
    reuse_or_create_input_selector
)


class CrossSectionRatioOfSacsMap:

    def __init__(self, datatable, atol=1e-05, rtol=1e-05,
                 maxord=16, selector_list=None):
        self.__atol = atol
        self.__rtol = rtol
        self.__maxord = maxord
        self.__numrows = len(datatable)
        self.__input, self.__output = self.__prepare(datatable, selector_list)

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

    def __prepare(self, datatable, selector_list):
        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        is_fis_row = datatable['NODE'].str.fullmatch('fis', na=False)
        if not is_fis_row.any():
            raise IndexError('fission spectrum missing in prior')
        priormask = np.logical_or(priormask, is_fis_row)
        priortable = datatable[priormask]
        expmask = datatable['REAC'].str.match('MT:10-R1:[0-9]+-R2:[0-9]+', na=False)
        if not np.any(expmask):
            return None, None
        exptable = datatable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE'].str.fullmatch('fis', na=False)]
        ensfis = fistable['ENERGY'].to_numpy()

        inpvars = []
        outvars = []
        raw_fisobj = reuse_or_create_input_selector(
            fistable.index, len(datatable), selector_list
        )
        inpvars.append(raw_fisobj)

        scl = get_legacy_to_pointwise_fis_factors(ensfis)
        unnorm_fisobj = raw_fisobj * Const(scl)

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
            idcs1red = priortable_red1.index
            # then the ones associated with the SACS in the denominator
            ens2 = priortable_red2['ENERGY'].to_numpy()
            idcs2red = priortable_red2.index

            # finally we need the indices of experimental measurements
            idcs_exp_red = exptable_red.index

            xsobj1 = reuse_or_create_input_selector(
                idcs1red, len(datatable), selector_list
            )
            xsobj2 = reuse_or_create_input_selector(
                idcs2red, len(datatable), selector_list
            )
            inpvars.append(xsobj1)
            inpvars.append(xsobj2)

            fisavg1 = FissionAverage(
                ens1, xsobj1, ensfis, unnorm_fisobj, check_norm=False,
                atol=self.__atol, rtol=self.__rtol, maxord=self.__maxord
            )
            fisavg2 = FissionAverage(
                ens2, xsobj2, ensfis, unnorm_fisobj, check_norm=False,
                atol=self.__atol, rtol=self.__rtol, maxord=self.__maxord
            )
            fisavg_ratio = fisavg1 / fisavg2

            outvar = Distributor(fisavg_ratio, idcs_exp_red, len(datatable))
            outvars.append(outvar)

        inp = InputSelectorCollection(inpvars)
        out = SumOfDistributors(outvars)
        return inp, out
