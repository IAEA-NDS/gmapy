import numpy as np
import pandas as pd
from .mapping_elements import (
    Selector,
    Distributor,
    reuse_or_create_input_selector
)


class RelativeErrorMap:

    def __init__(self, datatable, distributor_like, selector_list=None):
        self.__numrows = len(datatable)
        inp, out = self.__prepare(
            datatable, distributor_like, selector_list
        )
        self.__input = inp
        self.__output = out

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
        if self.__input is not None:
            return [self.__input]
        else:
            return []

    def get_distributors(self):
        if self.__output is not None:
            return [self.__output]
        else:
            return []

    def __prepare(self, datatable, distributor_like, selector_list):
        priormask = datatable['NODE'].str.match('relerr_', na=False)
        if not np.any(priormask):
            return None, None
        priortable = datatable[priormask]
        expmask = datatable['NODE'].str.match('exp_', na=False)
        exptable = datatable[expmask]
        # determine the source and target indices of the mapping
        expids = exptable['NODE'].str.extract(r'exp_([0-9]+)$')
        ptidx = exptable['PTIDX']
        rerr_expids = priortable['NODE'].str.extract(r'relerr_([0-9]+)$')
        rerr_ptidx = priortable['PTIDX']
        mapdf1 = pd.concat([expids, ptidx], axis=1)
        mapdf1.columns = ('expid', 'ptidx')
        mapdf1.reset_index(inplace=True, drop=False)
        mapdf1.set_index(['expid', 'ptidx'], inplace=True)
        mapdf2 = pd.concat([rerr_expids, rerr_ptidx], axis=1)
        mapdf2.columns = ('expid', 'ptidx')
        mapdf2.reset_index(inplace=True, drop=False)
        mapdf2.set_index(['expid', 'ptidx'], inplace=True)
        source_indices = mapdf2['index'].to_numpy()
        target_indices = mapdf1.loc[list(mapdf2.index), 'index'].to_numpy()
        # construct the selectors
        relerrors = reuse_or_create_input_selector(
            source_indices, len(datatable), selector_list
        )
        expquants = Selector(distributor_like, target_indices)
        abserrors = relerrors * expquants
        abserrors_dist = Distributor(abserrors, target_indices, len(datatable))

        inp = relerrors
        out = abserrors_dist
        return inp, out


def attach_relative_error_df(datatable):
    dt = datatable.copy()
    dt.sort_index()
    groupidx_df = dt.groupby('NODE').cumcount().to_frame(name='PTIDX')
    dt = pd.concat([dt, groupidx_df], axis=1)
    # create the relative errors dataframe
    isexp = dt['NODE'].str.match('exp_')
    relerr_dt = dt[isexp][['NODE', 'PTIDX', 'REAC', 'ENERGY']].copy()
    relerr_dt['NODE'] = relerr_dt['NODE'].str.replace('exp_', 'relerr_',
                                                      regex=False)
    relerr_dt['PRIOR'] = np.full(len(relerr_dt), 0.)
    # combine the two
    dt = pd.concat([dt, relerr_dt], axis=0, ignore_index=True)
    return dt