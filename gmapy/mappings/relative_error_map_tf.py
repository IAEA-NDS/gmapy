import numpy as np
import pandas as pd
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelectorCollection,
    InputSelector,
    Distributor,
)


class RelativeErrorMap:

    def __init__(self, datatable, orig_output, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionMap not applicable')
        self._datatable = datatable
        self._reduce = reduce
        if selcol is None:
            selcol = InputSelectorCollection()
        self._selcol = selcol
        self._orig_output = orig_output

    @classmethod
    def is_applicable(cls, datatable):
        return (
            (datatable['NODE'].str.match('exp_', na=False)).any() &
            (datatable['NODE'].str.match('relerr_([0-9]+)$')).any()
        ).any()

    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = priortable['NODE'].str.match('relerr_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]

        orig_output = self._orig_output
        selcol = self._selcol
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
        relerrors = selcol.define_selector(source_indices)(inputs)
        expquants = InputSelector(target_indices)(orig_output)
        abserrors = relerrors * expquants
        abserrors_dist = Distributor(target_indices, tar_len)(abserrors)
        return abserrors_dist
