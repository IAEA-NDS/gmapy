import numpy as np
import pandas as pd
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelectorCollection,
    InputSelector,
    Distributor,
    PiecewiseLinearInterpolation
)


class EnergyDependentUSUMap:

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
            (datatable['NODE'].str.match('endep_usu_', na=False)).any()
        ).any()

    def __call__(self, inputs):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)

        priormask = priortable['NODE'].str.match('endep_usu_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]

        orig_output = self._orig_output
        selcol = self._selcol
        out_list = []
        # establish the mapping for each USU component
        # associated with an experimental dataset
        rerr_expids = priortable['NODE'].str.extract(r'endep_usu_([0-9]+)$')
        rerr_expids = pd.unique(rerr_expids.iloc[:, 0])
        for expid in rerr_expids:
            srcnode = 'endep_usu_' + expid
            srcdt = priortable[priortable['NODE'] == srcnode]
            tarnode = 'exp_' + expid
            tardt = exptable[exptable['NODE'] == tarnode]
            src_idcs = np.array(srcdt.index)
            tar_idcs = np.array(tardt.index)
            src_ens = srcdt['ENERGY'].to_numpy()
            tar_ens = tardt['ENERGY'].to_numpy()
            inpvar = selcol.define_selector(src_idcs)(inputs)
            relerrs = PiecewiseLinearInterpolation(src_ens, tar_ens)(inpvar)
            expquants = InputSelector(tar_idcs)(orig_output)
            abserrs = relerrs * expquants
            abserrs_dist = Distributor(tar_idcs, tar_len)(abserrs)
            out_list.append(abserrs_dist)

        # target_indices = tmp_out.get_indices()
        res = tf.add_n(out_list)
        return res
