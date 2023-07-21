import pandas as pd
import tensorflow as tf
from .cross_section_base_map_tf import CrossSectionBaseMap
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelector,
    Distributor
)


class RelativeErrorMap(CrossSectionBaseMap):

    def __init__(self, datatable, propvals,
                 selcol=None, reduce=True, jacfun=None):
        super().__init__(datatable, selcol, reduce)
        self._orig_propvals = propvals
        self._orig_jacfun = jacfun

    @classmethod
    def is_applicable(cls, datatable):
        return (
            (datatable['NODE'].str.match('exp_', na=False)).any() &
            (datatable['NODE'].str.match('relerr_([0-9]+)$')).any()
        ).any()

    def _prepare_propagate(self):
        if self._prepared_propagate:
            return
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)
        self._tar_len = tar_len
        priormask = priortable['NODE'].str.match('relerr_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]
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
        self._source_indices = source_indices
        self._target_indices = target_indices
        self._prepared_propagate = True

    def __call__(self, inputs):
        self._prepare_propagate()
        selcol = self._selcol
        orig_propvals = self._orig_propvals
        relerrors = selcol.define_selector(self._source_indices)(inputs)
        expquants = InputSelector(self._target_indices)(orig_propvals)
        abserrors = relerrors * expquants
        abserrors_dist = Distributor(
            self._target_indices, self._tar_len
        )(abserrors)
        return abserrors_dist

    def jacobian(self, inputs, orig_jac=None):
        self._prepare_propagate()
        if orig_jac is None:
            orig_jac = self._orig_jacfun(inputs)
        orig_propvals = self._orig_propvals
        reshaped_target_indices = self._target_indices.reshape(-1, 1)
        relerrs = tf.gather(inputs, self._source_indices)
        ext_relerrs = tf.scatter_nd(
            reshaped_target_indices, relerrs, (self._tar_len,)
        )
        row_idcs = tf.squeeze(tf.slice(orig_jac.indices, (0, 0), (-1, 1)))
        # tf.gather op selects the relative normalization errors
        scaled_vals = orig_jac.values * tf.gather(ext_relerrs, row_idcs)
        jac1 = tf.sparse.SparseTensor(
            indices=orig_jac.indices, values=scaled_vals,
            dense_shape=orig_jac.dense_shape
        )
        sel_expquants = tf.gather(orig_propvals, self._target_indices)
        idcs = tf.stack((self._target_indices, self._source_indices), axis=1)
        jac2 = tf.sparse.SparseTensor(
            indices=idcs, values=sel_expquants,
            dense_shape=orig_jac.dense_shape
        )
        compound_jac = tf.sparse.add(jac1, jac2)
        return compound_jac
