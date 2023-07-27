import tensorflow as tf
import pandas as pd
from ..priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelectorCollection,
    Distributor
)


class CrossSectionBaseMap(tf.Module):

    def __init__(self, datatable, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError(f'{type(self)} is not applicable')
        self._datatable = datatable
        self._reduce = reduce
        if selcol is None:
            selcol = InputSelectorCollection()
        self._selcol = selcol
        # member vars
        self._src_len = None
        self._tar_len = None
        self._src_idcs_list = []
        self._tar_idcs_list = []
        self._propfun_list = []
        self._aux_list = []
        self._jacfun_list = []
        self._prepared_propagate = False
        self._prepared_jacobian = False

    @classmethod
    def is_applicable(cls, datatable):
        raise NotImplementedError('Please implement this method')

    @classmethod
    def _concat_datatable(self, datatable):
        if isinstance(datatable, (list, tuple)):
            return pd.concat(datatable, axis=0, ignore_index=True)
        else:
            return datatable

    def _add_lists(
        self, src_idcs_list, tar_idcs, propfun, jacfun=None, aux_list=None
    ):
        self._src_idcs_list.append(src_idcs_list)
        self._tar_idcs_list.append(tar_idcs)
        self._aux_list.append(aux_list)
        self._propfun_list.append(propfun)
        self._jacfun_list.append(jacfun)

    def _lists_iterator(self):
        it = zip(
            self._src_idcs_list, self._tar_idcs_list,
            self._propfun_list, self._jacfun_list, self._aux_list
        )
        for src_idcs_list, tar_idcs_list, propfun, jacfun, aux_list in it:
            yield src_idcs_list, tar_idcs_list, propfun, jacfun, aux_list

    def _base_prepare_propagate(self):
        if not self._prepared_propagate:
            priortable, exptable, src_len, tar_len = \
                prepare_prior_and_exptable(self._datatable, self._reduce)
            self._src_len = src_len
            self._tar_len = tar_len
            self._prepare_propagate(priortable, exptable)
            self._prepared_propagate = True

    def __call__(self, inputs):
        self._base_prepare_propagate()
        selcol = self._selcol
        out_list = []
        it = self._lists_iterator()
        for src_idcs_list, tar_idcs, propfun, _, _ in it:
            inpvars = tuple(
                selcol.define_selector(src_idcs)(inputs)
                for src_idcs in src_idcs_list
            )
            tmpres = propfun(*inpvars)
            outvar = Distributor(tar_idcs, self._tar_len)(tmpres)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res

    def _prepare_jacobian(self):
        self._base_prepare_propagate()
        it = self._lists_iterator()
        jacfun_list = []
        for _, _, propfun, _, _ in it:
            jacfun = self._generate_atomic_jacobian(propfun)
            jacfun_list.append(jacfun)
        self._jacfun_list = jacfun_list
        self._prepared_jacobian = True

    def _subset_sparse_matrix(self, spmat, row_idcs):
        # prepare the index map
        min_idx = tf.reduce_min(row_idcs)
        max_idx = tf.reduce_max(row_idcs)
        lgth = max_idx - min_idx + 1
        idcs_map = tf.fill([lgth], tf.constant(-1, tf.int64))
        shifted_row_idcs = row_idcs - min_idx
        idcs_map = tf.tensor_scatter_nd_update(
            idcs_map,
            tf.reshape(shifted_row_idcs, (-1, 1)),
            tf.range(tf.size(shifted_row_idcs), dtype=tf.int64)
        )
        # remove indices being beyond the range of row_idcs
        orig_row_idcs = tf.slice(spmat.indices, [0, 0], [-1, 1])
        orig_col_idcs = tf.slice(spmat.indices, [0, 1], [-1, 1])
        orig_values = spmat.values
        mask = tf.logical_and(
            tf.greater_equal(orig_row_idcs, min_idx),
            tf.less_equal(orig_row_idcs, max_idx)
        )
        orig_row_idcs = tf.boolean_mask(orig_row_idcs, mask)
        orig_col_idcs = tf.boolean_mask(orig_col_idcs, mask)
        orig_values = tf.boolean_mask(orig_values, tf.reshape(mask, (-1,)))
        # now remove the row indices not part of the result
        orig_shifted_row_idcs = orig_row_idcs - min_idx
        mask = tf.greater_equal(
            tf.gather_nd(idcs_map, tf.reshape(orig_shifted_row_idcs, (-1, 1))),
            tf.constant(0, dtype=tf.int64)
        )
        orig_shifted_row_idcs = tf.boolean_mask(orig_shifted_row_idcs, mask)
        orig_row_idcs = tf.boolean_mask(orig_row_idcs, mask)
        orig_col_idcs = tf.boolean_mask(orig_col_idcs, mask)
        orig_values = tf.boolean_mask(orig_values, mask)
        # reorder the row indices so that they match the order in row_idcs
        new_row_idcs = tf.gather_nd(
            idcs_map, tf.reshape(orig_shifted_row_idcs, (-1, 1))
        )
        # assemble final sparse tensor
        new_spmat = tf.sparse.SparseTensor(
            indices=tf.stack((new_row_idcs, orig_col_idcs), axis=1),
            values=orig_values,
            dense_shape=(tf.size(row_idcs), spmat.dense_shape[1])
        )
        new_spmat = tf.sparse.reorder(new_spmat)
        return new_spmat

    def _scatter_sparse_matrix(self, spmat, row_idcs, col_idcs, shape):
        if col_idcs is None:
            col_idcs_tf = tf.range(spmat.dense_shape[1], dtype=tf.int64)
        else:
            if isinstance(col_idcs, tf.Tensor):
                col_idcs_tf = col_idcs
            else:
                col_idcs_tf = tf.constant(col_idcs, dtype=tf.int64)
        col_slc = tf.slice(spmat.indices, [0, 1], [-1, 1])
        s = tf.gather(col_idcs_tf, col_slc)
        if row_idcs is None:
            row_idcs_tf = tf.range(spmat.dense_shape[0], dtype=tf.int64)
        else:
            if isinstance(row_idcs, tf.Tensor):
                row_idcs_tf = row_idcs
            else:
                row_idcs_tf = tf.constant(row_idcs, dtype=tf.int64)
        row_slc = tf.slice(spmat.indices, [0, 0], [-1, 1])
        t = tf.gather(row_idcs_tf, row_slc)
        z = tf.concat((t, s), axis=1)
        newmat = tf.sparse.SparseTensor(
            indices=z,
            values=spmat.values,
            dense_shape=shape
        )
        return newmat

    def _outer_jacobian_iterator(self, inputs):
        if not self._prepared_jacobian:
            self._prepare_jacobian()
            self._prepared_jacobian = True
        selcol = self._selcol
        it = self._lists_iterator()
        for src_idcs_list, tar_idcs, _, jacfun, _ in it:
            inpvars = []
            for src_idcs in src_idcs_list:
                cur_inpvar = selcol.define_selector(src_idcs)(inputs)
                inpvars.append(cur_inpvar)
            yield jacfun, inpvars, src_idcs_list, tar_idcs

    def _inner_jacobian_iterator(self, src_idcs_list, tar_idcs, jac_list):
        tar_idcs_tf = tf.constant(tar_idcs, dtype=tf.int64)
        for src_idcs, jac in zip(src_idcs_list, jac_list):
            red_curjac = tf.sparse.from_dense(jac)
            curjac = self._scatter_sparse_matrix(
                red_curjac, tar_idcs_tf, src_idcs,
                (self._tar_len, self._src_len)
            )
            yield curjac

    def jacobian(self, inputs):
        res = None
        outer_iter = self._outer_jacobian_iterator(inputs)
        for jacfun, inpvars, src_idcs_list, tar_idcs in outer_iter:
            jac_list = jacfun(*inpvars)
            inner_iter = self._inner_jacobian_iterator(
                src_idcs_list, tar_idcs, jac_list
            )
            for curjac in inner_iter:
                res = curjac if res is None else tf.sparse.add(res, curjac)
        return res

    def _generate_atomic_propagate(self, *args, **kwargs):
        raise NotImplementedError(
            'Please implement this method'
        )

    def _generate_atomic_jacobian(self, atomic_propagate_fun):
        def _atomic_jacobian(*inpvars):
            with tf.GradientTape() as tape:
                for iv in inpvars:
                    tape.watch(iv)
                predvals = atomic_propagate_fun(*inpvars)
            jac = tape.jacobian(predvals, inpvars)
            return jac
        return _atomic_jacobian
