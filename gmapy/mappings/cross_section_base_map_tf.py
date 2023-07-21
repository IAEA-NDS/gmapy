import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    InputSelectorCollection,
    Distributor
)


class CrossSectionBaseMap(tf.Module):

    def __init__(self, datatable, selcol=None, reduce=True):
        super().__init__()
        if not self.is_applicable(datatable):
            raise TypeError('CrossSectionRatioMap not applicable')
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

    def _rebase_sparse_matrix(self, spmat, row_idcs, col_idcs, shape):
        row_idcs_tf = tf.constant(row_idcs, dtype=tf.int64)
        col_idcs_tf = tf.constant(col_idcs, dtype=tf.int64)
        col_slc = tf.slice(spmat.indices, [0, 1], [-1, 1])
        s = tf.gather(col_idcs_tf, col_slc)
        row_slc = tf.slice(spmat.indices, [0, 0], [-1, 1])
        t = tf.gather(row_idcs_tf, row_slc)
        z = tf.concat((t, s), axis=1)
        newmat = tf.sparse.SparseTensor(
            indices=z,
            values=spmat.values,
            dense_shape=shape
        )
        return newmat

    def _jacobian_iterator(self, inputs):
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
            jac_list = jacfun(*inpvars)
            tar_idcs_tf = tf.constant(tar_idcs, dtype=tf.int64)
            for src_idcs, jac in zip(src_idcs_list, jac_list):
                red_curjac = tf.sparse.from_dense(jac)
                curjac = self._rebase_sparse_matrix(
                    red_curjac, tar_idcs_tf, src_idcs,
                    (self._tar_len, self._src_len)
                )
                yield curjac

    def jacobian(self, inputs):
        res = None
        for curjac in self._jacobian_iterator(inputs):
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
