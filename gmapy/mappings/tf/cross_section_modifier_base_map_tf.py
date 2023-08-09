import tensorflow as tf
from .cross_section_base_map_tf import CrossSectionBaseMap
from .mapping_elements_tf import (
    InputSelector,
    Distributor
)
from .tf_helperfuns import (
    scatter_sparse_matrix,
    subset_sparse_matrix
)


class CrossSectionModifierBaseMap(CrossSectionBaseMap):

    def __init__(self, datatable, orig_propvals,
                 selcol=None, reduce=True, jacfun=None):
        super().__init__(datatable, selcol, reduce)
        self._orig_propvals = orig_propvals
        self._orig_jacfun = jacfun

    @classmethod
    def is_applicable(cls, datatable):
        raise NotImplementedError('Please implement this method')

    def __call__(self, inputs):
        return self.propagate(inputs)

    def propagate(self, inputs):
        self._base_prepare_propagate()
        selcol = self._selcol
        out_list = []
        it = self._lists_iterator()
        for src_idcs_list, tar_idcs, propfun, _, _ in it:
            inpvars = tuple(
                selcol.define_selector(src_idcs)(inputs)
                for src_idcs in src_idcs_list
            )
            orig_propvals = InputSelector(tar_idcs)(self._orig_propvals)
            tmpres = propfun(orig_propvals, *inpvars)
            outvar = Distributor(tar_idcs, self._tar_len)(tmpres)
            out_list.append(outvar)

        res = tf.add_n(out_list)
        return res

    def jacobian(self, inputs, orig_jac=None):
        if orig_jac is None:
            orig_jac = self._orig_jacfun(inputs)
        res = None
        outer_iter = self._outer_jacobian_iterator(inputs)
        for jacfun, inpvars, src_idcs_list, tar_idcs in outer_iter:
            orig_propvals = InputSelector(tar_idcs)(self._orig_propvals)
            orig_propvals = tf.stop_gradient(orig_propvals)
            jac_list = jacfun(orig_propvals, *inpvars)
            # calculate contribution given by propagated original jacobian
            propjac = jac_list[0]
            red_orig_jac = subset_sparse_matrix(orig_jac, tar_idcs)
            red_orig_jac = tf.sparse.reorder(red_orig_jac)
            res1 = tf.sparse.sparse_dense_matmul(
                red_orig_jac, propjac, adjoint_a=True, adjoint_b=True
            )
            res1 = tf.sparse.transpose(tf.sparse.from_dense(res1))
            res1 = scatter_sparse_matrix(
                res1, tar_idcs, None, shape=(self._tar_len, self._src_len)
            )
            res = res1 if res is None else tf.sparse.add(res, res1)
            # calculate direct contributions
            inner_iter = self._inner_jacobian_iterator(
                src_idcs_list, tar_idcs, jac_list[1:]
            )
            for curjac in inner_iter:
                res = curjac if res is None else tf.sparse.add(res, curjac)
        return res
