import numpy as np
import tensorflow as tf
from .priortools import prepare_prior_and_exptable
from .mapping_elements_tf import (
    PiecewiseLinearInterpolation,
    InputSelectorCollection,
    Distributor
)


class CrossSectionRatioMap(tf.Module):

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
        return (
            datatable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            datatable['NODE'].str.match('exp_', na=False)
        ).any()

    def _prepare_propagate(self):
        if self._prepared_propagate is True:
            return

        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(self._datatable, self._reduce)
        self._src_len = src_len
        self._tar_len = tar_len

        priormask = (priortable['REAC'].str.match('MT:1-R1:', na=False) &
                     priortable['NODE'].str.match('xsid_', na=False))

        priortable = priortable[priormask]
        expmask = np.array(
            exptable['REAC'].str.match('MT:3-R1:[0-9]+-R2:[0-9]+', na=False) &
            exptable['NODE'].str.match('exp_', na=False)
        )

        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            # obtian the involved reactions
            string_groups = curreac.split('-')
            reac1id = int(string_groups[1].split(':')[1])
            reac2id = int(string_groups[2].split(':')[1])
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(reac1str, na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(reac2str, na=False)]
            # and in the exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            # some abbreviations
            src_idcs1 = np.array(priortable_red1.index)
            src_idcs2 = np.array(priortable_red2.index)
            src_en1 = np.array(priortable_red1['ENERGY'])
            src_en2 = np.array(priortable_red2['ENERGY'])
            tar_idcs = np.array(exptable_red.index)
            tar_en = np.array(exptable_red['ENERGY'])
            propfun = self._generate_atomic_propagate(src_en1, src_en2, tar_en)
            self._add_lists(
                (src_idcs1, src_idcs2), tar_idcs, propfun,
                (src_en1, src_en2, tar_en)
            )
        self._prepared_propagate = True

    def _prepare_jacobian(self):
        if self._prepared_jacobian:
            return
        self._prepare_propagate()
        it = self._lists_iterator()
        jacfun_list = []
        for _, _, propfun, _, _ in it:
            jacfun = self._generate_atomic_jacobian(propfun)
            jacfun_list.append(jacfun)
        self._jacfun_list = jacfun_list
        self._prepared_jacobian = True

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

    @tf.function
    def __call__(self, inputs):
        self._prepare_propagate()
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

    @tf.function
    def jacobian(self, inputs):
        self._prepare_jacobian()
        selcol = self._selcol
        res = tf.sparse.SparseTensor(
            indices=[[0, 0]],
            values=tf.zeros(shape=(1,), dtype=tf.float64),
            dense_shape=(self._tar_len, self._src_len)
        )
        it = self._lists_iterator()
        for src_idcs_list, tar_idcs, _, jacfun, _ in it:
            inpvars = []
            for src_idcs in src_idcs_list:
                cur_inpvar = selcol.define_selector(src_idcs)(inputs)
                inpvars.append(cur_inpvar)
            jac_list = jacfun(*inpvars)
            for src_idcs, jac in zip(src_idcs_list, jac_list):
                st = tf.sparse.from_dense(jac)
                src_idcs_tf = tf.constant(src_idcs, dtype=tf.int64)
                tar_idcs_tf = tf.constant(tar_idcs, dtype=tf.int64)
                slice1 = tf.reshape(
                    tf.slice(st.indices, [0, 1], [-1, 1]),
                    shape=(-1,)
                )
                s1 = tf.gather(src_idcs_tf, slice1)
                slice2 = tf.reshape(
                    tf.slice(st.indices, [0, 0], [-1, 1]),
                    shape=(-1,)
                )
                t1 = tf.gather(tar_idcs_tf, slice2)
                z = tf.stack((t1, s1), axis=1)
                st = tf.sparse.SparseTensor(
                    indices=z,
                    values=st.values,
                    dense_shape=(self._tar_len, self._src_len)
                )
                res = tf.sparse.add(res, st)

        return res

    def _generate_atomic_propagate(self, src_en1, src_en2, tar_en):
        def _atomic_propagate(inpvar1, inpvar2):
            inpvar1_int = PiecewiseLinearInterpolation(src_en1, tar_en)(inpvar1)
            inpvar2_int = PiecewiseLinearInterpolation(src_en2, tar_en)(inpvar2)
            tmpres = inpvar1_int / inpvar2_int
            return tmpres
        return _atomic_propagate

    def _generate_atomic_jacobian(self, atomic_propagate_fun):
        def _atomic_jacobian(*inpvars):
            with tf.GradientTape() as tape:
                for iv in inpvars:
                    tape.watch(iv)
                predvals = atomic_propagate_fun(*inpvars)
            jac = tape.jacobian(predvals, inpvars)
            return jac
        return _atomic_jacobian
