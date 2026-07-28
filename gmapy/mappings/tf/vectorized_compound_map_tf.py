import numpy as np
import tensorflow as tf
from .compound_map_tf import CompoundMap
from .mapping_elements_tf import InputSelector, Distributor
from .tf_helperfuns import coalesce_sparse_matrices
from .vectorized_helpers import (
    concat_coo_triplets,
    row_pair_pattern
)


def _make_legacy_hessfun(propfun):
    # graph-compiled weighted Hessian block of one legacy dataset;
    # the pfor-vectorized double backward pass is traced only once
    @tf.function
    def hessfun(wt, *inpvars):
        with tf.GradientTape() as t2:
            t2.watch(inpvars)
            with tf.GradientTape() as t1:
                t1.watch(inpvars)
                sval = tf.reduce_sum(wt * propfun(*inpvars))
            grads = t1.gradient(
                sval, inpvars,
                unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
            gcat = tf.concat(grads, axis=0)
        hblocks = t2.jacobian(
            gcat, inpvars, experimental_use_pfor=True,
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return tf.concat(hblocks, axis=1)
    return hessfun


def _coalesce_numpy_coo(rows, cols, vals, num_cols):
    lin_idcs = rows.astype(np.int64) * num_cols + cols.astype(np.int64)
    uniq, inv = np.unique(lin_idcs, return_inverse=True)
    summed = np.bincount(inv, weights=vals, minlength=len(uniq))
    return uniq // num_cols, uniq % num_cols, summed


class _SparseFactor:
    """Constant COO matrix whose values get row-wise scaled in the Jacobian."""

    def __init__(self, rows, cols, vals, shape):
        rows, cols, vals = _coalesce_numpy_coo(rows, cols, vals, shape[1])
        self.np_rows = rows
        self.np_cols = cols
        self.np_vals = vals
        self.spmat = tf.sparse.SparseTensor(
            indices=np.stack([rows, cols], axis=1),
            values=tf.constant(vals, dtype=tf.float64),
            dense_shape=shape
        )
        self.row_idcs = tf.constant(rows, dtype=tf.int64)
        self.vals = tf.constant(vals, dtype=tf.float64)

    def matvec(self, x):
        res = tf.sparse.sparse_dense_matmul(
            self.spmat, tf.reshape(x, (-1, 1))
        )
        return tf.reshape(res, (-1,))

    def row_scaled(self, rowfactors):
        vals = self.vals * tf.gather(rowfactors, self.row_idcs)
        return tf.sparse.SparseTensor(
            indices=self.spmat.indices, values=vals,
            dense_shape=self.spmat.dense_shape
        )


class VectorizedCompoundMap(CompoundMap):
    """CompoundMap with dataset-vectorized propagation and Jacobian.

    All map classes providing a vectorization spec are collapsed into
    the algebraic form

        y = (N x + g0) * (A x) / (B x + b0)     (elementwise)

    with constant sparse matrices A (numerator interpolation),
    B (denominator interpolation, zero rows outside ratio-type data,
    compensated by b0=1 there) and N (normalization parameter gather,
    compensated by g0=1 where there is none). The Jacobian follows
    analytically as

        J = diag(g/den) A - diag(y/den) B + diag(num/den) N

    so neither GradientTape nor pfor is involved. Maps without a
    vectorization spec (the SACS maps) and the modifier maps keep
    their per-dataset path; the graph size becomes independent of
    the number of datasets covered by the algebraic form.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._legacy_maps = []
        blocks = []
        for curmap in self._maplist:
            try:
                curblocks = list(curmap.vectorized_blocks())
            except NotImplementedError:
                self._legacy_maps.append(curmap)
                continue
            blocks.extend(curblocks)
        refmap = self._maplist[0]
        self._tar_len = refmap._tar_len
        self._src_len = refmap._src_len
        self._assemble_algebraic_form(blocks)

    def _assemble_algebraic_form(self, blocks):
        tar_len = self._tar_len
        src_len = self._src_len
        shape = (tar_len, src_len)
        # each target row must be claimed by at most one dataset,
        # otherwise the algebraic form cannot represent the sum
        claimed = np.concatenate(
            [blk['tar_idcs'] for blk in blocks] +
            [np.asarray(tar_idcs) for curmap in self._legacy_maps
             for tar_idcs in curmap._tar_idcs_list]
        )
        if len(np.unique(claimed)) != len(claimed):
            raise IndexError(
                'target rows claimed by more than one dataset'
            )
        num_parts = []
        den_parts = []
        norm_rows = []
        norm_cols = []
        b0 = np.ones(tar_len)
        g0 = np.ones(tar_len)
        for blk in blocks:
            num_parts.append(blk['num'])
            if blk['den'] is not None:
                den_parts.append(blk['den'])
                b0[blk['tar_idcs']] = 0.
            if blk['norm_col'] is not None:
                norm_rows.append(blk['tar_idcs'])
                norm_cols.append(
                    np.full(len(blk['tar_idcs']), blk['norm_col'])
                )
                g0[blk['tar_idcs']] = 0.
        self._amat = _SparseFactor(*concat_coo_triplets(num_parts), shape)
        self._bmat = (_SparseFactor(*concat_coo_triplets(den_parts), shape)
                      if den_parts else None)
        if norm_rows:
            nrows = np.concatenate(norm_rows)
            ncols = np.concatenate(norm_cols)
            self._nmat = _SparseFactor(
                nrows, ncols, np.ones(len(nrows)), shape
            )
        else:
            self._nmat = None
        self._b0 = tf.constant(b0, dtype=tf.float64)
        self._g0 = tf.constant(g0, dtype=tf.float64)

    def _algebraic_parts(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        num = self._amat.matvec(x)
        den = self._b0
        if self._bmat is not None:
            den = den + self._bmat.matvec(x)
        g = self._g0
        if self._nmat is not None:
            g = g + self._nmat.matvec(x)
        return num, den, g

    def _orig_propagate(self, inputs):
        num, den, g = self._algebraic_parts(inputs)
        res = g * num / den
        for curmap in self._legacy_maps:
            res = res + curmap(inputs)
        if not self._reduce:
            inp = InputSelector(self._indep_idcs)(inputs)
            inpdist = Distributor(self._indep_idcs, self._tar_len)(inp)
            res = res + inpdist
        return res

    def _build_whess_patterns(self):
        # second derivatives of f = g*a/b in terms of the linear
        # functionals a = (Ax)_i, b = (Bx+b0)_i, g = (Nx+g0)_i:
        # d2f/dadb = -g/b^2, d2f/dadg = 1/b, d2f/dbdg = -a/b^2,
        # d2f/db2 = 2ga/b^3, d2f/da2 = d2f/dg2 = 0; the sum
        # sum_i w_i d2f_i factorizes into X^T diag(c) Y products
        # whose sparsity patterns are fixed at construction
        pairs = []
        amat, bmat, nmat = self._amat, self._bmat, self._nmat
        if bmat is not None:
            pairs.append(('ab', amat, bmat, True))
            pairs.append(('bb', bmat, bmat, False))
        if nmat is not None:
            pairs.append(('ag', amat, nmat, True))
        if bmat is not None and nmat is not None:
            pairs.append(('bg', bmat, nmat, True))
        patterns = []
        for coeff_key, xmat, ymat, mirror in pairs:
            rows, out_k, out_l, base = row_pair_pattern(
                xmat.np_rows, xmat.np_cols, xmat.np_vals,
                ymat.np_rows, ymat.np_cols, ymat.np_vals
            )
            if len(rows) == 0:
                continue
            idcs = np.stack([out_k, out_l], axis=1)
            patterns.append({
                'coeff': coeff_key,
                'rows': tf.constant(rows, dtype=tf.int64),
                'indices': tf.constant(idcs, dtype=tf.int64),
                'mirror_indices': (
                    tf.constant(idcs[:, ::-1].copy(), dtype=tf.int64)
                    if mirror else None
                ),
                'base': tf.constant(base, dtype=tf.float64),
            })
        self._whess_patterns = patterns

    def _build_legacy_whess_meta(self):
        meta = []
        for curmap in self._legacy_maps:
            it = curmap._lists_iterator()
            for src_idcs_list, tar_idcs, propfun, _, _ in it:
                inp_idcs = [
                    tf.constant(np.asarray(s), dtype=tf.int64)
                    for s in src_idcs_list
                ]
                glob = np.concatenate(
                    [np.asarray(s) for s in src_idcs_list]
                )
                mesh_r, mesh_c = np.meshgrid(glob, glob, indexing='ij')
                out_idcs = np.stack(
                    [mesh_r.reshape(-1), mesh_c.reshape(-1)], axis=1
                )
                meta.append({
                    'hessfun': _make_legacy_hessfun(propfun),
                    'inp_idcs': inp_idcs,
                    'tar_idcs': tf.constant(
                        np.asarray(tar_idcs), dtype=tf.int64
                    ),
                    'out_indices': tf.constant(out_idcs, dtype=tf.int64),
                })
        self._legacy_whess_meta = meta

    def _legacy_weighted_hessian_parts(self, x, w):
        # second-order contributions of the legacy (SACS) datasets:
        # each dataset only involves a small parameter subset, so its
        # weighted Hessian block is computed with nested tapes and
        # scattered into the full parameter space
        if not hasattr(self, '_legacy_whess_meta'):
            self._build_legacy_whess_meta()
        n = self._src_len
        parts = []
        for meta in self._legacy_whess_meta:
            inpvars = [tf.gather(x, idcs) for idcs in meta['inp_idcs']]
            wt = tf.gather(w, meta['tar_idcs'])
            hcat = meta['hessfun'](wt, *inpvars)
            parts.append(tf.sparse.SparseTensor(
                meta['out_indices'], tf.reshape(hcat, (-1,)), (n, n)
            ))
        return parts

    def weighted_row_hessian(self, inputs, weights, include_legacy=True):
        """Compute sum_i weights[i] * hessian(f_i) as sparse matrix.

        f_i are the components of `propagate`. Components covered by
        the algebraic form are handled by a closed-form contraction
        with fixed sparsity pattern; the datasets of the legacy
        (SACS) maps contribute dense blocks over their small
        parameter subsets, computed with nested tapes (skipped with
        `include_legacy=False`). The result is a sparse
        (src_len, src_len) tensor.
        """
        if not hasattr(self, '_whess_patterns'):
            self._build_whess_patterns()
        x = tf.convert_to_tensor(inputs, dtype=tf.float64)
        num, den, g = self._algebraic_parts(x)
        w = tf.convert_to_tensor(weights, dtype=tf.float64)
        coeffs = {
            'ab': -w * g / (den * den),
            'ag': w / den,
            'bg': -w * num / (den * den),
            'bb': 2. * w * g * num / (den * den * den),
        }
        n = self._src_len
        parts = []
        for pat in self._whess_patterns:
            vals = pat['base'] * tf.gather(coeffs[pat['coeff']], pat['rows'])
            parts.append(
                tf.sparse.SparseTensor(pat['indices'], vals, (n, n))
            )
            if pat['mirror_indices'] is not None:
                parts.append(
                    tf.sparse.SparseTensor(pat['mirror_indices'], vals, (n, n))
                )
        if include_legacy:
            parts.extend(self._legacy_weighted_hessian_parts(x, w))
        if not parts:
            return tf.sparse.SparseTensor(
                tf.zeros((0, 2), dtype=tf.int64),
                tf.zeros((0,), dtype=tf.float64), (n, n)
            )
        return coalesce_sparse_matrices(parts, (n, n))

    def _orig_jacobian(self, inputs):
        num, den, g = self._algebraic_parts(inputs)
        parts = [self._amat.row_scaled(g / den)]
        if self._bmat is not None:
            parts.append(self._bmat.row_scaled(-g * num / (den * den)))
        if self._nmat is not None:
            parts.append(self._nmat.row_scaled(num / den))
        for curmap in self._legacy_maps:
            parts.extend(curmap._jacobian_parts(inputs))
        return coalesce_sparse_matrices(
            parts, (self._tar_len, self._src_len)
        )
