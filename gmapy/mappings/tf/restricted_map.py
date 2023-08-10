import tensorflow as tf
from .tf_helperfuns import subset_sparse_matrix


class RestrictedMap(tf.Module):

    def __init__(self, num_params, propfun, jacfun,
                 fixed_params=None, fixed_params_idcs=None):
        fixed_params = tf.reshape(
            tf.constant(fixed_params, dtype=tf.float64), (-1,)
        )
        fixed_params_idcs = tf.reshape(
            tf.constant(fixed_params_idcs, dtype=tf.int32), (-1, 1)
        )
        fixed_params_mask = tf.scatter_nd(
            fixed_params_idcs,
            tf.ones(tf.size(fixed_params_idcs), dtype=bool),
            (num_params,)
        )
        free_params_idcs = tf.where(~fixed_params_mask)
        self._num_all_params = tf.constant(num_params, dtype=tf.int32)
        self._num_free_params = tf.size(free_params_idcs)
        self._fixed_params = fixed_params
        self._fixed_params_idcs = fixed_params_idcs
        self._free_params_idcs = free_params_idcs
        self._orig_propfun = propfun
        self._orig_jacfun = jacfun

    def __call__(self, x):
        return self.propagate(x)

    def _assemble_paramvec(self, x):
        x = tf.reshape(x, (-1,))
        part1 = tf.scatter_nd(
            self._free_params_idcs, x, (self._num_all_params,)
        )
        part2 = tf.scatter_nd(
            self._fixed_params_idcs, self._fixed_params,
            (self._num_all_params,))
        return part1 + part2

    def propagate(self, x):
        x_full = self._assemble_paramvec(x)
        return self._orig_propfun(x_full)

    def jacobian(self, x):
        x_full = self._assemble_paramvec(x)
        orig_jac = self._orig_jacfun(x_full)
        tmp = tf.sparse.transpose(orig_jac)
        tmp = subset_sparse_matrix(tmp, self._free_params_idcs)
        jac = tf.sparse.transpose(tmp)
        return jac
