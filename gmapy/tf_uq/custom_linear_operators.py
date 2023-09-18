import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Perturb a `LinearOperator` with a rank `K` update."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "MyLinearOperatorLowRankUpdate",
]


# @tf_export("linalg.LinearOperatorLowRankUpdate")
@linear_operator.make_composite_tensor
class MyLinearOperatorLowRankUpdate(linear_operator.LinearOperator):
    """Perturb a `LinearOperator` with a rank `K` update.

    This operator acts like a [batch] matrix `A` with shape
    `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
    batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
    an `M x N` matrix.

    `LinearOperatorLowRankUpdate` represents `A = L + U D V^H`, where

    ```
    L, is a LinearOperator representing [batch] M x N matrices
    U, is a [batch] M x K matrix.  Typically K << M.
    D, is a [batch] K x K matrix.
    V, is a [batch] N x K matrix.  Typically K << N.
    V^H is the Hermitian transpose (adjoint) of V.
    ```

    If `M = N`, determinants and solves are done using the matrix determinant
    lemma and Woodbury identities, and thus require L and D to be non-singular.

    Solves and determinants will be attempted unless the "is_non_singular"
    property of L and D is False.

    In the event that L and D are positive-definite, and U = V, solves and
    determinants can be done using a Cholesky factorization.

    ```python
    # Create a 3 x 3 diagonal linear operator.
    diag_operator = LinearOperatorDiag(
        update_operator=[1., 2., 3.], is_non_singular=True, is_self_adjoint=True,
        is_positive_definite=True)

    # Perturb with a rank 2 perturbation
    operator = LinearOperatorLowRankUpdate(
        operator=diag_operator,
        u=[[1., 2.], [-1., 3.], [0., 0.]],
        update_operator=[11., 12.],
        v=[[1., 2.], [-1., 3.], [10., 10.]])

    operator.shape
    ==> [3, 3]

    operator.log_abs_determinant()
    ==> scalar Tensor

    x = ... Shape [3, 4] Tensor
    operator.matmul(x)
    ==> Shape [3, 4] Tensor
    ```

    ### Shape compatibility

    This operator acts on [batch] matrix with compatible shape.
    `x` is a batch matrix with compatible shape for `matmul` and `solve` if

    ```
    operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
    x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
    ```

    ### Performance

    Suppose `operator` is a `LinearOperatorLowRankUpdate` of shape `[M, N]`,
    made from a rank `K` update of `base_operator` which performs `.matmul(x)` on
    `x` having `x.shape = [N, R]` with `O(L_matmul*N*R)` complexity (and similarly
    for `solve`, `determinant`.  Then, if `x.shape = [N, R]`,

    * `operator.matmul(x)` is `O(L_matmul*N*R + K*N*R)`

    and if `M = N`,

    * `operator.solve(x)` is `O(L_matmul*N*R + N*K*R + K^2*R + K^3)`
    * `operator.determinant()` is `O(L_determinant + L_solve*N*K + K^2*N + K^3)`

    If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
    `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

    #### Matrix property hints

    This `LinearOperator` is initialized with boolean flags of the form `is_X`,
    for `X = non_singular`, `self_adjoint`, `positive_definite`,
    `update_operator_positive` and `square`. These have the following meaning:

    * If `is_X == True`, callers should expect the operator to have the
      property `X`.  This is a promise that should be fulfilled, but is *not* a
      runtime assert.  For example, finite floating point precision may result
      in these promises being violated.
    * If `is_X == False`, callers should expect the operator to not have `X`.
    * If `is_X == None` (the default), callers should have no expectation either
      way.
    """

    def __init__(self,
                 base_operator,
                 u,
                 update_operator=None,
                 v=None,
                 is_update_positive_definite=None,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 is_square=None,
                 name="MyLinearOperatorLowRankUpdate"):
        """Initialize a `MyLinearOperatorLowRankUpdate`.

        This creates a `LinearOperator` of the form `A = L + U D V^H`, with
        `L` a `LinearOperator`, `U, V` both [batch] matrices, and `D` a [batch]
        diagonal matrix.

        If `L` is non-singular, solves and determinants are available.
        Solves/determinants both involve a solve/determinant of a `K x K` system.
        In the event that L and D are self-adjoint positive-definite, and U = V,
        this can be done using a Cholesky factorization.  The user should set the
        `is_X` matrix property hints, which will trigger the appropriate code path.

        Args:
          base_operator:  Shape `[B1,...,Bb, M, N]`.
          u:  Shape `[B1,...,Bb, M, K]` `Tensor` of same `dtype` as `base_operator`.
            This is `U` above.
          update_operator:  Optional shape `[B1,...,Bb, K]` `Tensor` with same `dtype`
            as `base_operator`.  This is the diagonal of `D` above.
             Defaults to `D` being the identity operator.
          v:  Optional `Tensor` of same `dtype` as `u` and shape `[B1,...,Bb, N, K]`
             Defaults to `v = u`, in which case the perturbation is symmetric.
             If `M != N`, then `v` must be set since the perturbation is not square.
          is_update_positive_definite:  Python `bool`.
            If `True`, expect `update_operator > 0`.
          is_non_singular:  Expect that this operator is non-singular.
            Default is `None`, unless `is_positive_definite` is auto-set to be
            `True` (see below).
          is_self_adjoint:  Expect that this operator is equal to its hermitian
            transpose.  Default is `None`, unless `base_operator` is self-adjoint
            and `v = None` (meaning `u=v`), in which case this defaults to `True`.
          is_positive_definite:  Expect that this operator is positive definite.
            Default is `None`, unless `base_operator` is positive-definite
            `v = None` (meaning `u=v`), and `is_update_positive_definite`, in which case
            this defaults to `True`.
            Note that we say an operator is positive definite when the quadratic
            form `x^H A x` has positive real part for all nonzero `x`.
          is_square:  Expect that this operator acts like square [batch] matrices.
          name: A name for this `LinearOperator`.

        Raises:
          ValueError:  If `is_X` flags are set in an inconsistent way.
        """
        parameters = dict(
            base_operator=base_operator,
            u=u,
            update_operator=update_operator,
            v=v,
            is_update_positive_definite=is_update_positive_definite,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name
        )
        dtype = base_operator.dtype

        if update_operator is not None:
            if is_update_positive_definite and dtype.is_complex:
                logging.warn("Note: setting is_update_positive_definite with a complex "
                             "dtype means that diagonal is real and positive.")

        if update_operator is None:
            if is_update_positive_definite is False:
                raise ValueError(
                    "Default diagonal is the identity, which is positive.  However, "
                    "user set 'is_update_positive_definite' to False.")
            is_update_positive_definite = True

        # In this case, we can use a Cholesky decomposition to help us solve/det.
        self._use_cholesky = (
            base_operator.is_positive_definite and base_operator.is_self_adjoint
            and is_update_positive_definite
            and v is None)

        # Possibly auto-set some characteristic flags from None to True.
        # If the Flags were set (by the user) incorrectly to False, then raise.
        if base_operator.is_self_adjoint and v is None and not dtype.is_complex:
            if is_self_adjoint is False:
                raise ValueError(
                    "A = L + UDU^H, with L self-adjoint and D real diagonal.  Since"
                    " UDU^H is self-adjoint, this must be a self-adjoint operator.")
            is_self_adjoint = True

        # The condition for using a cholesky is sufficient for SPD, and
        # we no weaker choice of these hints leads to SPD.  Therefore,
        # the following line reads "if hints indicate SPD..."
        if self._use_cholesky:
            if (
                is_positive_definite is False
                or is_self_adjoint is False
                or is_non_singular is False):
                raise ValueError(
                    "Arguments imply this is self-adjoint positive-definite operator.")
            is_positive_definite = True
            is_self_adjoint = True

        with ops.name_scope(name):

            # Create U and V.
            self._u = linear_operator_util.convert_nonref_to_tensor(u, name="u")
            if v is None:
                self._v = self._u
            else:
                self._v = linear_operator_util.convert_nonref_to_tensor(v, name="v")

            if update_operator is None:
                self._update_operator = None
            else:
                self._update_operator = linear_operator_util.convert_nonref_to_tensor(
                    update_operator, name="update_operator")

            # Create base_operator L.
            self._base_operator = base_operator

            super(MyLinearOperatorLowRankUpdate, self).__init__(
                dtype=self._base_operator.dtype,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                parameters=parameters,
                name=name)

            # Create the diagonal operator D.
            self._set_update_operators(update_operator, is_update_positive_definite)
            self._is_update_positive_definite = is_update_positive_definite

            self._check_shapes()

    def _check_shapes(self):
        """Static check that shapes are compatible."""
        # Broadcast shape also checks that u and v are compatible.
        uv_shape = array_ops.broadcast_static_shape(
            self.u.shape, self.v.shape)

        batch_shape = array_ops.broadcast_static_shape(
            self.base_operator.batch_shape, uv_shape[:-2])

        tensor_shape.Dimension(
            self.base_operator.domain_dimension).assert_is_compatible_with(
                uv_shape[-2])

        if self._update_operator is not None:
            tensor_shape.dimension_at_index(uv_shape, -1).assert_is_compatible_with(
                self._update_operator.shape[-1])
            array_ops.broadcast_static_shape(
                batch_shape, self._update_operator.shape[:-1])

    def _set_update_operators(self, update_operator, is_update_positive_definite):
        """Set attributes self._update_operator and self._update_operator."""
        if update_operator is not None:
            self._update_operator = update_operator
        else:
            if tensor_shape.dimension_value(self.u.shape[-1]) is not None:
                r = tensor_shape.dimension_value(self.u.shape[-1])
            else:
                r = array_ops.shape(self.u)[-1]
            self._update_operator = linear_operator_identity.LinearOperatorIdentity(
                num_rows=r, dtype=self.dtype)

    @property
    def u(self):
        """If this operator is `A = L + U D V^H`, this is the `U`."""
        return self._u

    @property
    def v(self):
        """If this operator is `A = L + U D V^H`, this is the `V`."""
        return self._v

    @property
    def is_update_positive_definite(self):
        """If this operator is `A = L + U D V^H`, this hints `D > 0` elementwise."""
        return self._is_update_positive_definite

    @property
    def update_operator(self):
        """If this operator is `A = L + U D V^H`, this is `D`."""
        return self._update_operator

    @property
    def base_operator(self):
        """If this operator is `A = L + U D V^H`, this is the `L`."""
        return self._base_operator

    def _assert_self_adjoint(self):
        # Recall this operator is:
        #   A = L + UDV^H.
        # So in one case self-adjoint depends only on L
        if self.u is self.v and self.update_operator is None:
            return self.base_operator.assert_self_adjoint()
        # In all other cases, sufficient conditions for self-adjoint can be found
        # efficiently. However, those conditions are not necessary conditions.
        return super(MyLinearOperatorLowRankUpdate, self).assert_self_adjoint()

    def _shape(self):
        batch_shape = array_ops.broadcast_static_shape(
            self.base_operator.batch_shape,
            self.update_operator.batch_shape)
        batch_shape = array_ops.broadcast_static_shape(
            batch_shape,
            self.u.shape[:-2])
        batch_shape = array_ops.broadcast_static_shape(
            batch_shape,
            self.v.shape[:-2])
        return batch_shape.concatenate(self.base_operator.shape[-2:])

    def _shape_tensor(self):
        batch_shape = array_ops.broadcast_dynamic_shape(
            self.base_operator.batch_shape_tensor(),
            self.update_operator.batch_shape_tensor())
        batch_shape = array_ops.broadcast_dynamic_shape(
            batch_shape,
            array_ops.shape(self.u)[:-2])
        batch_shape = array_ops.broadcast_dynamic_shape(
            batch_shape,
            array_ops.shape(self.v)[:-2])
        return array_ops.concat(
            [batch_shape, self.base_operator.shape_tensor()[-2:]], axis=0)

    def _get_uv_as_tensors(self):
        """Get (self.u, self.v) as tensors (in case they were refs)."""
        u = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.u)
        if self.v is self.u:
            v = u
        else:
            v = tensor_conversion.convert_to_tensor_v2_with_dispatch(self.v)
        return u, v

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        u, v = self._get_uv_as_tensors()
        l = self.base_operator
        d = self.update_operator

        leading_term = l.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

        if adjoint:
            uh_x = math_ops.matmul(u, x, adjoint_a=True, adjoint_b=adjoint_arg)
            d_uh_x = d.matmul(uh_x, adjoint=adjoint)
            v_d_uh_x = math_ops.matmul(v, d_uh_x)
            return leading_term + v_d_uh_x
        else:
            vh_x = math_ops.matmul(v, x, adjoint_a=True, adjoint_b=adjoint_arg)
            d_vh_x = d.matmul(vh_x, adjoint=adjoint)
            u_d_vh_x = math_ops.matmul(u, d_vh_x)
            return leading_term + u_d_vh_x

    def _determinant(self):
        if self.is_positive_definite:
            return math_ops.exp(self.log_abs_determinant())
        # The matrix determinant lemma gives
        # https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        #   det(L + UDV^H) = det(D^{-1} + V^H L^{-1} U) det(D) det(L)
        #                  = det(C) det(D) det(L)
        # where C is sometimes known as the capacitance matrix,
        #   C := D^{-1} + V^H L^{-1} U
        u, v = self._get_uv_as_tensors()
        det_c = linalg_ops.matrix_determinant(self._make_capacitance(u=u, v=v))
        det_d = self.update_operator.determinant()
        det_l = self.base_operator.determinant()
        return det_c * det_d * det_l

    def _diag_part(self):
        # [U D V^T]_{ii} = sum_{jk} U_{ij} D_{jk} V_{ik}
        #                = sum_{j}  U_{ij} D_{jj} V_{ij}
        d = self._update_operator
        u, v = self._get_uv_as_tensors()
        if d is not None:
            d_vt = d.matmul(v, adjoint_arg=True)
            u_d_vt = u * tf.transpose(d_vt)
            diag_update_part = math_ops.reduce_sum(u_d_vt, axis=-1)
        else:
            diag_update_part = u * math_ops.conj(v)
        return self.base_operator.diag_part() + diag_update_part

    def _log_abs_determinant(self):
        u, v = self._get_uv_as_tensors()
        # Rcall
        #   det(L + UDV^H) = det(D^{-1} + V^H L^{-1} U) det(D) det(L)
        #                  = det(C) det(D) det(L)
        log_abs_det_d = self.update_operator.log_abs_determinant()
        log_abs_det_l = self.base_operator.log_abs_determinant()

        if self._use_cholesky:
            chol_cap_diag = array_ops.matrix_diag_part(
                linalg_ops.cholesky(self._make_capacitance(u=u, v=v)))
            log_abs_det_c = 2 * math_ops.reduce_sum(
                math_ops.log(chol_cap_diag), axis=[-1])
        else:
            det_c = linalg_ops.matrix_determinant(self._make_capacitance(u=u, v=v))
            log_abs_det_c = math_ops.log(math_ops.abs(det_c))
            if self.dtype.is_complex:
                log_abs_det_c = math_ops.cast(log_abs_det_c, dtype=self.dtype)

        return log_abs_det_c + log_abs_det_d + log_abs_det_l

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        if self.base_operator.is_non_singular is False:
            raise ValueError(
                "Solve not implemented unless this is a perturbation of a "
                "non-singular LinearOperator.")
        # The Woodbury formula gives:
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        #   (L + UDV^H)^{-1}
        #   = L^{-1} - L^{-1} U (D^{-1} + V^H L^{-1} U)^{-1} V^H L^{-1}
        #   = L^{-1} - L^{-1} U C^{-1} V^H L^{-1}
        # where C is the capacitance matrix, C := D^{-1} + V^H L^{-1} U
        # Note also that, with ^{-H} being the inverse of the adjoint,
        #   (L + UDV^H)^{-H}
        #   = L^{-H} - L^{-H} V C^{-H} U^H L^{-H}
        l = self.base_operator
        if adjoint:
            # If adjoint, U and V have flipped roles in the operator.
            v, u = self._get_uv_as_tensors()
            # Capacitance should still be computed with u=self.u and v=self.v, which
            # after the "flip" on the line above means u=v, v=u. I.e. no need to
            # "flip" in the capacitance call, since the call to
            # matrix_solve_with_broadcast below is done with the `adjoint` argument,
            # and this takes care of things.
            capacitance = self._make_capacitance(u=v, v=u)
        else:
            u, v = self._get_uv_as_tensors()
            capacitance = self._make_capacitance(u=u, v=v)

        # L^{-1} rhs
        linv_rhs = l.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        # V^H L^{-1} rhs
        vh_linv_rhs = math_ops.matmul(v, linv_rhs, adjoint_a=True)
        # C^{-1} V^H L^{-1} rhs
        if self._use_cholesky:
            capinv_vh_linv_rhs = linalg_ops.cholesky_solve(
                linalg_ops.cholesky(capacitance), vh_linv_rhs)
        else:
            capinv_vh_linv_rhs = linear_operator_util.matrix_solve_with_broadcast(
                capacitance, vh_linv_rhs, adjoint=adjoint)
        # U C^{-1} V^H M^{-1} rhs
        u_capinv_vh_linv_rhs = math_ops.matmul(u, capinv_vh_linv_rhs)
        # L^{-1} U C^{-1} V^H L^{-1} rhs
        linv_u_capinv_vh_linv_rhs = l.solve(u_capinv_vh_linv_rhs, adjoint=adjoint)

        # L^{-1} - L^{-1} U C^{-1} V^H L^{-1}
        return linv_rhs - linv_u_capinv_vh_linv_rhs

    def _make_capacitance(self, u, v):
        # C := D^{-1} + V^H L^{-1} U
        # which is sometimes known as the "capacitance" matrix.
        # L^{-1} U
        linv_u = self.base_operator.solve(u)
        # V^H L^{-1} U
        vh_linv_u = math_ops.matmul(v, linv_u, adjoint_a=True)
        # D^{-1} + V^H L^{-1} V
        capacitance = self._update_operator.inverse().add_to_tensor(vh_linv_u)
        return capacitance

    @property
    def _composite_tensor_fields(self):
        return ("base_operator", "u", "update_operator", "v", "is_update_positive_definite")

    @property
    def _experimental_parameter_ndims_to_matrix_ndims(self):
        return {
            "base_operator": 0,
            "u": 2,
            "update_operator": 1,
            "v": 2
        }
