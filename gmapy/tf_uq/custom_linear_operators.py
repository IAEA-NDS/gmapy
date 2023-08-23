import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops


class MyLinearOperatorLowRankUpdate(tf.linalg.LinearOperatorLowRankUpdate):

    def _log_abs_determinant(self):
        u, v = self._get_uv_as_tensors()
        # Recall
        #   det(L + UDV^H) = det(D^{-1} + V^H L^{-1} U) det(D) det(L)
        #                  = det(C) det(D) det(L)
        log_abs_det_d = self.diag_operator.log_abs_determinant()
        log_abs_det_l = self.base_operator.log_abs_determinant()

        if self._use_cholesky:
            chol_cap_diag = array_ops.matrix_diag_part(
              linalg_ops.cholesky(self._make_capacitance(u=u, v=v)))
            log_abs_det_c = 2 * math_ops.reduce_sum(
              math_ops.log(chol_cap_diag), axis=[-1])
        else:
            # FIXME: for complex matrices, this would yield the wrong result
            capa = self._make_capacitance(u=u, v=v)
            log_abs_det_c = linalg_ops.log_matrix_determinant(capa).log_abs_determinant
            if self.dtype.is_complex:
                log_abs_det_c = math_ops.cast(log_abs_det_c, dtype=self.dtype)
        return log_abs_det_c + log_abs_det_d + log_abs_det_l
