import tensorflow as tf


class CovarianceModel:
    """Base class for parametrized covariance matrices C0(u).

    Provides the covariance matrix as a linear operator and the
    covariance-parameter related blocks of log-probability Hessians.
    The blocks are defined purely in terms of C0(u) with a constant
    vector z and a constant matrix K:

        cross block:  d/du [ K^T C0(u)^-1 z ]
        covpar block: d2/du2 [ -1/2 z^T C0(u)^-1 z
                               (- 1/2 logdet C0(u) if include_logdet) ]

    The scaling of z and K needed to express the blocks of a
    likelihood with relative covariance matrix in terms of C0 is the
    responsibility of the caller.
    """

    def operator(self, u):
        raise NotImplementedError('please implement this method')

    def covpar_blocks(self, u, kmat, z, include_logdet=True):
        raise NotImplementedError('please implement this method')


class GenericCovarianceModel(CovarianceModel):
    """Model wrapping a function u -> LinearOperator.

    The derivative blocks are computed with gradient tapes, which
    involves one backward pass through the solve graph per
    covariance parameter (and second-order passes for the covpar
    block). Structured covariance models like
    `LowRankCovarianceModel` provide much faster closed forms.
    """

    def __init__(self, like_cov_fun):
        self._like_cov_fun = like_cov_fun

    def operator(self, u):
        return self._like_cov_fun(u)

    def _cross_block(self, u, kmat, z):
        z = tf.reshape(z, (-1, 1))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(u)
            cov0 = self.operator(u)
            v = cov0.solve(z)
            uvec = tf.reshape(tf.matmul(kmat, v, adjoint_a=True), (-1,))
        g = tape.jacobian(
            uvec, u, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return g

    def _chisqr_block(self, u, z):
        z = tf.reshape(z, (-1, 1))
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(u)
            with tf.GradientTape() as tape2:
                tape2.watch(u)
                cov0 = self.operator(u)
                val = -0.5 * tf.matmul(
                    tf.transpose(z), cov0.solve(z)
                )
            g = tape2.gradient(
                val, u, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
        h = tape1.jacobian(
            g, u, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return h

    def _logdet_block(self, u):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(u)
            with tf.GradientTape() as tape2:
                tape2.watch(u)
                cov0 = self.operator(u)
                val = -0.5 * cov0.log_abs_determinant()
            g = tape2.gradient(
                val, u, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
        h = tape1.jacobian(
            g, u, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return h

    def covpar_blocks(self, u, kmat, z, include_logdet=True):
        cross = self._cross_block(u, kmat, z)
        covpar = self._chisqr_block(u, z)
        if include_logdet:
            covpar += self._logdet_block(u)
        return cross, covpar


class LowRankCovarianceModel(CovarianceModel):
    """Covariance model C0(u) = A + S diag(d(u)) S^T.

    `base_operator` is the constant linear operator A, `smat` the
    constant matrix S and `diag_fun` the function u -> d(u) mapping
    the covariance parameters to the diagonal of the update (e.g.
    scattering squared USU uncertainties to their energy grid
    points). The derivative blocks follow in closed form from
    Woodbury identities: with W = C0^-1 S, V = S^T W, q = W^T z,
    B = diag(q) d'(u) and M = K^T W

        cross block  = -M B
        chisqr block = 1/2 d2[q^2] - B^T V B
        logdet block = -1/2 d2[diag(V)] + 1/2 d'(u)^T (V*V) d'(u)

    where d2[w] denotes the second derivative of w^T d(u) and *
    the elementwise product. Only derivatives of the small function
    `diag_fun` are obtained with tapes, so the cost is dominated by
    the single wide solve W.
    """

    def __init__(self, base_operator, smat, diag_fun):
        self._base_operator = base_operator
        self._smat = tf.convert_to_tensor(smat, dtype=tf.float64)
        self._diag_fun = diag_fun

    def operator(self, u):
        return tf.linalg.LinearOperatorLowRankUpdate(
            self._base_operator, self._smat, self._diag_fun(u),
            is_self_adjoint=True, is_positive_definite=True,
            is_diag_update_positive=True
        )

    def _diag_jacobian(self, u):
        with tf.GradientTape() as tape:
            tape.watch(u)
            d = self._diag_fun(u)
        return tape.jacobian(
            d, u, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

    def _weighted_diag_hessian(self, u, w):
        with tf.GradientTape() as t2:
            t2.watch(u)
            with tf.GradientTape() as t1:
                t1.watch(u)
                s = tf.reduce_sum(w * self._diag_fun(u))
            g = t1.gradient(
                s, u, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )
        return t2.jacobian(
            g, u, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

    def covpar_blocks(self, u, kmat, z, include_logdet=True):
        u = tf.convert_to_tensor(u, dtype=tf.float64)
        cov0 = self.operator(u)
        wmat = cov0.solve(self._smat)
        vmat = tf.matmul(self._smat, wmat, adjoint_a=True)
        q = tf.reshape(
            tf.matmul(wmat, tf.reshape(z, (-1, 1)), adjoint_a=True), (-1,)
        )
        jd = self._diag_jacobian(u)
        bmat = tf.reshape(q, (-1, 1)) * jd
        mmat = tf.matmul(kmat, wmat, adjoint_a=True)
        cross = -tf.matmul(mmat, bmat)
        covpar = 0.5 * self._weighted_diag_hessian(u, q * q)
        covpar -= tf.matmul(bmat, tf.matmul(vmat, bmat), adjoint_a=True)
        if include_logdet:
            vdiag = tf.linalg.diag_part(vmat)
            covpar -= 0.5 * self._weighted_diag_hessian(u, vdiag)
            covpar += 0.5 * tf.matmul(
                jd, tf.matmul(vmat * vmat, jd), adjoint_a=True
            )
        return cross, covpar
