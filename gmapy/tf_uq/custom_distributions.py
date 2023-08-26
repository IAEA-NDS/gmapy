import tensorflow as tf
import tensorflow_probability as tfp
import math
tfd = tfp.distributions


class BaseDistribution(tf.Module):

    def log_prob(self, x):
        raise NotImplementedError(
            'please implement this function in derived class'
        )

    def log_prob_and_gradient(self, x):
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            res = self.log_prob(x)
        g = tape.gradient(
            res, x, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return res, g

    def log_prob_gradient(self, x):
        return self.log_prob_and_gradient(x)[1]

    def log_prob_hessian(self, x):
        raise NotImplementedError(
            'please implement this function in derived class'
        )

    # convenience functions for tensorflow minimizer algos
    def neg_log_prob(self, x):
        return -self.log_prob(x)

    def neg_log_prob_and_gradient(self, x):
        res = self.log_prob_and_gradient(x)
        return (-res[0], -res[1])

    def neg_log_prob_gradient(self, x):
        return -self.log_prob_gradient(x)

    def neg_log_prob_hessian(self, x):
        return -self.log_prob_hessian(x)


class DistributionForParameterSubset(BaseDistribution):

    def __init__(self, dist, num_params, idcs=None):
        self._isempty = len(idcs) == 0
        idcs = [] if idcs is None else idcs
        self._num_params = num_params
        self._param_idcs = tf.reshape(idcs, (-1, 1))
        self._dist = dist

    def log_prob(self, x):
        if self._isempty:
            return tf.constant(0., dtype=tf.float64)
        x_red = tf.gather_nd(x, self._param_idcs)
        return self._dist.log_prob(x_red)

    def log_prob_hessian(self, x):
        num_params = self._num_params
        if self._isempty:
            return tf.zeros((num_params, num_params), dtype=tf.float64)
        param_idcs = self._param_idcs
        x_red = tf.gather_nd(x, param_idcs)
        hess_red = self._dist.log_prob_hessian(x_red)
        # scatter into the full hessian
        const_idcs2 = tf.reshape(param_idcs, (-1,))
        row_mesh, col_mesh = tf.meshgrid(const_idcs2, const_idcs2)
        idcs_mesh = tf.reshape(
            tf.stack([row_mesh, col_mesh], axis=-1), (-1, 2)
        )
        flat_hess_red = tf.reshape(hess_red, (-1,))
        hessian = tf.scatter_nd(
            idcs_mesh, flat_hess_red, (num_params, num_params)
        )
        return hessian


class UnnormalizedDistributionProduct(BaseDistribution):

    def __init__(self, distributions):
        self._distributions = distributions

    def log_prob(self, x):
        first = True
        for dist in self._distributions:
            if first:
                res = dist.log_prob(x)
                first = False
            else:
                res = tf.add(res, dist.log_prob(x))
        return res

    def log_prob_hessian(self, x):
        first = True
        for dist in self._distributions:
            if first:
                res = dist.log_prob_hessian(x)
                first = False
            else:
                res = tf.add(res, dist.log_prob_hessian(x))
        return res


class MultivariateNormal(BaseDistribution):

    def __init__(self, prior_loc, prior_scale):
        self._prior_loc = tf.reshape(
            tf.constant(prior_loc, dtype=tf.float64), (-1,)
        )
        self._prior_scale = prior_scale

    def log_prob(self, x):
        pdf = tfd.MultivariateNormalLinearOperator(
            loc=self._prior_loc, scale=self._prior_scale
        )
        return pdf.log_prob(x)

    def log_prob_hessian(self, x):
        prior_scale = self._prior_scale
        covmat = prior_scale.matmul(prior_scale.adjoint())
        hess = (-tf.linalg.inv(covmat).to_dense())
        return hess


class MultivariateNormalLikelihood(BaseDistribution):

    def __init__(self, num_params, propfun, jacfun, like_data, like_scale,
                 approximate_hessian=False, relative=False):
        self._propfun = propfun
        self._jacfun = jacfun
        self._like_data = like_data
        self._num_params = num_params
        self._approximate_hessian = approximate_hessian
        self._like_scale = like_scale
        self._relative = relative

    def _like_scale_fun(self, x):
        scale_op = tf.linalg.LinearOperatorDiag(x)
        comp_op = scale_op.matmul(self._like_scale)
        return comp_op

    def log_prob(self, x):
        propvals = self._propfun(x)
        like_scale = self._like_scale
        if self._relative:
            like_scale = self._like_scale_fun(propvals)
        pdf = tfd.MultivariateNormalLinearOperator(
            loc=propvals, scale=like_scale
        )
        return pdf.log_prob(self._like_data)

    def _log_prob_hessian_gls_part(self, x):
        like_scale = self._like_scale
        if self._relative:
            propvals = self._propfun(x)
            like_scale = self._like_scale_fun(propvals)
        jac = tf.sparse.to_dense(self._jacfun(x))
        u = like_scale.solve(jac)
        return (-tf.matmul(tf.transpose(u), u))

    def _log_prob_hessian_model_part(self, x):
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        propvals = self._propfun(x)
        like_scale = self._like_scale
        if self._relative:
            like_scale = self._like_scale_fun(propvals)
        like_data = tf.reshape(self._like_data, (-1, 1))
        propvals = tf.reshape(propvals, (-1, 1))
        # introduce factor -1 here instead of in front
        # of the Jacobian calculation below, as it is equivalent
        d = (-1) * (like_data - propvals)
        constvec = like_scale.solve(like_scale.solve(d), adjoint=True)
        col_list = []
        for i in range(self._num_params):
            print(f'Hessian elements related to {i}-th parameter')
            with tf.GradientTape() as tape:
                tape.watch(x)
                j = self._jacfun(x)
                u = tf.sparse.sparse_dense_matmul(j, constvec, adjoint_a=True)
                z = tf.gather(u, [[i]])
            g = tape.gradient(z, x)
            col_list.append(g)
        neg_hessian = tf.stack(col_list, axis=0)
        return (-neg_hessian)

    def log_prob_hessian(self, x):
        gls_part = self._log_prob_hessian_gls_part(x)
        if self._approximate_hessian:
            return gls_part
        else:
            model_part = self._log_prob_hessian_model_part(x)
            return gls_part + model_part


class MultivariateNormalLikelihoodWithCovParams(MultivariateNormalLikelihood):

    def __init__(self, num_params, num_covpars, propfun, jacfun,
                 like_data, like_cov_fun,
                 approximate_hessian=False):
        self._propfun = propfun
        self._jacfun = jacfun
        self._like_data = like_data
        self._like_cov_fun = like_cov_fun
        self._num_params = num_params
        self._num_covpars = num_covpars
        self._approximate_hessian = approximate_hessian

    def log_prob(self, x):
        x = tf.reshape(x, (-1,))
        pars, covpars = self.split_pars(x)
        covop = self._like_cov_fun(covpars)
        logdet = covop.log_abs_determinant()
        d = self._like_data - self._propfun(pars)
        chisqr = tf.matmul(
            tf.reshape(d, (1, -1)), covop.solve(tf.reshape(d, (-1, 1)))
        )
        log2pi = tf.math.log(tf.constant(2*math.pi, dtype=tf.float64))
        normfact = tf.cast(tf.size(d), dtype=tf.float64) * log2pi
        res = -0.5 * (normfact + logdet + chisqr)
        return tf.squeeze(res)

    def combine_pars(self, params, covpars):
        return tf.concat([params, covpars], axis=0)

    def split_pars(self, x):
        return tf.split(x, [self._num_params, self._num_covpars])

    def _log_prob_hessian_gls_part(self, like_cov, x):
        jac = tf.sparse.to_dense(self._jacfun(x))
        u = like_cov.solve(jac)
        return (-tf.matmul(tf.transpose(jac), u))

    def _log_prob_hessian_model_part(self, like_cov, x):
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        like_data = tf.reshape(self._like_data, (-1, 1))
        propvals = tf.reshape(self._propfun(x), (-1, 1))
        d = like_data - propvals
        constvec = like_cov.solve(d)
        col_list = []
        for i in range(self._num_params):
            print(f'Hessian elements related to {i}-th parameter')
            with tf.GradientTape() as tape:
                tape.watch(x)
                j = self._jacfun(x)
                u = tf.sparse.sparse_dense_matmul(j, constvec, adjoint_a=True)
                z = tf.gather(u, [[i]])
            g = tape.gradient(z, x)
            col_list.append(g)
        hessian = tf.stack(col_list, axis=0)
        return hessian

    def _log_prob_hessian_offdiag_part(self, x, covpars):
        # compute -dz dH z
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        like_cov_fun = self._like_cov_fun
        like_data = tf.reshape(self._like_data, (-1, 1))
        propvals = tf.reshape(self._propfun(x), (-1, 1))
        d = like_data - propvals
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(covpars)
            j = self._jacfun(x)
            like_cov = like_cov_fun(covpars)
            constvec = like_cov.solve(d)
            u = tf.sparse.sparse_dense_matmul(j, constvec, adjoint_a=True)
            u = tf.reshape(u, (-1,))
        g = tape.jacobian(u, covpars, experimental_use_pfor=True)
        return g

    def _log_prob_hessian_chisqr_wrt_covpars(self, x, covpars):
        # compute -z ddH z
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        like_cov_fun = self._like_cov_fun
        like_data = tf.reshape(self._like_data, (-1, 1))
        propvals = tf.reshape(self._propfun(x), (-1, 1))
        d = like_data - propvals
        d = tf.reshape(d, (-1, 1))
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(covpars)
            with tf.GradientTape() as tape2:
                tape2.watch(covpars)
                like_cov = like_cov_fun(covpars)
                u = -0.5 * tf.matmul(tf.transpose(d), like_cov.solve(d))
            g = tape2.gradient(u, covpars)
        h = tape1.jacobian(g, covpars, experimental_use_pfor=True)
        return h

    def _log_prob_hessian_logdet_wrt_covpars(self, covpars):
        # compute -dd(logdet covmat)
        if not isinstance(covpars, tf.Tensor):
            covpars = tf.constant(covpars, dtype=tf.float64)
        like_cov_fun = self._like_cov_fun
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(covpars)
            with tf.GradientTape() as tape2:
                tape2.watch(covpars)
                like_cov = like_cov_fun(covpars)
                u = -0.5 * like_cov.log_abs_determinant()
            g = tape2.gradient(u, covpars)
        h = tape1.jacobian(g, covpars, experimental_use_pfor=True)
        return h

    # first derivative
    #     2*dz H z + z dH z
    # second derivative
    #     2*ddz H z + 2*dz H dz + (param block)
    #     2*dz dH z (off-diagonal block)
    #     z ddH z (covparam block)
    #     dd logdet H (covparam block)
    def log_prob_hessian(self, x):
        x, covpars = self.split_pars(x)
        like_cov = self._like_cov_fun(covpars)
        pars_part = self._log_prob_hessian_gls_part(like_cov, x)
        if not self._approximate_hessian:
            model_part = self._log_prob_hessian_model_part(like_cov, x)
            pars_part += model_part
        offdiag_part = self._log_prob_hessian_offdiag_part(x, covpars)
        covpar_part = self._log_prob_hessian_logdet_wrt_covpars(covpars)
        covpar_part += self._log_prob_hessian_chisqr_wrt_covpars(x, covpars)
        res1 = tf.concat([pars_part, offdiag_part], axis=1)
        res2 = tf.concat([tf.transpose(offdiag_part), covpar_part], axis=1)
        res = tf.concat([res1, res2], axis=0)
        return res
