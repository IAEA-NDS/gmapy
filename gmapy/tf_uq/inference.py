from collections import namedtuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .auxiliary import (
    make_positive_definite,
    invert_symmetric_matrix
)
tfb = tfp.bijectors


def iterative_gls_estimate(
    startvals, propfun, jacfun, data, cov_linop_fun, rel_tol=1e-6, max_iters=50,
    rel_damp_unc=1000, must_converge=True, ret_optres=False
):
    curvals = startvals
    converged = False
    rel_damp_unc = tf.constant(rel_damp_unc, dtype=tf.float64)
    for cur_iter in range(max_iters):
        print(f'iteration {cur_iter}')
        expcov_linop = cov_linop_fun(curvals)
        S = jacfun(curvals)
        if isinstance(S, tf.sparse.SparseTensor):
            ST = tf.sparse.transpose(S)
            S_dense = tf.sparse.to_dense(S)
            inv_postcov = tf.sparse.sparse_dense_matmul(ST, expcov_linop.solve(S_dense))
        else:
            ST = tf.transpose(S)
            inv_postcov = ST @ expcov_linop.solve(S)
        # poor-man LM algorithm: constant damping term
        damp_abs = 1/tf.square(rel_damp_unc) * tf.linalg.diag(1/(curvals**2))
        inv_postcov_reg = inv_postcov + damp_abs

        propvals = propfun(curvals)
        d = tf.reshape(data, (-1,1)) - tf.reshape(propvals, (-1,1))
        if isinstance(S, tf.sparse.SparseTensor):
            rhs = tf.sparse.sparse_dense_matmul(ST, expcov_linop.solve(d))
        else:
            rhs = ST @ expcov_linop.solve(d)
        delta = tf.reshape(tf.linalg.solve(inv_postcov_reg, rhs), (-1,))
        curvals = curvals + delta

        relative_change = tf.linalg.norm(delta) / tf.linalg.norm(curvals)
        print(f'relative change: {relative_change}')
        if tf.linalg.norm(delta) < rel_tol * tf.linalg.norm(curvals):
            converged = True
            break

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine iterative GLS estimate. Try increasing `max_iters`'
        )

    if ret_optres:
        OptRes = namedtuple('OptimizationResult', ['position', 'converged', 'num_iterations'])
        return OptRes(
            curvals, converged, cur_iter
        )
    else:
        return curvals


def determine_MAP_estimate(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    max_inner_iters=500, max_outer_iters=10, nugget=1e-4,
    must_converge=True, ret_optres=False, compile_inner_iterations=False
):
    if isinstance(max_inner_iters, int):
        max_inner_iters = np.full(max_outer_iters, max_inner_iters,
                                  dtype=np.int32)
    # NOTE: compiling the BFGS loop removes the eager dispatch
    #   overhead but NOT the dense inverse-Hessian update of BFGS
    #   (O(n^2..n^3) per iteration), which dominates for large
    #   parameter counts; in graph mode the loop was also observed
    #   to consume much more memory than the eager one, so this is
    #   opt-in
    bfgs_minimize = tfp.optimizer.bfgs_minimize
    if compile_inner_iterations:
        bfgs_minimize = tf.function(tfp.optimizer.bfgs_minimize)
    outer_iter = 0
    converged = False
    refvals = startvals
    while not converged and outer_iter < max_outer_iters:
        max_inner_iter = \
            tf.constant(max_inner_iters[outer_iter], dtype=tf.int32)
        outer_iter += 1
        print(f'#  outer iteration {outer_iter}')
        print(f'-- running inner iteration with {max_inner_iter} iterations')
        # obtain an approximation of the posterior covariance matrix to aid optimization
        neg_log_post_hessian = neg_log_prob_hessian(refvals)
        fixed_neg_log_post_hessian = \
            make_positive_definite(neg_log_post_hessian, nugget)
        inv_neg_log_post_hessian = \
            invert_symmetric_matrix(fixed_neg_log_post_hessian)
        fixed_inv_neg_log_post_hessian = \
            make_positive_definite(inv_neg_log_post_hessian, nugget)
        # find peak of posterior distribution (to use it as a starting value of MCMC)
        optres = bfgs_minimize(
            neg_log_prob_and_gradient, initial_position=refvals,
            initial_inverse_hessian_estimate=fixed_inv_neg_log_post_hessian,
            max_iterations=max_inner_iter
        )
        converged = optres.converged.numpy()
        refvals = optres.position
        print(f'Inner iterations: {optres.num_iterations}')

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. ' +
            'Try increasing `max_inner_iters` and/or `max_outer_iters`'
        )
    if ret_optres:
        return optres
    else:
        return refvals


def determine_MAP_estimate_newton(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    max_iters=30, nugget=1e-4, tolerance=1e-8, armijo_c1=1e-4,
    max_step_halvings=40, must_converge=True, ret_optres=False
):
    """Determine the MAP estimate by a damped Newton iteration.

    In contrast to `determine_MAP_estimate`, which uses the Hessian
    matrix only as a preconditioner for BFGS inner iterations, the
    curvature information is recomputed via `neg_log_prob_hessian`
    in EVERY iteration. With a cheap (e.g. exact) Hessian this
    converges in a few iterations without the costly dense
    inverse-Hessian updates of BFGS. The Hessian is made positive
    definite by clipping its eigenvalues at `nugget`; if a Newton
    step does not achieve sufficient decrease in the line search,
    the clipping threshold is temporarily escalated
    (Levenberg-style damping).
    """
    x = tf.reshape(tf.convert_to_tensor(startvals, tf.float64), (-1,))
    fval, grad = neg_log_prob_and_gradient(x)
    converged = bool(tf.reduce_max(tf.abs(grad)) <= tolerance)
    num_iters = 0
    while not converged and num_iters < max_iters:
        num_iters += 1
        hess = neg_log_prob_hessian(x)
        cur_nugget = nugget
        step_found = False
        while not step_found and cur_nugget <= 1e8 * nugget:
            hess_pd = make_positive_definite(hess, cur_nugget)
            chol = tf.linalg.cholesky(hess_pd)
            direction = -tf.reshape(
                tf.linalg.cholesky_solve(chol, tf.reshape(grad, (-1, 1))),
                (-1,)
            )
            dgrad = float(tf.reduce_sum(direction * grad))
            if not np.isfinite(dgrad) or dgrad >= 0.:
                cur_nugget *= 10.
                continue
            # the predicted decrease of the Newton step is half the
            # squared Newton decrement; once it drops below the
            # round-off level of the objective, the iterate is
            # converged to numerical precision
            if -dgrad / 2. <= 1e-14 * (1. + abs(float(fval))):
                converged = True
                break
            alpha = 1.
            for _ in range(max_step_halvings):
                xnew = x + alpha * direction
                fnew, gnew = neg_log_prob_and_gradient(xnew)
                if (np.isfinite(float(fnew)) and
                        float(fnew) <= float(fval) + armijo_c1*alpha*dgrad):
                    step_found = True
                    break
                alpha *= 0.5
            if not step_found:
                cur_nugget *= 10.
        if converged or not step_found:
            break
        x, fval, grad = xnew, fnew, gnew
        gmax = float(tf.reduce_max(tf.abs(grad)))
        print(f'#  newton iteration {num_iters}: '
              f'fval={float(fval):.8e}  max|grad|={gmax:.3e}  '
              f'step={alpha:.3e}')
        converged = gmax <= tolerance

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. ' +
            'Try increasing `max_iters` and/or `nugget`'
        )
    if ret_optres:
        OptRes = namedtuple(
            'OptimizationResult',
            ['position', 'converged', 'num_iterations', 'objective_value']
        )
        return OptRes(x, converged, num_iters, fval)
    else:
        return x


def generate_MCMC_chain(
    startvals, log_prob, neg_log_prob_hessian, nugget=1e-10,
    step_size=0.001, num_burnin_steps=100, num_results=1000,
    num_leapfrog_steps=10
):
    # We use the negative Hessian of the logarithmized
    # probability density function as an approximation to the
    # covariance matrix of the targeted distribution.
    neg_log_post_hessian = neg_log_prob_hessian(startvals)
    neg_log_post_hessian = make_positive_definite(neg_log_post_hessian, nugget)
    inv_neg_log_post_hessian = invert_symmetric_matrix(neg_log_post_hessian)
    inv_neg_log_post_hessian = \
        make_positive_definite(inv_neg_log_post_hessian, nugget)
    # We then compute the Cholesky factor of this covariance matrix
    # which is used to define a bijector so that we can propose
    # sample vectors in this transformed space  whose elements are
    # approximately independent and distributed according to a
    # standard normal distribution.
    postcov_chol = tf.linalg.cholesky(inv_neg_log_post_hessian)
    del neg_log_post_hessian
    del inv_neg_log_post_hessian

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps
    )
    precond_bijector = tfb.ScaleMatvecTriL(
        scale_tril=postcov_chol
    )
    trafo_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
        hmc_kernel, precond_bijector
    )
    # NOTE: This class is defined so that the one_step method can be
    # decorated with tf.function to speed up the MCMC sampling.
    # Decorating run_chain below with tf.function leads to
    # main memory usage exceeding 30 GB and the process gets killed.
    class MySimpleStepSizeAdaptation(tfp.mcmc.SimpleStepSizeAdaptation):

        # NOTE: Decoration with tf.function triggers a warning
        #       that the function cannot be converted and will run
        #       as is. However, the observed performance indicates
        #       that the warning message is false and the
        #       function gets compiled.
        @tf.function
        def one_step(self, *state_and_results, **one_step_kwargs):
            return super().one_step(*state_and_results, **one_step_kwargs)
    # We calibrate the user-defined step size to increase sampling efficiency
    adaptive_hmc = MySimpleStepSizeAdaptation(
        inner_kernel=trafo_hmc_kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8)
    )

    def trace_everything(states, previous_kernel_results):
        return previous_kernel_results

    samples, tracing_info = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=startvals,
        kernel=adaptive_hmc,
        trace_fn=trace_everything
    )

    return samples, tracing_info
