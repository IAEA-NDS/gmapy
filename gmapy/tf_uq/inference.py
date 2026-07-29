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


def determine_MAP_estimate_trust_region(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    neg_log_prob_gn_hessian=None, max_iters=100, tolerance=1e-8,
    init_damping=1e-3, switch_damping=1e-5, geodesic_accel=True,
    accel_ratio=0.75, accel_fd_step=0.1,
    must_converge=True, ret_optres=False
):
    """Determine the MAP estimate by a trust-region Newton iteration.

    The (negative log posterior) Hessian is damped in
    Levenberg-Marquardt style with a dimensionless damping parameter
    relative to its largest eigenvalue, adapted according to the
    ratio of actual to predicted decrease of the objective. As the
    damped system is solved in the eigenbasis of the Hessian,
    damping adjustments do not require recomputing the Hessian.

    If a Gauss-Newton style Hessian function is provided via
    `neg_log_prob_gn_hessian` (e.g. relying on the GLS approximation
    of the likelihood Hessian, which is positive definite and a
    robust curvature model far away from the optimum), it is used
    during the globalization phase and replaced by the exact Hessian
    of `neg_log_prob_hessian` once the damping has decayed below
    `switch_damping`.

    With `geodesic_accel=True` the step is augmented by the geodesic
    acceleration of Transtrum & Sethna: the second-order correction
    `a = -H_damped^-1 k`, with `k` the second directional derivative
    of the gradient along the step obtained by a finite difference
    (one extra gradient evaluation), lets the iteration follow the
    long curved valleys of sloppy-model posteriors instead of
    chording across them. A step is only taken with acceleration if
    `|a| <= accel_ratio * |d|`; otherwise the damping is increased.
    """
    x = tf.reshape(tf.convert_to_tensor(startvals, tf.float64), (-1,))
    fval_t, grad = neg_log_prob_and_gradient(x)
    fval = float(fval_t)
    use_gn = neg_log_prob_gn_hessian is not None
    damping = init_damping
    converged = False
    num_iters = 0
    need_hessian = True
    rho = np.nan
    while not converged and num_iters < max_iters:
        num_iters += 1
        if use_gn and damping < switch_damping:
            use_gn = False
            need_hessian = True
            print('#  switching from Gauss-Newton to exact Hessian')
        if need_hessian:
            hessfun = (neg_log_prob_gn_hessian if use_gn
                       else neg_log_prob_hessian)
            hess = hessfun(x)
            eigvals, eigvecs = tf.linalg.eigh(
                (hess + tf.transpose(hess)) / 2.
            )
            emax = float(tf.reduce_max(tf.abs(eigvals)))
            eigvals = tf.maximum(eigvals, 1e-12 * emax)
            need_hessian = False
        c = tf.reshape(
            tf.matmul(eigvecs, tf.reshape(grad, (-1, 1)), adjoint_a=True),
            (-1,)
        )
        gmax = float(tf.reduce_max(tf.abs(grad)))
        if gmax <= tolerance:
            converged = True
            break
        # predicted decrease of the undamped Newton step below the
        # round-off level of the objective means convergence to
        # numerical precision (for the current curvature model)
        newton_decr_half = 0.5 * float(tf.reduce_sum(c * c / eigvals))
        if newton_decr_half <= 1e-14 * (1. + abs(fval)):
            if use_gn:
                use_gn = False
                need_hessian = True
                print('#  switching from Gauss-Newton to exact Hessian')
                continue
            converged = True
            break
        accepted = False
        accel_frac = 0.
        while not accepted and damping < 1e8:
            denom = eigvals + damping * emax
            dcoef = -c / denom
            direction = tf.reshape(
                tf.matmul(eigvecs, tf.reshape(dcoef, (-1, 1))), (-1,)
            )
            pred_decr = float(
                tf.reduce_sum(c * c / denom)
                - 0.5 * tf.reduce_sum(eigvals * dcoef * dcoef)
            )
            step = direction
            if geodesic_accel:
                h = accel_fd_step
                _, grad_h = neg_log_prob_and_gradient(x + h * direction)
                hess_d = tf.reshape(
                    tf.matmul(
                        eigvecs, tf.reshape(eigvals * dcoef, (-1, 1))
                    ), (-1,)
                )
                kvec = 2. * (grad_h - grad - h * hess_d) / (h * h)
                ck = tf.reshape(
                    tf.matmul(
                        eigvecs, tf.reshape(kvec, (-1, 1)), adjoint_a=True
                    ), (-1,)
                )
                accel = tf.reshape(
                    tf.matmul(eigvecs, tf.reshape(-ck / denom, (-1, 1))),
                    (-1,)
                )
                anorm = float(tf.linalg.norm(accel))
                dnorm = float(tf.linalg.norm(direction))
                if not np.isfinite(anorm) or anorm > accel_ratio * dnorm:
                    damping *= 4.
                    continue
                accel_frac = anorm / dnorm if dnorm > 0. else 0.
                step = direction + 0.5 * accel
            xnew = x + step
            fnew_t, gnew = neg_log_prob_and_gradient(xnew)
            fnew = float(fnew_t)
            rho = (fval - fnew) / pred_decr if pred_decr > 0. else -1.
            if np.isfinite(fnew) and fnew < fval:
                accepted = True
                x, fval, grad = xnew, fnew, gnew
                if rho > 0.75:
                    # NOTE: the damping floor must be far below the
                    #   ratio of smallest to largest Hessian
                    #   eigenvalue, otherwise the sloppy directions
                    #   never see an (almost) undamped Newton step
                    damping = max(0.5 * damping, 1e-14)
                elif rho < 0.25:
                    damping *= 4.
            else:
                damping *= 4.
        if not accepted:
            break
        need_hessian = True
        print(f'#  tr iteration {num_iters} '
              f'({"gn" if use_gn else "exact"}): fval={fval:.8e}  '
              f'max|grad|={float(tf.reduce_max(tf.abs(grad))):.3e}  '
              f'damping={damping:.1e}  rho={rho:.2f}  '
              f'accel={accel_frac:.2f}')

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. ' +
            'Try increasing `max_iters` and/or `init_damping`'
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
