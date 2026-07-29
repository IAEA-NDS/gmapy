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


def determine_MAP_estimate_lbfgs(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    max_inner_iters=3000, max_outer_iters=10, nugget=1e-4,
    num_correction_pairs=20, tolerance=1e-5,
    must_converge=True, ret_optres=False
):
    """Determine the MAP estimate by preconditioned L-BFGS.

    In each outer iteration the Hessian provided by
    `neg_log_prob_hessian` is evaluated at the reference point, made
    positive definite and used to transform the parameters
    (x = x_ref + L y with L L^T the inverse Hessian), so that the
    transformed problem has unit curvature at the reference point.
    It is then minimized by the limited-memory BFGS algorithm, whose
    per-iteration cost is O(num_correction_pairs * dim) in contrast
    to the dense matrix algebra per iteration of
    `determine_MAP_estimate`, while the accumulated correction pairs
    still capture the valley curvature of sloppy posteriors along
    the trajectory. The gradient norm of the transformed problem is
    the Newton decrement of the original one, so `tolerance` is
    dimensionless; its round-off floor is about
    sqrt(2e-14 * |objective|).
    """
    x = tf.reshape(tf.convert_to_tensor(startvals, tf.float64), (-1,))
    n = int(x.shape[0])
    converged = False
    total_inner = 0
    fval = None
    for outer_iter in range(1, max_outer_iters + 1):
        print(f'#  outer iteration {outer_iter}')
        hess = neg_log_prob_hessian(x)
        hess_pd = make_positive_definite(hess, nugget)
        inv_hess = invert_symmetric_matrix(hess_pd)
        inv_hess = make_positive_definite(inv_hess, nugget)
        lmat = tf.linalg.cholesky(inv_hess)
        x_ref = x

        def transformed_fun(y):
            xcur = x_ref + tf.linalg.matvec(lmat, y)
            f, g = neg_log_prob_and_gradient(xcur)
            return f, tf.linalg.matvec(lmat, g, adjoint_a=True)

        optres = tfp.optimizer.lbfgs_minimize(
            transformed_fun,
            initial_position=tf.zeros((n,), dtype=tf.float64),
            num_correction_pairs=num_correction_pairs,
            tolerance=tolerance,
            max_iterations=max_inner_iters
        )
        total_inner += int(optres.num_iterations)
        x = x_ref + tf.linalg.matvec(lmat, optres.position)
        fval = float(optres.objective_value)
        converged = bool(optres.converged)
        print(f'-- inner iterations: {int(optres.num_iterations)}  '
              f'fval={fval:.8e}  converged={converged}')
        if converged:
            break

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. Try increasing ' +
            '`max_inner_iters` and/or `max_outer_iters`'
        )
    if ret_optres:
        OptRes = namedtuple(
            'OptimizationResult',
            ['position', 'converged', 'num_iterations', 'objective_value']
        )
        return OptRes(x, converged, total_inner, fval)
    else:
        return x


def determine_MAP_estimate_precond_lbfgs(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    max_iters=5000, num_correction_pairs=30, tolerance=1e-5,
    nugget=1e-4, hessian_refresh_interval=100, armijo_c1=1e-4,
    max_step_halvings=30, must_converge=True, ret_optres=False
):
    """Determine the MAP estimate by preconditioned L-BFGS with a
    persistent correction-pair history.

    In contrast to `determine_MAP_estimate_lbfgs`, which relies on
    `tfp.optimizer.lbfgs_minimize` and therefore discards the
    accumulated correction pairs whenever the preconditioner is
    renewed, this implementation applies the inverse of the
    positive-definite projection of the Hessian provided by
    `neg_log_prob_hessian` directly as initial matrix of the L-BFGS
    two-loop recursion. The Hessian can hence be refreshed (every
    `hessian_refresh_interval` accepted steps and on line-search
    failure) WITHOUT discarding the correction pairs, which carry
    the valley curvature information accumulated along the
    trajectory. `tolerance` refers to the Newton decrement measured
    in the metric of the preconditioner; its round-off floor is
    about sqrt(2e-14 * |objective|).
    """
    def nlpg_np(xv):
        f, g = neg_log_prob_and_gradient(
            tf.constant(xv, dtype=tf.float64)
        )
        return float(f), np.array(g)

    def refresh_h0(xv):
        hess = np.array(
            neg_log_prob_hessian(tf.constant(xv, dtype=tf.float64))
        )
        hess = 0.5 * (hess + hess.T)
        eigvals, eigvecs = np.linalg.eigh(hess)
        return eigvecs, eigvals

    def h0inv(vec):
        return qmat @ ((qmat.T @ vec) / evals)

    x = np.array(
        tf.reshape(tf.convert_to_tensor(startvals, tf.float64), (-1,))
    )
    fval, grad = nlpg_np(x)
    qmat, evals_raw = refresh_h0(x)
    # saddle-free treatment of indefiniteness: the magnitude of
    # negative eigenvalues acts as damping, so that directions of
    # strong negative curvature receive small steps instead of the
    # huge ones a positive-clipping of the spectrum would produce
    cur_nugget = nugget
    evals = np.maximum(np.abs(evals_raw), cur_nugget)
    iters_since_refresh = 0
    pairs = []
    converged = False
    num_iters = 0
    while num_iters < max_iters:
        # convergence: Newton decrement in the preconditioner metric
        dec_sq = float(grad.dot(h0inv(grad)))
        if (np.sqrt(max(dec_sq, 0.)) <= tolerance
                or 0.5 * dec_sq <= 1e-14 * (1. + abs(fval))):
            converged = True
            break
        num_iters += 1
        # two-loop recursion with explicit initial matrix
        qvec = grad.copy()
        alphas = []
        for svec, yvec, rho in reversed(pairs):
            a = rho * svec.dot(qvec)
            alphas.append(a)
            qvec -= a * yvec
        rvec = h0inv(qvec)
        for (svec, yvec, rho), a in zip(pairs, reversed(alphas)):
            b = rho * yvec.dot(rvec)
            rvec += (a - b) * svec
        direction = -rvec
        dgrad = float(grad.dot(direction))
        if not np.isfinite(dgrad) or dgrad >= 0.:
            # stale correction pairs: fall back to the preconditioner
            pairs = []
            direction = -h0inv(grad)
            dgrad = float(grad.dot(direction))
        # backtracking line search
        alpha = 1.
        accepted = False
        for _ in range(max_step_halvings):
            xnew = x + alpha * direction
            fnew, gnew = nlpg_np(xnew)
            if (np.isfinite(fnew)
                    and fnew <= fval + armijo_c1 * alpha * dgrad):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if iters_since_refresh > 0:
                # renew preconditioner at the current position but
                # keep the correction pairs
                qmat, evals_raw = refresh_h0(x)
                cur_nugget = nugget
                evals = np.maximum(np.abs(evals_raw), cur_nugget)
                iters_since_refresh = 0
                print(f'#  iter {num_iters}: line search failed, '
                      'refreshed Hessian')
                continue
            if pairs:
                pairs = []
                print(f'#  iter {num_iters}: line search failed, '
                      'dropped correction pairs')
                continue
            if cur_nugget <= 1e8 * nugget:
                # last resort: escalate the eigenvalue floor of the
                # preconditioner (Levenberg-style damping)
                cur_nugget *= 10.
                evals = np.maximum(np.abs(evals_raw), cur_nugget)
                print(f'#  iter {num_iters}: line search failed, '
                      f'escalated damping to {cur_nugget:.1e}')
                continue
            print(f'#  iter {num_iters}: line search failed, giving up')
            break
        svec = xnew - x
        yvec = gnew - grad
        sy = float(svec.dot(yvec))
        if sy > 1e-10 * np.linalg.norm(svec) * np.linalg.norm(yvec):
            pairs.append((svec, yvec, 1. / sy))
            if len(pairs) > num_correction_pairs:
                pairs.pop(0)
        x, fval, grad = xnew, fnew, gnew
        iters_since_refresh += 1
        if iters_since_refresh >= hessian_refresh_interval:
            qmat, evals_raw = refresh_h0(x)
            cur_nugget = nugget
            evals = np.maximum(np.abs(evals_raw), cur_nugget)
            iters_since_refresh = 0
        if num_iters % 25 == 0:
            print(f'#  iter {num_iters}: fval={fval:.8e}  '
                  f'max|grad|={np.max(np.abs(grad)):.3e}  '
                  f'decrement={np.sqrt(max(dec_sq, 0.)):.3e}  '
                  f'alpha={alpha:.2e}  npairs={len(pairs)}')

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. Try increasing ' +
            '`max_iters`'
        )
    if ret_optres:
        OptRes = namedtuple(
            'OptimizationResult',
            ['position', 'converged', 'num_iterations', 'objective_value']
        )
        return OptRes(
            tf.constant(x, dtype=tf.float64), converged, num_iters, fval
        )
    else:
        return tf.constant(x, dtype=tf.float64)


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
