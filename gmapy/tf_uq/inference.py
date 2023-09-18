import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .auxiliary import (
    make_positive_definite,
    invert_symmetric_matrix
)
tfb = tfp.bijectors


def determine_MAP_estimate(
    startvals, neg_log_prob_and_gradient, neg_log_prob_hessian,
    max_inner_iters=500, max_outer_iters=10, nugget=1e-4,
    must_converge=True, ret_optres=False
):
    max_outer_iters = 20
    if isinstance(max_inner_iters, int):
        max_inner_iters = np.full(max_outer_iters, max_inner_iters,
                                  dtype=np.int32)
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
        optres = tfp.optimizer.bfgs_minimize(
            neg_log_prob_and_gradient, initial_position=refvals,
            initial_inverse_hessian_estimate=fixed_inv_neg_log_post_hessian,
            max_iterations=max_inner_iter
        )
        converged = optres.converged.numpy()
        refvals = optres.position

    if must_converge and not converged:
        raise ValueError(
            'Unable to determine MAP estimate. ' +
            'Try increasing `max_inner_iters` and/or `max_outer_iters`'
        )
    if ret_optres:
        return optres
    else:
        return refvals


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
