import sys
sys.path.append('../../..')
import time
import gc
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
from gmapy.tf_uq.auxiliary import (
    make_positive_definite,
    invert_symmetric_matrix
)
# NOTE: the numpy import is necessary because
#       a method in the post instance loaded via dill
#       depends on it
import numpy as np
import pandas as pd
tfb = tfp.bijectors


post, likelihood = load_objects(
    'output/01_model_preparation_output.pkl',
    'post', 'likelihood'
)
optres, = load_objects(
    'output/02_parameter_optimization_output.pkl',
    'optres'
)

# define essential input quantities for MCMC
optvals = optres.position
neg_log_post_hessian = post.neg_log_prob_hessian(optvals)
neg_log_post_hessian = make_positive_definite(neg_log_post_hessian, 1e-10)
inv_neg_log_post_hessian = invert_symmetric_matrix(neg_log_post_hessian)
inv_neg_log_post_hessian = make_positive_definite(inv_neg_log_post_hessian, 1e-10)

postcov_chol = tf.linalg.cholesky(inv_neg_log_post_hessian)
target_log_prob = likelihood.log_prob
del neg_log_post_hessian
del inv_neg_log_post_hessian
gc.collect()

# try the Hamilton Monte Carlo approach
# num_results = int(1e3)
# num_burnin_steps = int(2e3)
num_results = int(20e3)
num_burnin_steps = int(2e3)

hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob,
    step_size=0.001,
    num_leapfrog_steps=10
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


# adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
adaptive_hmc = MySimpleStepSizeAdaptation(
    inner_kernel=trafo_hmc_kernel,
    num_adaptation_steps=int(num_burnin_steps * 0.8)
)


def trace_everything(states, previous_kernel_results):
    return previous_kernel_results


# @tf.function
def run_chain():
    samples, tracing_info = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=optvals,
        kernel=adaptive_hmc,
        trace_fn=trace_everything
    )
    return samples, tracing_info


s1 = time.time()
chain, tracing_info = run_chain()
s2 = time.time()
print(f's2-s1: {s2-s1}')

save_objects(
    'output/03_mcmc_sampling_output.pkl', locals(),
    'chain', 'tracing_info'
)
