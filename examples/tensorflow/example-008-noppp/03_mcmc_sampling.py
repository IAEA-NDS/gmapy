import sys
sys.path.append('../../..')
import time
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
from gmapy.tf_uq.inference import generate_MCMC_chain
# These imports are necessary to satisfy
# dependencies of objects loaded by dill above
import numpy as np

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

s1 = time.time()
chain, tracing_info = generate_MCMC_chain(
    optvals, post.log_prob, post.neg_log_prob_hessian,
    nugget=1e-8, step_size=0.01, num_burnin_steps=int(5e3),
    num_results=int(50e3), num_leapfrog_steps=5
)
s2 = time.time()
print(f's2-s1: {s2-s1}')

save_objects(
    'output/03_mcmc_sampling_output.pkl', locals(),
    'chain', 'tracing_info'
)
