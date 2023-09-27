import sys
sys.path.append('../../..')
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, likelihood, priorvals, is_adj = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian

refvals = priorvals[is_adj]
optres = determine_MAP_estimate(
    refvals, neg_log_prob_and_gradient, neg_log_post_hessian,
    max_inner_iters=500, max_outer_iters=50, nugget=1e-4,
    ret_optres=True, must_converge=True
)

# save the optimized parameters
params = refvals

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'params')
