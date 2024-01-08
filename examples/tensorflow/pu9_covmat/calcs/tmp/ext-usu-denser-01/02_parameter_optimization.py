import sys
sys.path.append('../../../../../..')
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.inference import determine_MAP_estimate
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, likelihood, priorvals, is_adj, usu_df, red_usu_df, num_covpars = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj',
                 'usu_df', 'red_usu_df', 'num_covpars')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)
neg_log_post_hessian = post.neg_log_prob_hessian

covpars = np.full(num_covpars, 1.)
refvals = likelihood.combine_pars(priorvals[is_adj], covpars)

optres = determine_MAP_estimate(
    refvals, neg_log_prob_and_gradient,
    neg_log_post_hessian, max_inner_iters=500, max_outer_iters=50, nugget=1e-3,
    ret_optres=True, must_converge=True
)

# save the optimized parameters
refvals = optres.position
red_usu_df['USU'] = refvals.numpy()[-num_covpars:]
params, covpars = likelihood.split_pars(refvals)

opt_neg_hessian = neg_log_post_hessian(optres.position)
_, opt_neg_jac = neg_log_prob_and_gradient(optres.position)

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'params', 'covpars', 'usu_df', 'red_usu_df',
             'opt_neg_hessian', 'opt_neg_jac')
