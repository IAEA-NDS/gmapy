import sys
sys.path.append('../../..')
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.tf_uq.auxiliary import (
    make_positive_definite, invert_symmetric_matrix
)
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

post, likelihood, priorvals, is_adj, usu_df, red_usu_df, num_covpars = \
    load_objects('output/01_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj',
                 'usu_df', 'red_usu_df', 'num_covpars')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)

covpars = np.full(num_covpars, 1.)
refvals = likelihood.combine_pars(priorvals[is_adj], covpars)
max_outer_iters = 20
# max_inner_iters = np.round(np.linspace(50, 500, max_outer_iters)).astype(np.int32)
max_inner_iters = np.full(max_outer_iters, 500, dtype=np.int32)
outer_iter = 0
converged = False
while not converged and outer_iter < max_outer_iters:
    max_inner_iter = tf.constant(max_inner_iters[outer_iter], dtype=tf.int32)
    outer_iter += 1
    print(f'#  outer iteration {outer_iter}')
    print(f'-- running inner iteration with {max_inner_iter} iterations')
    # obtain an approximation of the posterior covariance matrix to aid optimization
    neg_log_post_hessian = post.neg_log_prob_hessian(refvals)
    fixed_neg_log_post_hessian = make_positive_definite(neg_log_post_hessian, 1e-4)
    inv_neg_log_post_hessian = invert_symmetric_matrix(fixed_neg_log_post_hessian)
    fixed_inv_neg_log_post_hessian = make_positive_definite(inv_neg_log_post_hessian, 1e-4)
    # find peak of posterior distribution (to use it as a starting value of MCMC)
    optres = tfp.optimizer.bfgs_minimize(
        neg_log_prob_and_gradient, initial_position=refvals,
        initial_inverse_hessian_estimate=fixed_inv_neg_log_post_hessian,
        max_iterations=max_inner_iter
    )
    converged = optres.converged.numpy()
    refvals = optres.position
    print(optres)
    print('USU parameters')
    red_usu_df['USU'] = optres.position.numpy()[-num_covpars:]
    print(red_usu_df)


# save the optimized parameters
params, covpars = likelihood.split_pars(refvals)

save_objects('output/02_parameter_optimization_output.pkl', locals(),
             'optres', 'params', 'covpars', 'usu_df', 'red_usu_df')
