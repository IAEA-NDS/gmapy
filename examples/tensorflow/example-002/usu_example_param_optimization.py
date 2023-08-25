import numpy as np
from pipeline_utils import (
    load_objects, save_objects,
    make_positive_definite, invert_symmetric_matrix
)
import tensorflow as tf
import tensorflow_probability as tfp

post, likelihood, priorvals, is_adj, usu_df, num_covpars = \
    load_objects('tmp/usu_example_model_preparation_output.pkl',
                 'post', 'likelihood', 'priorvals', 'is_adj', 'usu_df',
                 'num_covpars')

# speed it up!
neg_log_prob_and_gradient = tf.function(post.neg_log_prob_and_gradient)


# obtain an approximation of the posterior covariance matrix to aid optimization
covpars = np.full(num_covpars, 1.)
refvals = likelihood.combine_pars(priorvals[is_adj], covpars)
outer_iter = 0
converged = False
while not converged and outer_iter < 50:
    outer_iter += 1
    print(f'outer iteration {outer_iter}')
    neg_log_post_hessian = post.neg_log_prob_hessian(refvals)
    fixed_neg_log_post_hessian = make_positive_definite(neg_log_post_hessian, 1e-5)
    inv_neg_log_post_hessian = invert_symmetric_matrix(fixed_neg_log_post_hessian)
    fixed_inv_neg_log_post_hessian = make_positive_definite(inv_neg_log_post_hessian, 1e-5)
    # find peak of posterior distribution (to use it as a starting value of MCMC)
    optres = tfp.optimizer.bfgs_minimize(
        neg_log_prob_and_gradient, initial_position=refvals,
        initial_inverse_hessian_estimate=fixed_inv_neg_log_post_hessian, max_iterations=500)
    converged = optres.converged.numpy()
    refvals = optres.position
    print(optres)


# save the optimized parameters
params, covpars = likelihood.split_pars(refvals)
usu_df.UNC = refvals[-len(usu_df):]

save_objects('tmp/usu_example_param_optimization_output.pkl', locals(),
             'optres', 'params', 'covpars', 'usu_df')
