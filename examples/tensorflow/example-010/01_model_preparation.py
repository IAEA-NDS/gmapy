import sys
sys.path.append('../../..')
import pandas as pd
from scipy.sparse import block_diag
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.mappings.tf.compound_map_tf import CompoundMap
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table,
)
from gmapy.mappings.priortools import (
    attach_shape_prior,
    initialize_shape_prior,
    remove_dummy_datasets
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)
from gmapy.tf_uq.auxiliary import (
    make_positive_definite
)
import matplotlib.pyplot as plt
import plotly.express as px


tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = '../../../tests/testdata/data_and_sacs.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# full experimental table
is_rel_cov = True
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'], relative=is_rel_cov).toarray()
expcov_list, _ = create_datablock_covmat_list(db['datablock_list'], relative=is_rel_cov)
expchol_list = [np.linalg.cholesky(m.toarray()) for m in expcov_list] 
expchol_linop_list = [tf.linalg.LinearOperatorLowerTriangular(m) for m in expchol_list]
expchol_linop = tf.linalg.LinearOperatorBlockDiag(expchol_linop_list, is_non_singular=True, is_positive_definite=True)
if is_rel_cov:
    exptable['UNC'] = np.sqrt(expcov.diagonal()) * exptable['DATA']
else:
    exptable['UNC'] = np.sqrt(expcov.diagonal())


##################################################
# u5 case
##################################################
u5_prior_selection = (priortable.REAC.str.match('MT:1-R1:8') & 
                (priortable.ENERGY >= 0.15) & (priortable.ENERGY <= 200))
u5_priortable = priortable[u5_prior_selection].reset_index(drop=True)
u5_exp_selection = (exptable.REAC.str.match('MT:1-R1:8|MT:2-R1:8') &
                    (exptable.ENERGY >= 0.15) & (exptable.ENERGY <= 200))
u5_exptable = exptable[u5_exp_selection].reset_index(drop=True)
u5_priortable = attach_shape_prior((u5_priortable, u5_exptable))
u5_compmap = CompoundMap((u5_priortable, u5_exptable), reduce=True) 
u5_priortable['PRIOR'] = u5_optres.position  # depends on the optimization already performed once
u5_expcov = expcov[np.ix_(u5_exptable.index, u5_exptable.index)]
u5_exptable.reset_index(drop=True, inplace=True)
u5_propfun = tf.function(u5_compmap.propagate)
u5_jacfun = tf.function(u5_compmap.jacobian)

u5_expcov_linalg = tf.linalg.LinearOperatorLowerTriangular(np.linalg.cholesky(u5_expcov))
u5_likelihood = MultivariateNormalLikelihood(
    len(u5_priortable), u5_propfun, u5_jacfun,
    u5_exptable.DATA.to_numpy(), u5_expcov_linalg,
    approximate_hessian=True, relative=is_rel_cov
)
u5_neg_log_prob_and_gradient = tf.function(u5_likelihood.neg_log_prob_and_gradient)
u5_hess_start = u5_likelihood.log_prob_hessian(u5_priortable.PRIOR.to_numpy())
u5_inv_hess_start = make_positive_definite(np.linalg.inv(u5_hess_start), 1e-5)
u5_optres = tfp.optimizer.bfgs_minimize(
        u5_neg_log_prob_and_gradient,
        initial_position=u5_priortable.PRIOR.to_numpy(copy=True),
        initial_inverse_hessian_estimate=u5_inv_hess_start,
        max_iterations=5000
)
u5_priortable['EVAL'] = u5_optres.position.numpy()
propcss = u5_compmap(u5_optres.position)

# rescale the shape data to match the absolute data
def rescale_shape_data(group, priortable):
    curnode = group.name
    norm_node = curnode.replace('exp_', 'norm_')
    normfact = priortable.loc[priortable.NODE==norm_node, 'EVAL'].to_numpy()
    return group / normfact if len(normfact) == 1 else group

grouped = u5_exptable.groupby('NODE')
u5_exptable_new = u5_exptable.copy()
u5_exptable['DATA'] = grouped['DATA'].transform(rescale_shape_data, u5_priortable)

# perform the optimization again to check all normalization factors are one 
u5_priortable['PRIOR'] = u5_optres.position  # depends on the optimization already performed once
u5_likelihood = MultivariateNormalLikelihood(
    len(u5_priortable), u5_propfun, u5_jacfun,
    u5_exptable.DATA.to_numpy(), u5_expcov_linalg,
    approximate_hessian=True, relative=is_rel_cov
)
u5_neg_log_prob_and_gradient = tf.function(u5_likelihood.neg_log_prob_and_gradient)
u5_hess_start = u5_likelihood.log_prob_hessian(u5_priortable.PRIOR.to_numpy())
u5_inv_hess_start = make_positive_definite(np.linalg.inv(u5_hess_start), 1e-5)
u5_optres = tfp.optimizer.bfgs_minimize(
        u5_neg_log_prob_and_gradient,
        initial_position=u5_priortable.PRIOR.to_numpy(copy=True),
        initial_inverse_hessian_estimate=u5_inv_hess_start,
        max_iterations=5000
)
u5_priortable['EVAL'] = u5_optres.position.numpy()

# do the evaluation with absolute covariance matrix 
# and conversion of shape to absolute data
u5_propcss = u5_compmap(u5_optres.position).numpy()
if is_rel_cov:
    u5_absexpcov = u5_expcov * u5_propcss.reshape((-1, 1)) * u5_propcss.reshape((1, -1))
else:
    u5_absexpocv = u5_expcov


u5_absexpcov_chol_linalg = tf.linalg.LinearOperatorLowerTriangular(
    np.linalg.cholesky(u5_absexpcov)
)

u5_absexptable = u5_exptable.copy()
u5_absexptable['REAC'] = u5_absexptable['REAC'].str.replace('MT:2-', 'MT:1-')
u5_abspriortable = u5_priortable[~u5_priortable.NODE.str.match('norm_')].copy()
u5_abscompmap = CompoundMap((u5_abspriortable, u5_absexptable), reduce=True)
u5_abspropfun = tf.function(u5_abscompmap.propagate)
u5_absjacfun = tf.function(u5_abscompmap.jacobian)

u5_abslikelihood = MultivariateNormalLikelihood(
    len(u5_abspriortable), u5_abspropfun, u5_absjacfun,
    u5_absexptable.DATA.to_numpy(), u5_absexpcov_chol_linalg,
    approximate_hessian=True, relative=False
)
# do the optimization
u5_abs_neg_log_prob_and_gradient = tf.function(u5_abslikelihood.neg_log_prob_and_gradient)
u5_abs_hess_start = u5_abslikelihood.log_prob_hessian(u5_abspriortable.PRIOR.to_numpy())
u5_abs_inv_hess_start = make_positive_definite(np.linalg.inv(u5_abs_hess_start), 1e-5)
u5_abs_optres = tfp.optimizer.bfgs_minimize(
        u5_abs_neg_log_prob_and_gradient,
        initial_position=u5_abspriortable.PRIOR.to_numpy(copy=True),
        initial_inverse_hessian_estimate=u5_abs_inv_hess_start,
        max_iterations=5000
)
u5_abspriortable['EVAL'] = u5_abs_optres.position.numpy()

# save all required quantities
for cur_reac, group in u5_absexptable.groupby('NODE'):
    plt.errorbar(group.ENERGY, group.DATA, yerr=group.UNC, linestyle='')
    plt.scatter(group.ENERGY, group.DATA, linestyle='', marker='o')

tmpsel = u5_abspriortable.NODE.str.match('xsid_8')
plt.plot(u5_abspriortable.ENERGY[tmpsel], u5_abspriortable.EVAL[tmpsel], color='blue')
tmpsel = u5_priortable.NODE.str.match('xsid_8')
plt.plot(u5_abspriortable.ENERGY[tmpsel], u5_priortable.EVAL[tmpsel], color='green')
plt.axhline(y=2.4)
plt.xlim(10, 100)
plt.show()

eval_u5_abscov = -np.linalg.inv(u5_abslikelihood.log_prob_hessian(u5_abspriortable['EVAL'].to_numpy()).numpy())
np.sqrt(eval_u5_abscov.diagonal())


u5_abspriortable.to_csv('u5_eval_dataframe.csv', sep='\t')
u5_absexptable.to_csv('u5_exp_dataframe.csv', sep='\t')
np.savetxt('u5_eval_covmat.csv', eval_u5_abscov, delimiter='\t')
np.savetxt('u5_absexpcov.csv', u5_absexpcov, delimiter='\t')

eval_u5_abscov_restored = np.loadtxt('u5_eval_covmat.csv', delimiter='\t')
u5_absexpcov = np.loadtxt('u5_absexpcov.csv', delimiter='\t')
u5_abspriortable_restored = pd.read_csv('u5_eval_dataframe.csv', delimiter='\t')
u5_absexpcov_restored = np.loadtxt('u5_absexpcov.csv', delimiter='\t')

np.allclose(eval_u5_abscov_restored, eval_u5_abscov)
np.allclose(u5_absexpcov_restored, u5_absexpcov)
