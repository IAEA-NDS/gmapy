import sys
sys.path.append('../../..')
import pandas as pd
from scipy.sparse import block_diag, csr_matrix, diags
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    save_objects
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
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
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.tf.energy_dependent_absolute_usu_map_tf import (
    EnergyDependentAbsoluteUSUMap,
    create_endep_abs_usu_df
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)

tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = '../../../tests/testdata/data_and_sacs.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
# priorcov = create_prior_covmat(db['prior_list'])
priorcov = diags(priortable.PRIOR * 0.2)

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

# remove all non-linear quantities
exptable = exptable.loc[exptable.REAC.str.match('MT:(1|5)-')]
exptable[exptable.REAC.str.match('MT:5-')]
keep_idcs = np.array(exptable.index)
exptable.reset_index(inplace=True, drop=True)
# subset covariance matrix
expcov = expcov.toarray()[np.ix_(keep_idcs, keep_idcs)]

compmap = CompoundMapTF((priortable, exptable), reduce=True)

# some convenient shortcuts
priorvals = priortable.PRIOR.to_numpy()
expvals = exptable.DATA.to_numpy()

# speed up the pdf log_prob calculations exploiting the block diagonal structure
db_idcs = np.sort(exptable.DB_IDX.unique())
expcov_list = []
for db_idx in db_idcs:
    curidcs = np.array(exptable[exptable.DB_IDX == db_idx].index)
    expcov_list.append(expcov[np.ix_(curidcs, curidcs)])

expchol_list = [tf.linalg.cholesky(x) for x in expcov_list]
expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
        x, is_non_singular=True, is_square=True
    ) for x in expchol_list]

# generate a restricted mapping blending out the fixed parameters
is_adj = priorcov.diagonal() != 0.
adj_idcs = np.where(is_adj)[0]
fixed_pars = priorvals[~is_adj]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=fixed_pars, fixed_params_idcs=fixed_idcs
)
propfun = tf.function(restrimap.propagate)
jacfun = tf.function(restrimap.jacobian)

# generate the experimental covariance matrix
expcov_chol = tf.linalg.LinearOperatorBlockDiag(
    expchol_op_list, is_non_singular=True, is_square=True)
expcov_linop = tf.linalg.LinearOperatorComposition(
    [expcov_chol, expcov_chol.adjoint()],
    is_self_adjoint=True, is_positive_definite=True
)

# generate the prior distribution
is_adj_constr = is_adj & np.isfinite(priorcov.diagonal())
is_adj_constr_idcs = np.where(is_adj_constr)[0]
priorcov_chol = tf.linalg.LinearOperatorDiag(np.sqrt(priorcov.diagonal()[is_adj_constr]))
prior_red = MultivariateNormal(priorvals[is_adj_constr], priorcov_chol)
prior_red.log_prob_hessian(priorvals[is_adj_constr])
prior = DistributionForParameterSubset(
    prior_red, len(adj_idcs), is_adj_constr_idcs
)

# generate the likelihood
likelihood = MultivariateNormalLikelihood(
    len(adj_idcs), propfun, jacfun, expvals, expcov_chol,
    approximate_hessian=True, relative=False
)

# combine prior and likelihood into posterior
post = UnnormalizedDistributionProduct([prior, likelihood])


save_objects('output/01_model_preparation_output.pkl', locals(),
             'post', 'likelihood', 'priorvals', 'is_adj',
             'priortable', 'exptable', 'expcov', 'expcov_linop')
