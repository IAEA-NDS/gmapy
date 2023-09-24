import sys
sys.path.append('../../..')
import pandas as pd
from scipy.sparse import block_diag
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
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.tf.energy_dependent_absolute_usu_map_tf import (
    EnergyDependentAbsoluteUSUMap,
    create_endep_abs_usu_df
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihoodWithCovParams,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)
from gmapy.tf_uq.custom_linear_operators import MyLinearOperatorLowRankUpdate

tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = 'inpdata/data_test1r1_USU.gma'
db = read_gma_database(db_path)

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])
# add a missing prior point (because expdata goes up to 2.2 MeV)
new_row = priortable.iloc[0:1].copy()
new_row.ENERGY = 2.2
priortable = pd.concat([priortable, new_row], ignore_index=True, axis=0)
priorcov = block_diag([priorcov, [[np.inf]]])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

# For testing: perturb one dataset and see if we get a change
# exptable.loc[exptable.NODE == 'exp_1025', 'DATA'] *= 1.5

# initialize the normalization errors
priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

# some convenient shortcuts
priorvals = priortable.PRIOR.to_numpy()
expvals = exptable.DATA.to_numpy()

# speed up the pdf log_prob calculations exploiting the block diagonal structure
expcov_list, idcs_tuples = create_datablock_covmat_list(db['datablock_list'], relative=True)
expchol_list = [tf.linalg.cholesky(x.toarray()) for x in expcov_list]
expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
        x, is_non_singular=True, is_square=True
    ) for x in expchol_list]

# generate a restricted mapping blending out the fixed parameters
is_adj = priorcov.diagonal() != 0.
adj_idcs = np.where(is_adj)[0]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
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
# generate the USU mapping
usu_df = create_endep_abs_usu_df(
        exptable, ('MT:1-R1:1',), priortable.ENERGY.to_numpy()[:-1], np.arange(len(priortable)-1, dtype=np.float64)
)
# usu_df = create_endep_abs_usu_df(
#     exptable, ('MT:1-R1:1',), np.array((1., 2.)), np.array((3., 4.))
# )
usu_map = EnergyDependentAbsoluteUSUMap((usu_df, exptable), reduce=True)
usu_jac = tf.sparse.to_dense(usu_map.jacobian(usu_df.PRIOR.to_numpy()))


def create_like_cov_fun(usu_df, expcov_linop, Smat):
    def like_cov_fun(u):
        # define the ids for tf.nn.embedding_lookup
        ids = np.zeros((len(usu_df),), dtype=np.int32)
        for i, cur_en in enumerate(uniq_energies):
            cur_idcs = usu_df.index[usu_df.ENERGY == cur_en].to_numpy()
            ids[cur_idcs] = i
        # scatter the uncertainties to the appropriate places
        tf_ids = tf.constant(ids, dtype=tf.int32)
        uncs = tf.nn.embedding_lookup(u, tf_ids)
        update_operator = tf.linalg.LinearOperatorDiag(
            tf.square(uncs) + 1e-8, is_positive_definite=True, is_square=True
        )
        # covop = tf.linalg.LinearOperatorLowRankUpdate(
        covop = MyLinearOperatorLowRankUpdate(
            expcov_linop, Smat, update_operator=update_operator,
            is_self_adjoint=True, is_positive_definite=True,
            is_update_positive_definite=True
        )
        return covop
    uniq_energies = np.sort(pd.unique(usu_df.ENERGY))
    return like_cov_fun


num_covpars = len(np.unique(usu_df.ENERGY))
like_cov_fun = create_like_cov_fun(usu_df, expcov_linop, usu_jac)

# generate the prior distribution
is_adj_constr = is_adj & np.isfinite(priorcov.diagonal())
is_adj_constr_idcs = np.where(is_adj_constr)[0]
priorcov_chol = tf.linalg.LinearOperatorDiag(np.sqrt(priorcov.diagonal()[is_adj_constr]))
prior_red = MultivariateNormal(priorvals[is_adj_constr], priorcov_chol)
prior_red.log_prob_hessian(priorvals[is_adj_constr])
prior = DistributionForParameterSubset(
    prior_red, len(adj_idcs) + num_covpars, is_adj_constr_idcs
)

# generate the likelihood
likelihood = MultivariateNormalLikelihoodWithCovParams(
    len(adj_idcs), num_covpars, propfun, jacfun, expvals, like_cov_fun,
    approximate_hessian=True, relative=True
)

# combine prior and likelihood into posterior
post = UnnormalizedDistributionProduct([prior, likelihood])

save_objects('output/01_model_preparation_output.pkl', locals(),
             'post', 'likelihood', 'priorvals', 'is_adj', 'usu_df',
             'num_covpars', 'priortable', 'exptable', 'expcov', 'like_cov_fun', 'compmap')
