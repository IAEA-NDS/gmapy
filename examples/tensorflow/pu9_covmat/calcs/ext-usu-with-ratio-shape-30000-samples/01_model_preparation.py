import sys
sys.path.append('../../../../..')
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
    remove_dummy_datasets
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

tfd = tfp.distributions
tfb = tfp.bijectors

# retrieve prior estimates and covariances from the database
db_path = '../../../../../tests/testdata/data_and_sacs.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

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

# relevant USU error contributions
# abs U5(n,f) at 1, 5, 15 MeV (clear USU around 2 MeV region)
# abs PU9(n,f) at 1, 5 MeV (likely no USU but to be conservative)
# shape U5(n,f) at 0, 1, 5, 15 MeV (likely USU in the low energy range (not thermal), at about 2 MeV nd at 15 MeV) 
# shape PU9(n,f) at 1, 5 MeV (likely USU at about 2 MeV)
# MT:3-R1:10-R2:8 at 0, 1, 5, 15, 30, 100, 200
# MT:3-R1:9-R2:8 at 0, 1, 5, 15, 30, 60
# MT:4-R1:10-R2:8 at 1, 5 MeV (likely USU in 1-5 MeV range)
# MT:4-R1:9-R2:8 at 0, 1, 5, 15 (likely USU at 

# reaction-specific USU components
usu_dfs = []
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:1-R1:8',), (1., 5., 15.), (1e-2,)*3))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:1-R1:9',), (1., 5., 15.), (1e-2,)*3))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:2-R1:8',), (0., 1., 5., 15.), (1e-2,)*4))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:2-R1:9',), (1., 5.), (1e-2,)*2))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:3-R1:10-R2:8',), (0., 1., 5., 15., 30., 100., 200.), (1e-2,)*7))
usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:3-R1:9-R2:8',), (0., 1., 5., 15., 30., 60.), (1e-2,)*6))
# usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:4-R1:10-R2:8',), (1., 5.), (1e-2,)*2))
# usu_dfs.append(create_endep_abs_usu_df(exptable, ('MT:4-R1:9-R2:8',), (0., 1., 5., 15.), (1e-2,)*4))
reac_usu_df = pd.concat(usu_dfs, ignore_index=True)
reac_usu_df['MAPCOL'] = (
    'reac_usu_' + reac_usu_df['REAC'] + '/' + reac_usu_df['ENERGY'].map('{:1.2e}'.format)
)
exptable['MAPCOL'] = (
    'reac_usu_' + exptable['REAC'] + '/' + exptable['ENERGY'].map('{:1.2e}'.format)
)
# energy dependent USU component for all shape ratio components
mt4_reacs = exptable[exptable.REAC.str.match('MT:4-.*R.:(8|9|10)')].REAC.unique()
mt4_usu_df = create_endep_abs_usu_df(
    exptable, mt4_reacs, (1.5e-4, 1e-3, 1e-2, 1e-1, 1, 5, 15, 30), (1e-2,)*8
)
mt4_usu_df['MAPCOL'] = 'mt4_usu/' + mt4_usu_df['ENERGY'].map('{:1.2e}'.format)
mt4_sel = exptable.REAC.isin(mt4_reacs)
exptable.loc[mt4_sel, 'MAPCOL'] = (
    'mt4_usu/' + exptable.loc[mt4_sel, 'ENERGY'].map('{:1.2e}'.format)
)

orig_usu_df = pd.concat([reac_usu_df, mt4_usu_df], ignore_index=True)
orig_usu_map = EnergyDependentAbsoluteUSUMap((orig_usu_df, exptable), reduce=True)
orig_usu_jac = tf.sparse.to_dense(orig_usu_map.jacobian(orig_usu_df.PRIOR.to_numpy()))

# energy dependent USU components for all (absolute) ratio components
mt3_reacs = exptable[exptable.REAC.str.match('MT:3-.*R.:(8|9|10)')].REAC.unique()
mt3_usu_df = create_endep_abs_usu_df(
    exptable, mt3_reacs, (1.5e-4, 1e-3, 1e-2, 1e-1, 1, 5, 15, 30), (1e-2,)*8
)
mt3_usu_df['MAPCOL'] = 'mt3_usu/' + mt3_usu_df['ENERGY'].map('{:1.2e}'.format)
mt3_sel = exptable.REAC.isin(mt3_reacs)
exptable.loc[mt3_sel, 'MAPCOL'] = (
    'mt3_usu/' + exptable.loc[mt3_sel, 'ENERGY'].map('{:1.2e}'.format)
)
mt3_usu_map = EnergyDependentAbsoluteUSUMap((mt3_usu_df, exptable), reduce=True)
mt3_usu_jac = tf.sparse.to_dense(mt3_usu_map.jacobian(mt3_usu_df.PRIOR.to_numpy()))

usu_jac = tf.concat((orig_usu_jac, mt3_usu_jac), axis=1)
usu_df = pd.concat([orig_usu_df, mt3_usu_df], ignore_index=True) 


def create_like_cov_fun(usu_df, expcov_linop, Smat, mapcol):

    def map_uncertainties(u):
        ids = np.zeros((len(usu_df),), dtype=np.int32)
        for index, row in red_usu_df.iterrows(): 
            curcat = row[mapcol]
            cur_idcs = usu_df.index[usu_df[mapcol] == curcat].to_numpy()
            ids[cur_idcs] = index
        # scatter the uncertainties to the appropriate places
        tf_ids = tf.constant(ids, dtype=tf.int32)
        uncs = tf.nn.embedding_lookup(u, tf_ids)
        return uncs

    def like_cov_fun(u):
        uncs = map_uncertainties(u)
        # covop = tf.linalg.LinearOperatorLowRankUpdate(
        covop = tf.linalg.LinearOperatorLowRankUpdate(
            expcov_linop, Smat, tf.square(uncs) + 1e-7,
            is_self_adjoint=True, is_positive_definite=True,
            is_diag_update_positive=True
        )
        return covop
    red_usu_df = usu_df[['MAPCOL', 'ENERGY']].drop_duplicates()
    red_usu_df['TYPE'] = usu_df['MAPCOL'].str.extract('(.*)/.*')
    red_usu_df.sort_values(
        ['TYPE', 'ENERGY'], ascending=True, ignore_index=True, inplace=True
    )
    return like_cov_fun, red_usu_df.copy()


like_cov_fun, red_usu_df = create_like_cov_fun(usu_df, expcov_linop, usu_jac, 'MAPCOL')
num_covpars = len(red_usu_df)

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
             'post', 'likelihood', 'priorvals', 'is_adj', 'usu_df', 'red_usu_df',
             'num_covpars', 'priortable', 'exptable', 'expcov', 'like_cov_fun', 'compmap', 'restrimap')
