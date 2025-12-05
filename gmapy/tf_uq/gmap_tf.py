from typing import Optional
import pandas as pd
import numpy as np
import tensorflow as tf
from copy import deepcopy
from ..mappings.tf.restricted_map import (
    RestrictedMap,
)
from ..mappings.tf.compound_map_tf import (
    CompoundMap as CompoundMapTF,
)
from ..mappings.priortools import (
    remove_dummy_datasets,
    attach_shape_prior,
    initialize_shape_prior,
)

from ..data_management.tablefuns import create_experiment_table
from ..data_management.uncfuns import create_experimental_covmat
from ..data_management.tablefuns import create_prior_table
from ..data_management.uncfuns import create_prior_covmat

from .custom_distributions import (
    MultivariateNormalLikelihoodWithCovParams,
    ChiSquarePseudoDistWithCovParams,
)
from .inference import (
    iterative_gls_estimate,
    determine_MAP_estimate,
)


def create_cov_linop_fun(expcov_list):
    """Create a linear covariance operator function.

    Create a function that returns a TensorFlow LinearOperator
    representing a covariance matrix. The generated function
    expects formally a 1d input tensor as argument for 
    compatibility with the gmapy 
    `MultivariateNormalLikelihoodWithCovParams` class but does
    not use it. 
    """
    def cov_linop_fun(x):
        expchol_list = [tf.linalg.cholesky(x) for x in expcov_list]
        expchol_op_list = [tf.linalg.LinearOperatorLowerTriangular(
                x, is_non_singular=True, is_square=True
            ) for x in expchol_list]
        expcov_chol = tf.linalg.LinearOperatorBlockDiag(
            expchol_op_list, is_non_singular=True, is_square=True)
        expcov_linop = tf.linalg.LinearOperatorComposition(
            [expcov_chol, expcov_chol.adjoint()],
            is_self_adjoint=True, is_positive_definite=True
        )
        return expcov_linop
    return cov_linop_fun


def update_expvals(
    exptable: pd.DataFrame, expvals: Optional[tf.Variable]=None
) -> tf.Variable:
    """Create or update the experimental values."""
    if expvals is None:
        return tf.Variable(exptable.DATA.to_numpy(), dtype=tf.float64)
    else:
        expvals.assign(exptable.DATA.to_numpy())
        return expvals


def update_expcov_list(
    exptable: pd.DataFrame, expcov: np.array,
    expcov_list: Optional[list[tf.Variable]]=None
) -> list[tf.Variable]:
    """Create or update list of covariance matrix blocks."""
    block_lens = exptable['DB_IDX'].value_counts().sort_index().to_numpy()
    block_stops = np.cumsum(block_lens)
    block_starts = np.concatenate([[0], block_stops[:-1]], axis=0)

    must_create = False
    if expcov_list is None:
        must_create = True
        expcov_list = []

    for i, (sta, sto) in enumerate(zip(block_starts, block_stops)):
        curmat = tf.constant(expcov[sta:sto, sta:sto], dtype=tf.float64)
        if must_create:
            expcov_list.append(tf.Variable(curmat))
        else:
            expcov_list[i].assign(curmat)

    return expcov_list


def prepare_stat_model(
    prior: list, datablocks: list, remove_dummy: bool=True, mt6_ppp=False,
    relative=True, optim_type='iterative-gls', optim_opts=None,
) -> dict:
    """Prepare all quantities for statistical inference."""
    if optim_opts is None:
        optim_opts = {}

    prior = deepcopy(prior)
    datablocks = deepcopy(datablocks)

    if remove_dummy:
        remove_dummy_datasets(datablocks)

    priortable = create_prior_table(prior)
    priorcov = create_prior_covmat(prior)

    exptable = create_experiment_table(datablocks)
    expcov = create_experimental_covmat(datablocks, relative=relative).toarray()
    exptable['UNC'] = np.sqrt(expcov.diagonal())

    expvals = update_expvals(exptable, None)
    expcov_list = update_expcov_list(exptable, expcov, None)

    # build the covariance linear operator fun
    cov_linop_fun = create_cov_linop_fun(expcov_list)

    # construct priortable and mapping to experimental data
    priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
    compmap = CompoundMapTF((priortable, exptable), reduce=True)

    # some convenient shortcuts
    is_adj = priorcov.diagonal() != 0.
    priorvals = priortable.PRIOR.to_numpy()
    refvals = priorvals.copy()
    reluncs = exptable['UNC'].to_numpy()

    initialize_shape_prior((priortable, exptable), compmap, refvals=refvals, uncs=reluncs)

    # define model propagation and jacobian function
    adj_idcs = np.where(is_adj)[0]
    fixed_idcs = np.where(~is_adj)[0]
    restrimap = RestrictedMap(
        len(priorvals), compmap.propagate, compmap.jacobian,
        fixed_params=priorvals[fixed_idcs], fixed_params_idcs=fixed_idcs
    )

    # determine the MT6 (SACS) experiment indices
    mt6_idcs=None
    if not mt6_ppp:
        mt6_idcs = np.where(exptable.REAC.str.startswith('MT:6-'))[0]

    # determine indices of normalization factors
    norm_sel = priortable.loc[is_adj, 'NODE'].str.match('norm_').to_numpy()
    norm_idcs = np.where(norm_sel)[0]
    norm_values = np.ones(len(norm_idcs), dtype=int)

    distfun = (
        ChiSquarePseudoDistWithCovParams if optim_type.lower() == 'chisquare'
        else MultivariateNormalLikelihoodWithCovParams  # for MLE and iterative GLS
    )

    propfun = tf.function(restrimap.propagate)
    jacfun = tf.function(restrimap.jacobian)

    likelihood = distfun(
        len(adj_idcs), 0, propfun, jacfun,
        expvals, cov_linop_fun, approximate_hessian=True, relative=relative,
        no_ppp_idcs=mt6_idcs, cov_freeze_param_idcs=norm_idcs, cov_freeze_param_values=norm_values
    )

    if optim_type.lower() == 'iterative-gls':
        optim_func = lambda x: iterative_gls_estimate(
            x,
            likelihood.get_model_prediction,
            likelihood.get_model_jacobian,
            expvals,
            likelihood.get_covariance_linop,
            ret_optres=False, **optim_opts
        )
    elif optim_type.lower() in ('mle', 'chisquare'):
        neg_log_prob_and_gradient = tf.function(likelihood.neg_log_prob_and_gradient)
        neg_log_prob_hessian = likelihood.log_prob_hessian 

        optim_func = lambda x: determine_MAP_estimate(
            x, neg_log_prob_and_gradient, neg_log_prob_hessian, ret_optres=False, **optim_opts
        )
    else:
        raise ValueError('Unknown optimization type')

    return {
        'optim_func': optim_func,
        'priortable': priortable,
        'exptable': exptable,
        'expcov_rel': expcov,
        'likelihood': likelihood,
        'is_adj': is_adj,
        'expvals': expvals,
        'expcov_list': expcov_list,
    }


def evaluate_gma_database(
    prior: list, datablocks: list, remove_dummy: bool=True, mt6_ppp=False,
    relative=True, optim_type='iterative-gls', optim_opts=None,
) -> dict:

    model_info = prepare_stat_model(
        prior, datablocks, remove_dummy, mt6_ppp,
        relative, optim_type, optim_opts
    )

    is_adj = model_info['is_adj']
    priorvals = model_info['priortable']['PRIOR']
    startvals = tf.constant(priorvals[is_adj], dtype=tf.float64)
    optim_func = model_info['optim_func']

    optres = optim_func(startvals)
    result_df = model_info['priortable'].copy()
    result_df.loc[is_adj, 'POST'] = optres.numpy()
    result_df.loc[~is_adj, 'POST'] = result_df.loc[~is_adj, 'PRIOR']
    return {
        'table': result_df,
        'likelihood': model_info['likelihood'],
        'exptable': model_info['exptable'],
        'expcov_rel': model_info['expcov_rel'],
        'expvals': model_info['expvals'],
        'expcov_list': model_info['expcov_list'],
    }

