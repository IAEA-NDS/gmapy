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
from ..mappings.priortools import remove_dummy_datasets
from ..data_management.tablefuns import create_experiment_table
from ..data_management.uncfuns import create_experimental_covmat
from .gmap_tf import (
    prepare_stat_model,
    update_expvals,
    update_expcov_list,
)


def update_unreduced_gmadb_prior(gmadb, priortable, is_adj, newvals):
    """Update the values in the prior used for reduction.

    Copy the values from the `PRIOR` column in `priortable`
    to the appropriate location in the `gmadb` dict.
    Importantly, `gmadb` contains the data *before reduction*.

    Args:
        gmadb (dict): Data structure with the prior and the
            unreduced experimental data.
        priortable (pd.DataFrame): Dataframe with the prior
            information and additionally a column `PRIOR`
            with updated prior values.

    Returns:
        None: Modifications of gmadb are performed inplace.
    """
    priortable = priortable.copy()
    priortable.loc[is_adj, 'POST'] = newvals
    xsprior_df = priortable[priortable.NODE.str.startswith('xsid_')]
    for xsidstr, group_df in xsprior_df.groupby('NODE'): 
        prior_id = int(xsidstr.split('_')[1]) - 1
        postxs = group_df['POST'].to_list()
        curprior = gmadb['prior'][prior_id] 
        # The original prior for reduction has
        # two limit points more
        assert len(curprior['EN'])-2 == len(postxs) 
        assert curprior['EN'][1:-1] == group_df['ENERGY'].to_list() 
        curprior['CS'][1:-1] = postxs
        curprior['CS'][0] = postxs[0]
        curprior['CS'][-1] = postxs[-1]


def reduce_database_iteratively(
    orig_gmadb, max_iters=10, rel_tol=1e-6,
    remove_dummy=True, mt6_ppp=False,
    optim_type='iterative-gls', optim_opts=None
):
    """Reduce experimental data."""
    orig_gmadb = deepcopy(orig_gmadb)
    if remove_dummy:
        remove_dummy_datasets(orig_gmadb['datablocks'])

    try:
        from datpy.datpy import reduce_database
    except ImportError:
        raise ImportError(
            'Unable to import the `reduce_database` function '
            'from the `datpy.datpy` module. Is the `datpy` package '
            'installed?'
        )

    new_gmadb = reduce_database(orig_gmadb)

    model_info = prepare_stat_model(
        new_gmadb['prior'], new_gmadb['datablocks'],
        mt6_ppp=mt6_ppp, optim_type=optim_type, optim_opts=optim_opts
    )

    is_adj = model_info['is_adj']
    priortable = model_info['priortable']
    expvals = model_info['expvals']
    expcov_list = model_info['expcov_list']
    optim_func = model_info['optim_func']

    startvals = tf.constant(
        priortable.loc[is_adj, 'PRIOR'].to_numpy(),
        dtype=tf.float64
    )

    for i in range(max_iters):
        print(f'Outer iteration: {i}')
        # determine posterior
        optres = optim_func(startvals)
        # udpate prior values in unreduced database
        update_unreduced_gmadb_prior(
            orig_gmadb, priortable, is_adj, optres.numpy()
        )
        # perform reduction to obtain GMA database
        new_gmadb = reduce_database(orig_gmadb)
        # retrieve updated reduced experimental data.
        old_expvals = expvals.numpy()
        exptable = create_experiment_table(new_gmadb['datablocks'])
        expcov = create_experimental_covmat(
            new_gmadb['datablocks'], relative=True
        ).toarray()
        # update the tf.Variables expvals and expcov_list.
        # these changes are also registered in the likelihood instance 
        expvals = update_expvals(exptable, expvals)
        expcov_list = update_expcov_list(exptable, expcov, expcov_list) 

        delta = expvals.numpy() - old_expvals
        delta_norm = np.linalg.norm(delta)
        expvals_norm = np.linalg.norm(expvals)
        relative_change = delta_norm / expvals_norm
        print(f'relative change: {relative_change}')
        if relative_change <= rel_tol:
            break
        # use the current best estimate
        # as starting value in next iteration.
        startvals = optres 

    num_iters = i+1

    return {
        'orig_gmadb': orig_gmadb,
        'new_gmadb': new_gmadb,
        'num_iters': num_iters
    }
