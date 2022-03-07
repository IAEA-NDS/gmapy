import numpy as np
import pandas as pd
import re
from .compound_map import CompoundMap



def attach_shape_prior(priortable, exptable, refvals=None, uncs=None):
    """Attach experimental normalization constants to prior."""
    if refvals is None:
        raise ValueError('Please provide reference values for propagation')
    if uncs is None:
        raise ValueError('Please provide uncertainties for weighting')

    # obtain all experimental points that are affected by unknown normalization
    shape_mt_ids = (2,4,8,9)
    is_shape = exptable['REAC'].str.match('MT:[2489]-')
    shape_exp_df = exptable[is_shape]
    exp_groups = shape_exp_df.groupby('NODE', sort=False)
    # augment the prior with the experimental normalization factors
    norm_prior_dic = {'NODE': [], 'PRIOR': [], 'REAC': [], 'ENERGY': []}
    norm_index_dic = {}
    for cur_exp, cur_exp_df in exp_groups:
        norm_prior_dic['NODE'].append(re.sub('^exp_', 'norm_', cur_exp))
        norm_prior_dic['PRIOR'].append(1.)
        norm_prior_dic['REAC'].append('NA')
        norm_prior_dic['ENERGY'].append(0.)
    norm_df = pd.DataFrame.from_dict(norm_prior_dic)
    ext_priortable = pd.concat([priortable, norm_df], axis=0, ignore_index=True)
    ext_refvals = np.concatenate([refvals, norm_df['PRIOR'].to_numpy()])

    # calculate a first estimate of normalization factor
    # using the propagated prior values
    compmap = CompoundMap()
    propvals = compmap.propagate(ext_priortable, exptable, ext_refvals)
    for cur_exp, cur_exp_df in exp_groups:
        cur_propvals = propvals[cur_exp_df.index]
        cur_uncs = uncs[cur_exp_df.index]
        cur_expvals = cur_exp_df['DATA'].to_numpy()

        invsquareuncs = 1. / np.square(cur_uncs)
        weight = invsquareuncs / np.sum(invsquareuncs)
        invexpscale = np.sum(weight * cur_propvals / cur_expvals)
        # factor to apply to propagated values
        # to obtain experimental values
        cur_expscale = 1. / invexpscale

        normmask = ext_priortable['NODE'] == re.sub('^exp_', 'norm_', cur_exp)
        ext_priortable.loc[normmask, 'PRIOR'] = cur_expscale

    return ext_priortable



def update_dummy_datapoints(exptable, refvals):
    """Replace values of dummy datapoints by those in refvals."""
    sel = exptable['NODE'].str.fullmatch('exp_90[0-9]')
    exptable.loc[sel, 'DATA'] = refvals[exptable.loc[sel].index]

