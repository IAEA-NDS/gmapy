import numpy as np
import pandas as pd
import re


SHAPE_MT_IDS = (2,4,8,9)


def attach_shape_prior(datatable, mapping, refvals=None, uncs=None):
    """Attach experimental normalization constants to prior."""
    if refvals is None:
        raise ValueError('Please provide reference values for propagation')
    if uncs is None:
        raise ValueError('Please provide uncertainties for weighting')

    # split datatable into priortable and exptable
    priortable = datatable[datatable['NODE'].str.match('xsid_')]
    exptable = datatable[datatable['NODE'].str.match('exp_')]

    # obtain all experimental points that are affected by unknown normalization
    mtnums = exptable['REAC'].str.extract('^ *MT:([0-9]+)-', expand=False)
    mtnums = mtnums.astype('int')
    is_shape = mtnums.map(lambda x: True if x in SHAPE_MT_IDS else False).to_numpy()

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
    ext_datatable = pd.concat([datatable, norm_df], axis=0, ignore_index=True)
    ext_refvals = np.concatenate([refvals, norm_df['PRIOR'].to_numpy()])

    # calculate a first estimate of normalization factor
    # using the propagated prior values
    propvals = mapping.propagate(ext_datatable, ext_refvals)
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

        normmask = ext_datatable['NODE'] == re.sub('^exp_', 'norm_', cur_exp)
        ext_datatable.loc[normmask, 'PRIOR'] = cur_expscale
        ext_datatable.loc[normmask, 'UNC'] = np.inf

    return ext_datatable



def update_dummy_datapoints(datatable, refvals):
    """Replace values of dummy datapoints by those in refvals."""
    sel = datatable['NODE'].str.fullmatch('exp_90[0-9]')
    datatable.loc[sel, 'DATA'] = refvals[datatable.loc[sel].index]



def update_dummy_datapoints2(datablock_list, refvals):
    cur_idx = 0
    for db in datablock_list:
        for ds in db['datasets']:
            next_idx = cur_idx + len(ds['CSS'])
            if re.match('^90[0-9]$', str(ds['NS'])):
                ds['CSS'] = refvals[cur_idx:next_idx]
            cur_idx = next_idx



def remove_dummy_datasets(datablock_list):
    dummy_db_idcs = []
    for db_idx, db in enumerate(datablock_list):
        dummy_ds_idcs = []
        for ds_idx, ds in enumerate(db['datasets']):
            if re.match('^90[0-9]$', str(ds['NS'])):
                dummy_ds_idcs.append(ds_idx)
        if not np.all(dummy_ds_idcs == np.arange(len(dummy_ds_idcs))):
            raise IndexError('mix of dummy and non-dummy datasets in datablock not allowed')
        if len(dummy_ds_idcs) > 0:
            dummy_db_idcs.append(db_idx)
    for db_idx in reversed(dummy_db_idcs):
        del datablock_list[db_idx]



def propagate_mesh_css(datatable, mapping, refvals):
    refvals = refvals.copy()
    # set temporarily normalization factors to 1.
    # to obtain the cross section. Otherwise, we
    # would obtain the cross section renormalized
    # with the experimental normalization factor
    selidx = datatable[datatable['NODE'].str.match('norm_')].index
    refvals[selidx] = 1.
    # calculate PPP correction
    propvals = mapping.propagate(datatable, refvals)
    return propvals



def calculate_PPP_correction(datatable, mapping, refvals, uncs):
    """Calculate the PPP corrected uncertainties."""
    # calculate PPP correction
    propvals = propagate_mesh_css(datatable, mapping, refvals)
    effuncs = uncs * propvals / datatable['DATA'].to_numpy()
    # but no PPP correction for fission averages
    is_sacs = (datatable['REAC'].str.match('MT:6-') &
               datatable['NODE'].str.match('exp_'))

    sacs_idx = datatable[is_sacs].index
    effuncs[sacs_idx] = uncs[sacs_idx]
    return effuncs

