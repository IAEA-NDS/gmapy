import numpy as np
import pandas as pd
import re


SHAPE_MT_IDS = (2,4,8,9)


def prepare_prior_and_exptable(datatable, reduce):
    expmask = datatable['NODE'].str.match('exp_')
    priortable = datatable.loc[~expmask]
    exptable = datatable.loc[expmask]
    if not reduce:
        src_len = len(datatable)
        tar_len = len(datatable)
    else:
        datatable = datatable.sort_index(inplace=False)
        priortable = priortable.reset_index(drop=True)
        exptable = exptable.reset_index(drop=True)
        src_len = len(priortable)
        tar_len = len(exptable)
    return priortable, exptable, src_len, tar_len


def attach_shape_prior(datatable):
    """Attach experimental normalization constants to prior."""
    # split datatable into priortable and exptable
    exptable = datatable[datatable['NODE'].str.match('exp_', na=False)]

    # obtain all experimental points that are affected by unknown normalization
    mtnums = exptable['REAC'].str.extract('^ *MT:([0-9]+)-', expand=False)
    mtnums = mtnums.astype('int')
    is_shape = mtnums.map(lambda x: True if x in SHAPE_MT_IDS else False).to_numpy()

    shape_exp_df = exptable[is_shape]
    exp_groups = shape_exp_df.groupby('NODE', sort=False)
    # augment the prior with the experimental normalization factors
    norm_prior_dic = {'NODE': [], 'PRIOR': [], 'REAC': [], 'ENERGY': []}
    for cur_exp, cur_exp_df in exp_groups:
        norm_prior_dic['NODE'].append(re.sub('^exp_', 'norm_', cur_exp))
        norm_prior_dic['PRIOR'].append(1.)
        norm_prior_dic['REAC'].append(cur_exp_df['REAC'].iloc[0])
        norm_prior_dic['ENERGY'].append(0.)
    norm_df = pd.DataFrame.from_dict(norm_prior_dic)
    ext_datatable = pd.concat([datatable, norm_df], axis=0, ignore_index=True)
    return ext_datatable


def initialize_shape_prior(datatable, mapping=None, refvals=None, uncs=None):
    exptable = datatable[datatable['NODE'].str.match('exp_', na=False)]
    mtnums = exptable['REAC'].str.extract('^ *MT:([0-9]+)-', expand=False)
    mtnums = mtnums.astype('int')
    is_shape = mtnums.map(lambda x: True if x in SHAPE_MT_IDS else False).to_numpy()
    shape_exp_df = exptable[is_shape]
    exp_groups = shape_exp_df.groupby('NODE', sort=False)
    if mapping is None or refvals is None or uncs is None:
        # we don't have any prior estimates to propagate
        # and/or uncertainties of the experiments given
        # so we assume that the normalization factors are one
        normmask = datatable['NODE'].str.match('^norm_', na=False)
        datatable.loc[normmask, 'UNC'] = np.inf
        datatable.loc[normmask, 'PRIOR'] = 1.
    else:
        # calculate a first estimate of normalization factor
        # using the propagated prior values
        propvals = mapping.propagate(refvals, datatable)
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

            normmask = datatable['NODE'] == re.sub('^exp_', 'norm_', cur_exp)
            datatable.loc[normmask, 'PRIOR'] = cur_expscale
            datatable.loc[normmask, 'UNC'] = np.inf

    return None


def update_dummy_datapoints(datatable, refvals):
    """Replace values of dummy datapoints by those in refvals."""
    sel = datatable['NODE'].str.fullmatch('exp_90[0-9]', na=False)
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


def propagate_mesh_css(datatable, mapping, refvals, prop_normfact=False, mt6_exp=False,
                       prop_usu_errors=False, prop_relerr=False):
    refvals = refvals.copy()
    # set temporarily normalization factors to 1.
    # to obtain the cross section. Otherwise, we
    # would obtain the cross section renormalized
    # with the experimental normalization factor
    if not prop_normfact:
        norm_selidx = datatable[datatable['NODE'].str.match('norm_', na=False)].index
        normvals = refvals[norm_selidx]
        refvals[norm_selidx] = 1.
    # or renomalized by the usu errors...
    if not prop_usu_errors:
        usu_selidx = datatable[datatable['NODE'].str.match('usu_', na=False)].index
        usuvals = refvals[usu_selidx]
        refvals[usu_selidx] = 0.
    # or relative error components
    if not prop_relerr:
        relerr_selidx = datatable[datatable['NODE'].str.match('relerr_', na=False)].index
        relerrvals = refvals[relerr_selidx]
        refvals[relerr_selidx] = 0.
    # calculate PPP correction
    propvals = mapping.propagate(refvals, datatable)
    if not prop_normfact:
        propvals[norm_selidx] = normvals
    if not prop_usu_errors:
        propvals[usu_selidx] = usuvals
    if not prop_relerr:
        propvals[relerr_selidx] = relerrvals
    # the substitution of propagated values by experimental ones
    # for MT6 (SACS) is there to facilitate the PPP correction
    # as done by Fortran GMAP, which does not apply it to MT6.
    if mt6_exp:
        dt = datatable
        mt6_idcs = dt.index[(dt.NODE.str.match('exp_') &
                            dt.REAC.str.match('MT:6-R1:'))]
        propvals[mt6_idcs] = datatable.loc[mt6_idcs, 'DATA']
    return propvals


def calculate_PPP_correction(datatable, mapping, refvals, uncs):
    """Calculate the PPP corrected uncertainties."""
    # calculate PPP correction
    propvals = propagate_mesh_css(datatable, mapping, refvals)
    effuncs = uncs * propvals / datatable['DATA'].to_numpy()
    # but no PPP correction for fission averages
    is_sacs = (datatable['REAC'].str.match('MT:6-', na=False) &
               datatable['NODE'].str.match('exp_', na=False))

    sacs_idx = datatable[is_sacs].index
    effuncs[sacs_idx] = uncs[sacs_idx]
    return effuncs
