import numpy as np
import pandas as pd
import scipy.sparse as sps
import re
from ..data_management.datablock_api import dataset_iterator
from ..data_management import dataset_api as dsapi
from ..data_management.quantity_types import SHAPE_MT_IDS
import tensorflow as tf


def prepare_prior_and_exptable(datatable, reduce, reset_index=True):
    is_datatable_split = isinstance(datatable, (list, tuple))
    if is_datatable_split:
        priortable = datatable[0]
        exptable = datatable[1]
    else:
        expmask = datatable['NODE'].str.match('exp_')
        priortable = datatable.loc[~expmask]
        exptable = datatable.loc[expmask]
    if not reduce:
        if is_datatable_split:
            raise ValueError(
                'if list with priortable and exptable '
                'is provided, it is required that `reduce=True`'
            )
        src_len = len(datatable)
        tar_len = len(datatable)
    else:
        if reset_index:
            priortable = priortable.reset_index(drop=True)
            exptable = exptable.reset_index(drop=True)
        src_len = len(priortable)
        tar_len = len(exptable)
    return priortable, exptable, src_len, tar_len


def prepare_prior_and_likelihood_quantities(datatable, covmat):
    priortable, exptable, _, _ = prepare_prior_and_exptable(
        datatable, reduce=True, reset_index=False
    )
    priorvals = priortable['PRIOR'].to_numpy(copy=True)
    expvals = exptable['DATA'].to_numpy(copy=True)
    priorcov = covmat[:, priortable.index][priortable.index, :]
    expcov = covmat[:, exptable.index][exptable.index, :]
    priortable = priortable.reset_index(drop=True)
    exptable = exptable.reset_index(drop=True)
    return {
        'priortable': priortable,
        'exptable': exptable,
        'priorvals': priorvals,
        'priorcov': priorcov,
        'expvals': expvals,
        'expcov': expcov,
    }


def attach_shape_prior(datatable, covmat=None, raise_if_exists=True):
    """Attach experimental normalization constants to prior."""
    # split datatable into priortable and exptable if not already done
    if isinstance(datatable, (list, tuple)):
        exptable = datatable[1]
        datatable = datatable[0]
    else:
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
        norm_node_name = re.sub('^exp_', 'norm_', cur_exp)
        if (datatable.NODE == norm_node_name).any():
            if raise_if_exists:
                raise IndexError(
                    f'normalization node {norm_node_name} already exists'
                )
            else:
                continue
        norm_prior_dic['NODE'].append(norm_node_name)
        norm_prior_dic['PRIOR'].append(1.)
        norm_prior_dic['REAC'].append(cur_exp_df['REAC'].iloc[0])
        norm_prior_dic['ENERGY'].append(0.)
    norm_df = pd.DataFrame.from_dict(norm_prior_dic)
    ext_datatable = pd.concat([datatable, norm_df], axis=0, ignore_index=True)
    if covmat is not None:
        normuncs = np.full(len(norm_df), np.inf, dtype='d')
        normcov = sps.diags(np.square(normuncs), dtype='d')
        ext_priorcov = sps.block_diag(
            [covmat, normcov], format='csr', dtype='d'
        )
    if covmat is None:
        return ext_datatable
    else:
        return ext_datatable, ext_priorcov


def initialize_shape_prior(datatable, mapping=None, refvals=None, uncs=None):
    # split datatable into priortable and exptable if not already done
    if isinstance(datatable, (list, tuple)):
        exptable = datatable[1]
        datatable = datatable[0]
    else:
        exptable = datatable[datatable['NODE'].str.match('exp_', na=False)]
    mtnums = exptable['REAC'].str.extract('^ *MT:([0-9]+)-', expand=False)
    mtnums = mtnums.astype('int')
    is_shape = mtnums.map(lambda x: True if x in SHAPE_MT_IDS else False).to_numpy()
    shape_exp_df = exptable[is_shape]
    exp_groups = shape_exp_df.groupby('NODE', sort=False)
    normmask = datatable['NODE'].str.match('^norm_', na=False)
    datatable.loc[normmask, 'PRIOR'] = 1.
    datatable.loc[normmask, 'UNC'] = np.inf
    if refvals is None and 'PRIOR' in datatable:
        refvals = np.empty((len(datatable),), dtype='d')
        refvals[datatable.index] = datatable.PRIOR.to_numpy()
    else:
        refvals = np.array(refvals)
        norm_idcs = datatable.index[normmask].to_numpy()
        refvals[norm_idcs] = 1.
    if uncs is None and 'UNC' in exptable:
        uncs = np.empty((np.max(exptable.index)+1,), dtype='d')
        uncs[exptable.index] = (exptable.UNC / exptable.DATA).to_numpy()
    if not (mapping is None or refvals is None or uncs is None):
        # calculate a first estimate of normalization factor
        # using the propagated prior values
        try:
            propvals = mapping.propagate(refvals)
        except TypeError:
            # invoke old style propagate interface which is
            # not supported by tensorflow CompoundMap anymore
            propvals = mapping.propagate(refvals, datatable)
        # required type conversion because indexing of
        # tf.Tensors by numpy arrays does not work
        if isinstance(propvals, tf.Tensor):
            propvals = propvals.numpy()
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
    else:
        raise IndexError(
            'Could not initialize normalization factors because ' +
            'essential information is missing: \n' +
            '- either provide refvals arg or column `PRIOR` in datatable \n' +
            '- either provide uncs arg or column `UNC` in datatable \n' +
            '- provide arg compmap with a CompoundMapping object')

    return None


def update_dummy_datapoints(datatable, refvals):
    """Replace values of dummy datapoints by those in refvals."""
    sel = datatable['NODE'].str.fullmatch('exp_90[0-9]', na=False)
    datatable.loc[sel, 'DATA'] = refvals[datatable.loc[sel].index]


def update_dummy_datapoints2(datablock_list, refvals):
    cur_idx = 0
    for db in datablock_list:
        datasets = tuple(dataset_iterator(db))
        for ds in datasets:
            dstype = dsapi.get_dataset_type(ds)
            dsid = dsapi.get_dataset_identifier(ds)
            css = dsapi.get_measured_values(ds)
            next_idx = cur_idx + len(css)
            if (dstype == 'legacy-experiment-dataset' and
                    re.match('^90[0-9]$', str(dsid))):
                dsapi.add_measured_values(ds, refvals[cur_idx:next_idx])
            cur_idx = next_idx


def remove_dummy_datasets(datablock_list):
    dummy_db_idcs = []
    for db_idx, db in enumerate(datablock_list):
        dummy_ds_idcs = []
        datasets = tuple(dataset_iterator(db))
        for ds_idx, ds in enumerate(datasets):
            dstype = dsapi.get_dataset_type(ds)
            dsid = dsapi.get_dataset_identifier(ds)
            if (dstype == 'legacy-experiment-dataset' and
                    re.match('^90[0-9]$', str(dsid))):
                dummy_ds_idcs.append(ds_idx)
        if len(dummy_ds_idcs) > 0:
            dummy_db_idcs.append(db_idx)
    for db_idx in reversed(dummy_db_idcs):
        del datablock_list[db_idx]


def create_propagate_source_mask(
    datatable, prop_normfact=False, prop_usu_errors=False
):
    idcs_list = []
    vals_list = []
    if not prop_normfact:
        norm_selidx = datatable[datatable['NODE'].str.match('norm_', na=False)].index
        idcs_list.append(norm_selidx)
        vals_list.append(np.full(len(norm_selidx), 1.))
    if not prop_usu_errors:
        usu_selidx = datatable[datatable['NODE'].str.match('usu_|endep_usu_', na=False)].index
        idcs_list.append(usu_selidx)
        vals_list.append(np.full(len(usu_selidx), 0.))
    if len(idcs_list) > 0:
        idcs = np.concatenate(idcs_list)
        vals = np.concatenate(vals_list)
    else:
        idcs = np.empty(0, dtype=int)
        vals = np.empty(0, dtype=float)
    return {'idcs': idcs, 'vals': vals}


def create_propagate_target_mask(datatable, mt6_exp=False):
    idcs_list = []
    vals_list = []
    if mt6_exp:
        dt = datatable
        mt6_idcs = dt.index[(dt.NODE.str.match('exp_') &
                            dt.REAC.str.match('MT:6-R1:'))]
        mt6_vals = datatable.loc[mt6_idcs, 'DATA'].to_numpy(copy=True)
        idcs_list.append(mt6_idcs)
        vals_list.append(mt6_vals)
    if len(idcs_list) > 0:
        idcs = np.concatenate(idcs_list)
        vals = np.concatenate(vals_list)
    else:
        idcs = np.empty(0, dtype=int)
        vals = np.empty(0, dtype=float)
    return {'idcs': idcs, 'vals': vals}


def apply_mask(arr, mask):
    if mask is None:
        return arr
    idcs = mask['idcs']
    vals = mask['vals']
    if len(arr.shape) == 1:
        arr[idcs] = vals
    elif len(arr.shape) == 2:
        arr[idcs, :] = vals.reshape(-1, 1)
    return arr


def propagate_mesh_css(datatable, mapping, refvals, prop_normfact=False,
                       mt6_exp=False, prop_usu_errors=False, prop_relerr=False):
    refvals = refvals.copy()
    # set temporarily normalization factors to 1.
    # to obtain the cross section. Otherwise, we
    # would obtain the cross section renormalized
    # with the experimental normalization factor
    source_mask = create_propagate_source_mask(
        datatable, prop_normfact=prop_normfact, prop_usu_errors=prop_usu_errors
    )
    # the substitution of propagated values by experimental ones
    # for MT6 (SACS) is there to facilitate the PPP correction
    # as done by Fortran GMAP, which does not apply it to MT6.
    target_mask = create_propagate_target_mask(
        datatable, mt6_exp=mt6_exp
    )
    save_vals = refvals[source_mask['idcs']]
    apply_mask(refvals, source_mask)
    propvals = mapping.propagate(refvals, datatable)
    apply_mask(propvals, target_mask)
    propvals[source_mask['idcs']] = save_vals
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
