import numpy as np
import pandas as pd
from collections import OrderedDict
from .datablock_api import (
    dataset_iterator
)
from . import dataset_api as dsapi
from . import priorblock_api as priorapi


def create_prior_table(prior_list):
    df = []
    for item in prior_list:
        # append to the dataframe
        quant_type = priorapi.get_quantity_type(item)
        reac_id = priorapi.get_reaction_identifier(item)
        reacstr = f'MT:{quant_type}-R1:{reac_id}'
        prd = OrderedDict()
        prd['NODE'] = priorapi.get_nodename(item)
        prd['REAC'] = reacstr
        prd['ENERGY'] = priorapi.get_energies(item)
        prd['PRIOR'] = priorapi.get_values(item)
        prd['UNC'] = priorapi.get_uncertainties(item)
        prd['DESCR'] = priorapi.get_description(item)
        curdf = pd.DataFrame.from_dict(prd)
        df.append(curdf)

    df = pd.concat(df, axis=0, ignore_index=True)
    return df


def create_dataframe_from_experiment_dataset(
    dataset, datablock_index, dataset_index
):
    ds = dataset
    quant_part = 'MT:' + str(dsapi.get_quantity_type(ds))
    reac_ids = dsapi.get_reaction_identifiers(ds)
    reac_part = ''.join(
        ['-R%d:%d' % (i+1, r) for i, r in enumerate(reac_ids)]
    )
    curdf = pd.DataFrame.from_dict({
        'NODE': 'exp_' + str(dsapi.get_dataset_identifier(ds)),
        'REAC':   quant_part + reac_part,
        'ENERGY': dsapi.get_incident_energies(ds),
        'PRIOR':  0.,
        'UNC':    np.nan,
        'DATA':   dsapi.get_measured_values(ds),
        'DB_IDX': datablock_index,
        'DS_IDX': dataset_index,
        'AUTHOR': dsapi.get_authors_string(ds),
        'PUBREF': dsapi.get_publication_string(ds)
    })
    return curdf


def create_experiment_table(datablock_list):
    """Extract experiment dataframe from datablock list."""
    df_list = []
    for dbidx, db in enumerate(datablock_list):
        datasets = tuple(dataset_iterator(db))
        for dsidx, ds in enumerate(datasets):
            curdf = create_dataframe_from_experiment_dataset(
                ds, dbidx, dsidx
            )
            df_list.append(curdf)

    expdf = pd.concat(df_list, ignore_index=True)
    cols = ['NODE', 'REAC', 'ENERGY', 'DATA',
            'DB_IDX', 'DS_IDX', 'AUTHOR', 'PUBREF']
    expdf = expdf.reindex(columns=cols)
    return expdf
