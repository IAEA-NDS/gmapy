import numpy as np
import pandas as pd
from collections import OrderedDict
from .datablock_api import (
    dataset_iterator
)
from .dataset_api import (
    get_dataset_identifier,
    get_quantity_type,
    get_reaction_identifiers,
    get_measured_values,
    get_incident_energies,
    get_authors_string,
    get_publication_string
)
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
    quant_part = 'MT:' + str(get_quantity_type(ds))
    reac_ids = get_reaction_identifiers(ds)
    reac_part = ''.join(
        ['-R%d:%d' % (i+1, r) for i, r in enumerate(reac_ids)]
    )
    curdf = pd.DataFrame.from_dict({
        'NODE': 'exp_' + str(get_dataset_identifier(ds)),
        'REAC':   quant_part + reac_part,
        'ENERGY': get_incident_energies(ds),
        'PRIOR':  0.,
        'UNC':    np.nan,
        'DATA':   get_measured_values(ds),
        'DB_IDX': datablock_index,
        'DS_IDX': dataset_index,
        'AUTHOR': get_authors_string(ds).strip(),
        'PUBREF': get_publication_string(ds).strip()
    })
    return curdf


def create_experiment_table(datablock_list):
    """Extract experiment dataframe from datablock list."""
    df_list = []
    for dbidx, db in enumerate(datablock_list):
        if db['type'] not in (
            'legacy-experiment-datablock', 'simple-experiment-datablock'
        ):
            raise ValueError('Unsupported type of datablock')
        for dsidx, ds in enumerate(db['datasets']):
            if ds['type'] == 'legacy-experiment-dataset':
                curdf = create_dataframe_from_experiment_dataset(
                    ds, dbidx, dsidx
                )
            else:
                raise ValueError('Unsupported type of dataset')
            df_list.append(curdf)

    expdf = pd.concat(df_list, ignore_index=True)
    cols = ['NODE', 'REAC', 'ENERGY', 'DATA', 'DB_IDX', 'DS_IDX', 'AUTHOR', 'PUBREF']
    expdf = expdf.reindex(columns=cols)
    return expdf
