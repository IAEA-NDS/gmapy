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


def create_prior_table(prior_list):
    curid = 0
    df = []
    for item in prior_list:
        if item['type'] == 'legacy-prior-cross-section':
            curid += 1
            xsid = item['ID']
            # error checking
            if curid != xsid:
                raise IndexError('prior items must be sorted according to ID in prior_list ' +
                        'but prior item with ID %d violates this constraint' % xsid)
            if not np.all(np.sort(item['EN']) == item['EN']):
                raise ValueError('Energies of prior mesh must be sorted, but this is ' +
                        'not the case for prior block with ID %d' % xsid)
            if not len(item['EN']) == len(item['CS']):
                raise IndexError('Energy mesh and cross sections are of unequal length ' +
                        'for prior block with ID %d' % xsid)

            # append to the dataframe
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'xsid_' + str(xsid)
            prd['REAC'] = 'MT:1-R1:' + str(xsid)
            prd['ENERGY'] = item['EN']
            prd['PRIOR'] = item['CS']
            prd['UNC'] = np.inf
            prd['DESCR'] = item['CLAB'].strip()
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)

        elif item['type'] == 'legacy-fission-spectrum':
            # error checking
            if not np.all(np.sort(item['ENFIS']) == item['ENFIS']):
                raise ValueError('Energies of prior mesh of fission spectrum must be sorted ' +
                        'but this is not the case for the the legacy-fission-spectrum')
            if not len(item['ENFIS']) == len(item['FIS']):
                raise IndexError('Energy mesh and fission spectrum values must be of same length')
            # append to the dataframe
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'fis'
            prd['REAC'] = 'NA'
            prd['ENERGY'] = item['ENFIS']
            prd['PRIOR'] = item['FIS']
            prd['UNC'] = 0.
            prd['DATA'] = np.nan
            prd['DESCR'] = 'fission spectrum'
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)

        else:
            raise ValueError('Unknown type "%s" of prior block' % item['type'])

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
