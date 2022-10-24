import numpy as np
import pandas as pd
from collections import OrderedDict



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

        elif item['type'] == 'modern-fission-spectrum':
            # the static part of the fission spectrum
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'fis_modern'
            prd['REAC'] = 'NA'
            prd['ENERGY'] = item['energies']
            prd['PRIOR'] = item['spectrum']
            prd['UNC'] = 0.
            prd['DATA'] = np.nan
            prd['DESCR'] = 'modern fission spectrum'
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)
            # the variables associated with the errors
            cov_ens = item['covmat']['energies'][:-1]
            prd = prior_dic = OrderedDict()
            prd['NODE'] = 'fis_errors'
            prd['REAC'] = 'NA'
            prd['ENERGY'] = cov_ens
            prd['PRIOR'] = np.full(len(cov_ens), 0.)
            prd['UNC'] = np.nan
            prd['DATA'] = np.nan
            prd['DESCR'] = 'modern fission spectrum errors'
            curdf = pd.DataFrame.from_dict(prd)
            df.append(curdf)
        else:
            raise ValueError('Unknown type "%s" of prior block' % item['type'])

    df = pd.concat(df, axis=0, ignore_index=True)
    return df



def create_experiment_table(datablock_list):
    """Extract experiment dataframe from datablock list."""
    df_list = []
    for dbidx, db in enumerate(datablock_list):
        if db['type'] != 'legacy-experiment-datablock':
            raise ValueError('Datablock must be of type "legacy-experiment-datablock"')
        for dsidx, ds in enumerate(db['datasets']):
            curdf = pd.DataFrame.from_dict({
                'NODE': 'exp_' + str(ds['NS']),
                'REAC': 'MT:' + str(ds['MT']) +
                        ''.join(['-R%d:%d'%(i+1,r) for i,r in enumerate(ds['NT'])]),
                'ENERGY': ds['E'],
                'PRIOR':  0.,
                'UNC':    np.nan,
                'DATA':   ds['CSS'],
                'DB_IDX': dbidx,
                'DS_IDX': dsidx
            })
            df_list.append(curdf)

    expdf = pd.concat(df_list, ignore_index=True)
    cols = ['NODE', 'REAC', 'ENERGY', 'DATA', 'DB_IDX', 'DS_IDX']
    expdf = expdf.reindex(columns = cols)
    return expdf

