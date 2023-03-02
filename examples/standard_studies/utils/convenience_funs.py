import sys
sys.path.append('..')
sys.path.append('../..')
from copy import deepcopy
import pandas as pd
import numpy as np
from gmapy.data_management.dataset import Dataset
from gmapy.data_management.datablock import Datablock
from gmapy.data_management.datablock_list import DatablockList
from gmapy.mappings.compound_map import CompoundMap
import json


def print_dataset_info(dataset):
    print(f'---- DATASET {dataset["NS"]} ----')
    print(f'author: {dataset["CLABL"].strip()} {dataset["YEAR"]}')
    NT_nums = ','.join([str(d) for d in dataset['NT'].tolist()])
    print(f'MT: {dataset["MT"]} - NT: {NT_nums}')
    print('')


def remove_datasets(datablock_list, filterfun):
    remove_ids = []
    for curblock in datablock_list:
        curdatasets = curblock['datasets']
        for curdataset in curdatasets:
            if filterfun(curdataset):
                print_dataset_info(curdataset)
                remove_ids.append(curdataset['NS'])
    remove_ids = [int(v) for v in remove_ids]
    dblist = DatablockList(deepcopy(datablock_list))
    dblist.remove_datasets(remove_ids)
    return(dblist.get_datablock_list())


def remove_absolute_datasets(datablock_list, reacid):
    def filterfun(ds):
        return ds['MT'] == 1 and ds['NT'] == reacid 
    return remove_datasets(datablock_list, filterfun)


def remove_ratio_datasets(datablock_list, reacid1, reacid2):
    def filterfun(ds):
        if ds['MT'] != 3:
            return False
        if reacid1 is not None and ds['NT'][0] != reacid1:
            return False
        if reacid2 is not None and ds['NT'][1] != reacid2:
            return False
        return True
    return remove_datasets(datablock_list, filterfun)


def get_sacs_predictions(datatable, covmat=None):
    compmap = CompoundMap(fix_sacs_jacobian=True,
                          legacy_integration=False)
    dt = datatable
    # keep only cross section and fission
    mask = np.logical_or(dt['NODE'].str.match('xsid_'), dt['NODE'] == 'fis')
    tmpdt = dt[mask] 
    # add ratio of sacs measurements
    dt_u5 = pd.DataFrame.from_dict({'NODE': ['exp_sacs_u5'], 'REAC': ['MT:6-R1:8'], 'ENERGY': [0.]})
    dt_pu9 = pd.DataFrame.from_dict({'NODE': ['exp_sacs_pu9'], 'REAC': ['MT:6-R1:9'], 'ENERGY': [0.]})
    dt_u8 = pd.DataFrame.from_dict({'NODE': ['exp_sacs_u8'], 'REAC': ['MT:6-R1:10'], 'ENERGY': [0.]})
    tmpdt = pd.concat([datatable, dt_u5, dt_u8, dt_pu9], axis=0, ignore_index=True)
    # propagate 
    compmap = CompoundMap(fix_sacs_jacobian=True, legacy_integration=False)
    refvals = tmpdt['POST'].to_numpy()
    preds = compmap.propagate(refvals, tmpdt)
    if covmat is not None:
        # remove the fission spectrum
        # TODO: (this is hacky and should be changed in the gmapi package)
        S = compmap.jacobian(refvals, tmpdt)
        mask1 = tmpdt['NODE'].str.match('xsid_') | tmpdt['NODE'].str.match('norm_')
        mask2 = tmpdt['NODE'].str.match('exp_sacs_') 
        S = S[mask2,:]
        S = S[:,mask1]
        sacs_cov = S @ covmat @ S.T
        sacs_unc = np.sqrt(np.diag(sacs_cov))
    return {'u5_sacs': float(preds[tmpdt['NODE'] == 'exp_sacs_u5']),
            'u5_sacs_unc': sacs_unc[0],
            'u8_sacs': float(preds[tmpdt['NODE'] == 'exp_sacs_u8']),
            'u8_sacs_unc': sacs_unc[1],
            'pu9_sacs': float(preds[tmpdt['NODE'] == 'exp_sacs_pu9']),
            'pu9_sacs_unc': sacs_unc[2]}


def output_sacs_preds(filename, ref_sacs, sacs_list, comment):
    fout = open(filename, 'a')
    fout.write(comment)
    fout.write('\nreference calculation:')
    pretty_object = json.dumps(ref_sacs, indent=4)
    fout.write(pretty_object)
    fout.write('\n\nsome absolute cross section removed (mt 8: U5(n,f), mt9: Pu9(n,f), mt10: U8(n,f)')
    for curitem in sacs_list:
        pretty_object = json.dumps(curitem, indent=4)
        fout.write(pretty_object)
    fout.close()


def use_mannhart_sacs(datablock_list):
    # we wrap the datablock_list in a dedicated class
    dblist = DatablockList(datablock_list)

    # and remove all existing SACS and ratio of SACS
    dblist.remove_datasets_by_mtnums([6,10])
    id_list = dblist.list_dataset_ids()

    # construct a new datablock with the SACS and ratio of SACS measurements
    new_datablock = Datablock()
    ds = Dataset()
    ds.define_metadata(2500, 1976, 'Heaton U5nf')
    ds.define_quantity(6, [8])
    ds.define_measurements([1.], [1.216])
    ds.add_norm_uncertainty(1.62)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2501, 100, 'Grundl-William U5nf/U8nf')
    ds.define_quantity(10, [8,10])
    ds.define_measurements([1.], [3.73])
    ds.add_norm_uncertainty(1.2)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2504, 100, 'Gundl-Gilliam U5f/Pu9f')
    ds.define_quantity(10, [8,9])
    ds.define_measurements([1.], [0.666])
    ds.add_norm_uncertainty(0.9)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2505, 1976, 'Heaton Pu9f/U5f')
    ds.define_quantity(10, [9,8])
    ds.define_measurements([1.], [1.5])
    ds.add_norm_uncertainty(1.6)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2506, 1976, 'Heaton U8f/U5f')
    ds.define_quantity(10, [10,8])
    ds.define_measurements([1.], [0.2644])
    ds.add_norm_uncertainty(1.32)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2507, 1985, 'Schroeder U8f/U5f')
    ds.define_quantity(10, [10,8])
    ds.define_measurements([1.], [0.269])
    ds.add_norm_uncertainty(1.2)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2508, 1985, 'Schroeder Pu9f/U5f')
    ds.define_quantity(10, [9,8])
    ds.define_measurements([1.], [1.5])
    ds.add_norm_uncertainty(0.8)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2509, 1985, 'Schroeder U5f')
    ds.define_quantity(6, [8])
    ds.define_measurements([1.], [1.234])
    ds.add_norm_uncertainty(1.45)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2510, 1985, 'Schroeder U8f')
    ds.define_quantity(6, [10])
    ds.define_measurements([1.], [0.332])
    ds.add_norm_uncertainty(1.5)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2511, 1985, 'Schroeder Pu9f')
    ds.define_quantity(6, [9])
    ds.define_measurements([1.], [1.844])
    ds.add_norm_uncertainty(1.3)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2512, 1978, 'Knoll U5f')
    ds.define_quantity(6, [8])
    ds.define_measurements([1.], [1.215])
    ds.add_norm_uncertainty(1.79)
    new_datablock.add_datasets(ds)

    ds = Dataset()
    ds.define_metadata(2513, 1978, 'Knoll Pu9')
    ds.define_quantity(6, [9])
    ds.define_measurements([1.], [1.79])
    ds.add_norm_uncertainty(2.26)
    new_datablock.add_datasets(ds)

    # define the correlation matrix
    tmp = np.array([
    [ 100,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],      #   1    Grundl memo ABS (rev. Heaton)
    [  23, 100,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],      #   2    Grundl/Gilliam ratio
    [  -9,  15, 100,   0,   0,   0,   0,   0,   0,   0,   0,   0],      #   4    Grundl/Gilliam ratio
    [   0,   0,   0, 100,   0,   0,   0,   0,   0,   0,   0,   0],      #   5    Heaton ratio
    [   0,   0,   0,  60, 100,   0,   0,   0,   0,   0,   0,   0],      #   6    Heaton ratio
    [   0,   0,   0,   0,   0, 100,   0,   0,   0,   0,   0,   0],      #   7    Schroeder ratio
    [   0,   0,   0,   0,   0,  77, 100,   0,   0,   0,   0,   0],      #   8    Schroeder ratio
    [   0,   0,   0,   0,   0,   0,   0, 100,   0,   0,   0,   0],      #   9    Schroeder ABS
    [   0,   0,   0,   0,   0,   0,   0,  50, 100,   0,   0,   0],      #  10    Schroeder ABS
    [   0,   0,   0,   0,   0,   0,   0,  50,  50, 100,   0,   0],      #  11    Schroeder ABS
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 100,   0],      #  12    Davis/Knoll abs
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 100]       #  13    Davis/Knoll
    ])
    # make the covariance matrix symmetric
    cormat = (tmp + tmp.T) / 1e2
    np.fill_diagonal(cormat, 1)

    new_datablock.define_correlation_matrix(cormat)

    # add it to the list
    dblist.add_datablock(new_datablock)
    id_list = dblist.list_dataset_ids()
    duplicates = [num for num in id_list if id_list.count(num) > 1]
    # TODO: check the duplicate
    if len(duplicates) > 0:
        pass

    # cast it to a normal list that can be understood by run_gmap_simplified
    datablock_list = dblist.get_datablock_list(dict)
    return datablock_list

