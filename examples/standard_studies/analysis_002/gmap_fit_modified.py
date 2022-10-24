import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from copy import deepcopy
from gmapy.data_management.datablock_list import DatablockList
from gmapy.gmap import run_gmap_simplified
from gmapy.data_management.database_IO import (read_json_gma_database,
        read_legacy_gma_database)

from utils.convenience_funs import (get_sacs_predictions,
        remove_absolute_datasets, output_sacs_preds, print_dataset_info,
        use_mannhart_sacs, remove_datasets)

def remove_abs_xs_u5(datablock_list):
    def filterfun(ds):
        if ds['MT'] != 1:
            return False
        if ds['NT'] != 8:
            return False
        else:
            return True
    return remove_datasets(datablock_list, filterfun)

def remove_abs_xs_pu9(datablock_list):
    def filterfun(ds):
        if ds['MT'] != 1:
            return False
        if ds['NT'] != 9:
            return False
        return True
    return remove_datasets(datablock_list, filterfun)

def remove_abs_xs_pu9_ecut(datablock_list, ecut):
    def filterfun(ds):
        if ds['MT'] != 1:
            return False
        if ds['NT'] != 9:
            return False
        if min(ds['E']) > ecut:
            return False
        return True
    return remove_datasets(datablock_list, filterfun)

def remove_thermal_axton(datablock_list):
    def filterfun(ds):
        if 'AXTON' not in ds['CLABL'].upper():
            return False
        if ds['MT'] != 1:
            return False
        if ds['NT'][0] not in (8,9):
            return False
        return True
    return remove_datasets(datablock_list, filterfun)


# read the database with NIFFTE TPC data
dbpath = '../../../legacy-tests/test_004/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']

# remove datasets
datablock_list = use_mannhart_sacs(db_dic['datablock_list'])
#datablock_list = remove_abs_xs_u5(datablock_list)
#datablock_list = remove_abs_xs_pu9(datablock_list)
#datablock_list = remove_abs_xs_pu9_ecut(datablock_list, ecut=0.01)
datablock_list = remove_thermal_axton(datablock_list)

for curdb in datablock_list:
    for curds in curdb['datasets']:
        # convert ratio pu9/u5 data to shape ratio
        if (curds['MT'] == 3
            and ((curds['NT'][0] == 9 and curds['NT'][1]==8)
                  or (curds['NT'][0] == 8 and curds['NT'][1]==9))) :
            curds['MT'] = 4
        # convert ratio pu9/u5 or inverse to shape ratio if energy below 45 keV
        #if (curds['MT'] == 3
        #    and (min(curds['E']) < 1)
        #    and ((curds['NT'][0] == 9 and curds['NT'][1]==8)
        #          or (curds['NT'][0] == 8 and curds['NT'][1]==9))) :
        #    curds['MT'] = 4
        # change value of thermal TNC of Pu9(n,f)
        #if curds['NS'] == 925:
        #    curds['CSS'] = [752.4]
        #if curds['NS'] == 919:
        #    curds['CSS'] = [587.3]
        #if curds['NS'] == 920:
        #    curds['CSS'] = [99.5]
        #if curds['NS'] == 926:
        #    curds['CSS'] = [269.8]


gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
ref_table = gls_result['table']
ref_sacs = get_sacs_predictions(ref_table, gls_result['postcov'])
