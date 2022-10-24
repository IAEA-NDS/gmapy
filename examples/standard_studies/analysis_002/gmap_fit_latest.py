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

# read the database with NIFFTE TPC data
dbpath = '../../../legacy-tests/test_004/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']

# remove datasets
#datablock_list = use_mannhart_sacs(db_dic['datablock_list'])
datablock_list = db_dic['datablock_list']

gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
ref_table = gls_result['table']
ref_sacs = get_sacs_predictions(ref_table, gls_result['postcov'])
