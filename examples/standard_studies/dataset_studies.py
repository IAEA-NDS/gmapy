# Identify all SACS measurements in the STD 2017 database

import sys
sys.path.append('..')
sys.path.append('../..')
from copy import deepcopy
from gmapi.data_management.datablock_list import DatablockList
from gmapi.gmap import run_gmap_simplified
from gmapi.data_management.database_IO import (read_json_gma_database,
        read_legacy_gma_database)

from convenience_funs import print_dataset_info

# read the database with NIFFTE TPC data
dbpath = '../../legacy-tests/test_002/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']
datablock_list = db_dic['datablock_list']

for curdb in datablock_list:
    for curds in curdb['datasets']:
        if curds['MT'] == 6:
            print_dataset_info(curds)
            print(f'value: {float(curds["CSS"])}')
            print('\n\n')

