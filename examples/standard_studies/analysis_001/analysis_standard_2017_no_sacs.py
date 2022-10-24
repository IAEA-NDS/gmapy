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
        remove_absolute_datasets, output_sacs_preds)

# read the database with NIFFTE TPC data
dbpath = '../../../legacy-tests/test_002/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']
datablock_list = db_dic['datablock_list']

# removing all sacs
dblist = DatablockList(datablock_list)
dblist.remove_datasets_by_mtnums([6,10])
datablock_list = dblist.get_datablock_list()

# reference calculation
gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
ref_table = gls_result['table']
ref_sacs = get_sacs_predictions(ref_table, gls_result['postcov'])

# systematically remove a group of absolute cross sections
# 8: U5(n,f), 9: PU9(n,f), 10: U8(n,f)
exclude_reacids = (8,9,10) 
res_list = []
for cur_reacid in exclude_reacids:
    reduced_dblist = remove_absolute_datasets(datablock_list, cur_reacid)
    gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=reduced_dblist)
    curtable = gls_result['table']
    curcov = gls_result['postcov']
    cur_sacs = get_sacs_predictions(curtable, curcov)
    cur_sacs.update({'removed_reacid': cur_reacid})
    res_list.append(cur_sacs)


output_sacs_preds('output/results_04.txt', ref_sacs, res_list, comment="""
############################################
standards data 2017 but all sacs measurements removed
""")
