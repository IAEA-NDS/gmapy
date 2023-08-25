import sys
sys.path.append('../..')
from gmapy.data_management.dataset import Dataset
from gmapy.data_management.datablock import Datablock
from gmapy.data_management.datablock_list import DatablockList
import numpy as np
from gmapy.gmap import run_gmap_simplified
from gmapy.data_management.database_IO import (read_json_gma_database,
        read_legacy_gma_database)

# read the database of standards 2017 and perform the gls fitting
dbpath = '../../legacy-tests/test_002/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']
datablock_list = db_dic['datablock_list']
gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
gls_result['table'].to_csv('gma_result_2017.csv')

# read the database with NIFFTE TPC data and perform the gls fitting
dbpath = '../../legacy-tests/test_004/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']
datablock_list = db_dic['datablock_list']
gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
gls_result['table'].to_csv('gma_result_with_niffte_tpc.csv')

# now we read the database, remove all sacs measumrents
# and add all the sacs and ratio of sacs values used in
# the Mannhart evaluation
# read the json database
#dbpath = '../legacy-tests/test_002/input/gmadata.json'
dbpath = '../../legacy-tests/test_004/input/data.gma'
db_dic = read_legacy_gma_database(dbpath)
prior_list = db_dic['prior_list']
datablock_list = db_dic['datablock_list']

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

# check it
new_datablock.get_covariance_matrix(unit='percent').toarray()
new_datablock.get_correlation_matrix()


# add it to the list
dblist.add_datablock(new_datablock)
id_list = dblist.list_dataset_ids()
duplicates = [num for num in id_list if id_list.count(num) > 1]


# cast it to a normal list that can be understood by run_gmap_simplified
datablock_list = dblist.get_datablock_list(dict)

gls_result = run_gmap_simplified(prior_list=prior_list, datablock_list=datablock_list)
gls_result['table'].to_csv('gma_with_sacs_ratios.csv')
