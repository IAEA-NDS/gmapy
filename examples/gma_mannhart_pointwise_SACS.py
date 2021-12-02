from database_reading import read_gma_database
from database_writing_utils import (extract_dataset_from_datablock,
        extract_prior_datatable, extract_experiment_datatable,
        extract_fission_datatable, generate_dataset_text,
        write_mode_setup, write_IPP_setup, write_fission_text,
        write_prior_text)
from GMAP import run_GMA_program

from gmap_snippets import label_to_MT

import pandas as pd
import numpy as np
import os


# load the database and reduce it to the reactions in the Mannhart evaluation
orig_dbfile = 'tests/test_002/input/data.gma'
db = read_gma_database(orig_dbfile)
datablocks = db['datablock_list']

MODC = db['datablock_list'][0].MODC
MOD2 = db['datablock_list'][0].MOD2
MPPP = db['MPPP']
MODAP = db['MODAP']
IPP = db['IPP']

prior_dt = extract_prior_datatable(db['APR'])
fission_dt = extract_fission_datatable(db['fisdata'])
experiment_dt = extract_experiment_datatable(db['datablock_list'], prior_dt)

# some small overview
prior_dt.reac.unique()
experiment_dt.obstype.unique()
experiment_dt[experiment_dt.obstype == 'fission_average']

# reduce prior to reactions of Mannhart evaluation 
reacs = ['U5(n,f)', 'U8(n,f)', 'PU9(n,f)']

prior_dt = prior_dt[prior_dt['reac'].isin(reacs)]
tr = {
    'U5(n,f)': 1,
    'U8(n,f)': 2,
    'PU9(n,f)': 3
}

prior_dt.reset_index(inplace=True, drop=True)
for currow in range(len(prior_dt.index)):
    curreac = prior_dt.iloc[currow].reac
    prior_dt.at[currow,'xsid'] = tr[curreac]

prior_dt.sort_values(by='xsid', inplace=True)
prior_dt.reset_index(inplace=True, drop=True)

# set up the experimental data

# mannhart prior
#  1 1.205E+3 50.0000    U-235(N,F)  PRIO
#  2 1.332E+3 50.0000    NP-237(N,F) PRIO
#  3 3.180E+2 50.0000    U-238(N,F)  PRIO
#  4 1.807E+3 50.0000    PU-239(N,F) PRIO
man_gma_dic = {
    1: 1, # U5(n,f)
    3: 2, # U8(n,f)
    4: 3  # PU9(n,f)
}


expdata = [
[ 1,  1, 0, 1.216E+3, 1.6200,  'U235F',         'GRUNDL MEMO (Rev. Heaton 1976 ABSOL U-235f),indc(nds)-146, p.237'],
[ 2,  1, 3, 3.73    , 1.2000,  'U235F/U238F',   'GRUNDL-GILLIAM indc(nds)-146, p.244, inverse = 0.2681'],
[ 3,  2, 1, 1.123E+0, 1.4700,  'NP237F/U235F',  'derived from GRUNDL-GILLIAM indc(nds)-146, p.244, 1.123 = 1366 (1.2%) (Np) / 1216 (1.6%)(U5)'],
[ 4,  1, 4, 0.666   , 0.9000,  'U235F/PU239F',  'GRUNDL-GILLIAM indc(nds)-146, p.244, inverse = 1.502 (0.9%)'],
[ 5,  4, 1, 1.500E+0, 1.6000,  'PU239F/U235F',  'Heaton 1976 ANL-76-90 60% ratio correl derived from common ratio uncert.'],
[ 6,  3, 1, 0.2644  , 1.3200,  'U238F/U235F',   'Heaton 1976 ANL-76-90 60% ratio correl derived from common ratio uncert.'],
[ 7,  3, 1, 0.269E+0, 1.2000,  'U238F/U235F',   'Schroeder JANS 50(1985)154'],
[ 8,  4, 1, 1.500E+0, 0.8000,  'PU239F/U235F',  'Schroeder JANS 50(1985)154'],
[ 9,  1, 0, 1.234E+3, 1.4500,  'U235F',         'Schroeder JANS 50(1985)154'],
[10,  3, 0, 0.332E+3, 1.5000,  'U238F',         'Schroeder JANS 50(1985)154'],
[11,  4, 0, 1.844E+3, 1.3000,  'PU239F',        'Schroeder JANS 50(1985)154'],
[12,  1, 0, 1.215E+3, 1.7900,  'U235F',         'Davis/Knoll 1978'],
[13,  4, 0, 1.790E+3, 2.2600,  'PU239F',        'Davis/Knoll 1978']
]

# xs [mbarn]
# unc [%]
expdata_dt = pd.DataFrame.from_records(expdata, columns = ['idx', 'reac_id1', 'reac_id2', 'xs', 'unc', 'reacstr', 'descr'])

# convert to barn for GMAP
expdata_dt['xs'] /= 1000

# correlation matrix
cormat1d = [ 
  100,                                                                   #    1    Grundl memo ABS (rev. Heaton)
   23, 100,                                                              #    2    Grundl/Gilliam ratio
   -7,  29, 100,                                                         #    3    Grundl/Gilliam ratio
   -9,  15,  36,  100,                                                   #    4    Grundl/Gilliam ratio
    0,   0,   0,    0, 100,                                              #    5    Heaton ratio
    0,   0,   0,    0,  60, 100,                                         #    6    Heaton ratio
    0,   0,   0,    0,   0,   0, 100,                                    #    7    Schroeder ratio
    0,   0,   0,    0,   0,   0,  77, 100,                               #    8    Schroeder ratio
    0,   0,   0,    0,   0,   0,   0,   0, 100,                          #    9    Schroeder ABS
    0,   0,   0,    0,   0,   0,   0,   0,  50, 100,                     #   10    Schroeder ABS
    0,   0,   0,    0,   0,   0,   0,   0,  50,  50, 100,                #   11    Schroeder ABS 
    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,  100,          #   12    Davis/Knoll abs
    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   59, 100      #   13    Davis/Knoll abs
]


# make symmetric correlation matrix
dim = int(np.round(-1/2 + np.sqrt(1/4 + 2*len(cormat1d))))
cormat2d = np.zeros((dim,dim))
cormat2d[np.tril_indices(dim)] = cormat1d
diagels = np.diag(cormat2d).copy()
cormat2d += cormat2d.T
np.fill_diagonal(cormat2d, diagels)

# prepare for GMA calculation
drop_rows = []
for i in range(len(expdata_dt.index)):
    man_xsid1 = expdata_dt.iloc[i, 1]
    man_xsid2 = expdata_dt.iloc[i, 2]
    # we discard ratio measurements for the time being
    # or if it is neptunium (because no prior in the GMA database)
    if man_xsid2 != 0 or man_xsid1 == 2 or man_xsid2 == 2:
        drop_rows.append(i)

expdata_dt.drop(drop_rows, inplace=True)
expdata_dt.reset_index(inplace=True, drop=True)

# drop also the elements from the correlation matrix
keep_idcs = np.array([expdata_dt.iloc[:,0]-1]).T
cormat2d = cormat2d[keep_idcs, keep_idcs.T]

# determine the GMA prior indices based on Mannhart evaluation indices 
expdata_dt['prior_id1'] = 0
expdata_dt['prior_id2'] = 0
for i in range(len(expdata_dt.index)):
    man_xsid = expdata_dt.iloc[i, 1] 
    expdata_dt.loc[i,'prior_id1'] = man_gma_dic[man_xsid]


# create a list with the datasets
dataset_list = [] 

for currow in range(len(expdata_dt.index)):
    partialuncs = np.zeros((1,12))
    partialuncs[0,2] = expdata_dt.loc[currow,'unc']
    dataset = {
            'reference': expdata_dt.loc[currow, 'descr'][:32],
            'author': 'NA',
            'dataset_id': currow,
            'year': 2021,
            'obstype': label_to_MT('fission_average'),
            'reacids': np.array([expdata_dt.loc[currow, 'prior_id1']]),
            'energies': np.array([100]), # does not matter for obstype fission_average   
            'measurements': np.array([expdata_dt.loc[currow,'xs']]),
            'partialuncs': partialuncs,
            'cormat': None if currow < len(expdata_dt.index)-1 else cormat2d/100,
            'NNCOX': 0
            }
    dataset_list.append(dataset)

# create dummy datasets (as regularization or can be regarded as xs prior
# to make GLS problem well-posed)

dataset_list2 = []

for curxsid in prior_dt.xsid.unique(): 
    curdt = prior_dt[prior_dt.xsid == curxsid]
    curdt = curdt.sort_values(by='energy')
    cur_reac = curdt.iloc[0]['reac']
    cur_ens = np.array(curdt['energy'])
    cur_xs = np.array(curdt['xs'])
    partialuncs = np.zeros((len(cur_ens),12))
    partialuncs[:,2] = 10
    dataset = {
            'reference': 'dummy ' + cur_reac,
            'author': 'NA',
            'dataset_id': int(1000 + curxsid),
            'year': 2021,
            'obstype': label_to_MT('xs'),
            'reacids': np.array([curxsid]),
            'energies': cur_ens,  
            'measurements': cur_xs, 
            'partialuncs': partialuncs,
            'cormat': None,
            'NNCOX': 0
            }
    dataset_list2.append(dataset)


# make the datablock text
datasets_text = '\n'.join([generate_dataset_text(x) for x in dataset_list])
datablock_text = '\n'.join(['BLCK', datasets_text, 'EDBL'])

tmp = [['BLCK', generate_dataset_text(d), 'EDBL'] for d in dataset_list2] 
datablock2_text = '\n'.join([y for x in tmp for y in x])

# create the output file
MODAP = 20
mode_text = write_mode_setup(MODC, MOD2, 0, MODAP, MPPP)
ipp_text = write_IPP_setup(IPP[1], IPP[2], IPP[3], IPP[4], IPP[5], IPP[6], IPP[7], IPP[8])
fis_text = write_fission_text(fission_dt)
prior_text = write_prior_text(prior_dt)

gma_cont = '\n'.join([mode_text, ipp_text, fis_text, prior_text,
    datablock_text, datablock2_text, 'END*'])

# write the file
outdir = '../tmptest'
with open(os.path.join(outdir, 'data.gma'), 'w') as f:
    f.write(gma_cont)

# run GMAP on this file
osj = os.path.join
run_GMA_program(osj(outdir, 'data.gma'), osj(outdir, 'gma.res'), osj(outdir, 'plot.dta'))

# FURTHER INVESTIGATION

# do the GMAP calculation step-by-step
# db_dic = read_gma_database(osj(outdir, 'data.gma'))
# APR = db_dic['APR']
# datablock_list = db_dic['datablock_list']
# fisdata = db_dic['fisdata']
# LABL = db_dic['LABL']
# MPPP = db_dic['MPPP']
# IPP = db_dic['IPP']
# MODAP = db_dic['MODAP']
# 
# 
# from gmap_functions import link_prior_and_datablocks, add_compinfo_to_datablock
# 
# link_prior_and_datablocks(APR, datablock_list)
# for datablock in datablock_list:
#     add_compinfo_to_datablock(datablock, fisdata, APR, MPPP)
# 
# gauss = gls_update(datablock_list, APR)
# 

