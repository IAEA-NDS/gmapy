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

dummy_en = 100

prior_dt = pd.DataFrame.from_records([
        {'xsid': 1, 'reac': 'U235F', 'energy': dummy_en, 'xs': 1205}, 
        {'xsid': 2, 'reac': 'NP237F', 'energy': dummy_en, 'xs': 1332},
        {'xsid': 3, 'reac': 'U238F', 'energy': dummy_en, 'xs': 318},
        {'xsid': 4, 'reac': 'PU239F', 'energy': dummy_en, 'xs': 1807}
        ])

expdata_dt_list = []
cormat_list = []

# first datablock

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

# bundle together
expdata_dt_list.append(expdata_dt)
cormat_list.append(cormat2d)


# EXPERIMENT : ADAMOV
expdata2 = [  
 [1, 3, 1, 2.741E-1, 1.6600,  'U238F/U235F',    '*1.0       ADAMOV'],
 [2, 2, 1, 1.139E+0, 1.8000,  'NP237F/U235F',   '*1.0       ADAMOV'],
 [3, 4, 1, 1.475E+0, 1.5000,  'PU239F/U235F',   '*1.0       ADAMOV']
]
expdata2_dt = pd.DataFrame.from_records(expdata2, columns = ['idx', 'reac_id1', 'reac_id2', 'xs', 'unc', 'reacstr', 'descr'])

cormat2d2 = np.array(
 [[100,  29, 29],
  [ 29, 100, 29],
  [ 29,  29,100]]
)

# bundle together
expdata_dt_list.append(expdata2_dt)
cormat_list.append(cormat2d2)

# EXPERIMENT: NBS-VNC + SPIEGEL
expdata3 = [
 [1, 2, 1,  1.035E+0, 8.4800, 'NP237F/U235F',  '*2.0      NBS-VNC'],
 [2, 3, 1,  2.491E-1, 5.2200, 'U238F/U235F',   '*2.0      NBS-VNC']
]
expdata3_dt = pd.DataFrame.from_records(expdata3, columns = ['idx', 'reac_id1', 'reac_id2', 'xs', 'unc', 'reacstr', 'descr'])

cormat2d3 = np.array([
[100, 33],
[ 33,100]
])

# bundle together
expdata_dt_list.append(expdata3_dt)
cormat_list.append(cormat2d3)


# create a list with the datasets
datablock_text_list = []

for idx in range(len(expdata_dt_list)):
    basenum = (idx+1)*1000
    expdata_dt = expdata_dt_list[idx]
    cormat2d = cormat_list[idx]
    dataset_list = [] 
    for currow in range(len(expdata_dt.index)):
        reac_id1 = expdata_dt.loc[currow, 'reac_id1']
        reac_id2 = expdata_dt.loc[currow, 'reac_id2']
        partialuncs = np.zeros((1,12))
        partialuncs[0,2] = expdata_dt.loc[currow,'unc']
        dataset = {
            'reference': expdata_dt.loc[currow, 'descr'][:32],
            'author': 'NA',
            'dataset_id': basenum+currow,
            'year': 2021,
            'obstype': label_to_MT('xs') if reac_id2 == 0 else label_to_MT('xs_ratio'), 
            'reacids': np.array([reac_id1] if reac_id2==0 else [reac_id1, reac_id2]),
            'energies': np.array([dummy_en]),
            'measurements': np.array([expdata_dt.loc[currow,'xs']]),
            'partialuncs': partialuncs,
            'cormat': None if currow < len(expdata_dt.index)-1 else cormat2d/100,
            'NNCOX': 0
	    }
        dataset_list.append(dataset)
    # make the datablock text
    datasets_text = '\n'.join([generate_dataset_text(x) for x in dataset_list])
    datablock_text = '\n'.join(['BLCK', datasets_text, 'EDBL'])
    datablock_text_list.append(datablock_text)

all_datablock_texts = '\n'.join(datablock_text_list)


MODC = 3
MOD2 = 0
MPPP = 0
MODAP = 20
IPP = [1, 1, 1, 0, 0, 1, 0, 1]

# prepare the datafile
mode_text = write_mode_setup(MODC, MOD2, 0, MODAP, MPPP)
ipp_text = write_IPP_setup(IPP[0], IPP[1], IPP[2], IPP[3], IPP[4], IPP[5], IPP[6], IPP[7])
prior_text = write_prior_text(prior_dt)

gma_cont = '\n'.join([mode_text, ipp_text, prior_text,
    all_datablock_texts, 'END*'])

# write the file
outdir = 'tmptest'
with open(os.path.join(outdir, 'data.gma'), 'w') as f:
    f.write(gma_cont)

# run GMAP on this file
osj = os.path.join
run_GMA_program(osj(outdir, 'data.gma'), osj(outdir, 'gma.res'), osj(outdir, 'plot.dta'))






