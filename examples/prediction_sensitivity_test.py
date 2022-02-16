from database_reading import read_gma_database
from GMAP import run_GMA_program

from inference_new import (extract_predictions, extract_sensitivity_matrix,
        extract_covariance_matrix, extract_measurements, extract_prior_values)

from inference import (link_prior_and_datablocks, add_compinfo_to_datablock,
        fill_AA_AM_COV, update_all_estimates)

from database_writing_utils import (extract_prior_datatable,
        extract_experiment_datatable, extract_fission_datatable)

from gmap_snippets import label_to_MT

import pandas as pd
import numpy as np
import os


# load the database and reduce it to the reactions in the Mannhart evaluation
orig_dbfile = 'tests/test_002/input/data.gma'
db = read_gma_database(orig_dbfile)
datablock_list = db['datablock_list']
fisdata = db['fisdata']
APR = db['APR']
MPPP = db['MPPP']

# prepare the datastructures
link_prior_and_datablocks(APR, datablock_list)

# calculate sensitivity information and correlation matrices
for datablock in datablock_list:
    add_compinfo_to_datablock(datablock, fisdata, APR, MPPP)

# examples of extracting information
priorvals = extract_prior_values(APR)
preds = extract_predictions(datablock_list)
meas = extract_measurements(datablock_list)
predvec = preds[:, np.newaxis]
S = extract_sensitivity_matrix(datablock_list, APR)
covmat = extract_covariance_matrix(datablock_list)


# functions to calculate predictions and analytic gradient (sensitivity matrix)

def get_predictions(priorvals):
    update_all_estimates(APR, priorvals)
    for data in datablock_list:
        fill_AA_AM_COV(data, fisdata, APR)
    preds = extract_predictions(datablock_list)
    return preds

def get_sensmat(priorvals):
    update_all_estimates(APR, priorvals)
    for data in datablock_list:
        fill_AA_AM_COV(data, fisdata, APR)
    sensmat = extract_sensitivity_matrix(datablock_list, APR)
    return sensmat


# function for numerical differentiation
def numdiff(func, x0, h=1e-6, idcs=None):
    y0 = func(x0)
    numinp = len(x0)
    numout = len(y0)
    jac = np.zeros((numout, len(idcs)))
    for i, idx in enumerate(idcs):
        xdelta = np.zeros(numinp)
        xdelta[idx] = h
        senscol = (-func(x0+2*xdelta) + 8*func(x0+xdelta) - 8*func(x0-xdelta) + func(x0-2*xdelta)) / (12*h)
        jac[:, i] = senscol
    return jac
        

# randomly check the sensitivies to hundred prior values
np.random.seed(44)
# np.random.seed(43)
paridcs = np.sort(np.unique(np.random.randint(0,len(priorvals),100)))
import time
start_time = time.time()
numS = numdiff(get_predictions, priorvals, idcs=paridcs)
anaS = get_sensmat(priorvals)
end_time = time.time()
print('time taken %s' % (end_time-start_time))


# analyze the results

redanaS = anaS[:,paridcs].toarray()
reldiff = np.abs(redanaS - numS)/ np.maximum(np.abs(redanaS),np.abs(numS)) 
absdiff = np.abs((redanaS - numS))


from matplotlib import pyplot as plt
idcs = np.where(redanaS != 0)  # 1e-6
fig, ax = plt.subplots(figsize=(10,7))
ax.hist(reldiff[idcs], bins=[0,1e-7,1e-6, 1e-4])
ax.hist(reldiff[idcs], bins=[1e-4, 1e-2, 1e-1, 1e-1, 1])
plt.xscale('log')
plt.xlabel('relative difference between numeric and analytical sensitivity element')
plt.ylabel('number of elements in sensitivity matrix')
plt.show()

# figure out the underlying reactions 

preds = get_predictions(priorvals)
redpriorvals = priorvals[paridcs]

relredanaS = redanaS / np.outer(pres, redpriorvals)
relnumS = numS / np.outer(preds, redpriorvals)
reldiff = np.abs(relredanaS - relnumS)/ np.maximum(np.abs(relredanaS),np.abs(relnumS)) 

#strange_idcs = np.where(np.logical_and(reldiff > 1e-3, relredanaS > 1e-3))   
strange_idcs = np.where(np.logical_and(reldiff > 1e-6, redanaS > 0))   
strange_idcs
redanaS[strange_idcs[0],strange_idcs[1]]

prior_dt = extract_prior_datatable(db['APR'])
fission_dt = extract_fission_datatable(db['fisdata'])
expdata_dt = extract_experiment_datatable(datablock_list, prior_dt)

dt1 = prior_dt.loc[strange_idcs[1]]
dt2 = expdata_dt.loc[strange_idcs[0]]
dt1.reset_index(inplace=True, drop=True)
dt2.reset_index(inplace=True, drop=True)
dt = pd.concat([dt1,dt2], axis=1)
dt['abssensel1'] = redanaS[strange_idcs[0], strange_idcs[1]]
dt['abssensel2'] = numS[strange_idcs[0], strange_idcs[1]]
dt['relsensel1'] = relredanaS[strange_idcs[0], strange_idcs[1]]
dt['relsensel2'] = relnumS[strange_idcs[0], strange_idcs[1]]
dt['reldiff'] = dt['abssensel1'] / dt['abssensel2'] - 1 
dt

np.where(np.abs(preds - meas)/meas > 1)

# prepare the datastructures
link_prior_and_datablocks(APR, datablock_list)
# calculate sensitivity information and correlation matrices
for datablock in datablock_list:
    add_compinfo_to_datablock(datablock, fisdata, APR, MPPP)

