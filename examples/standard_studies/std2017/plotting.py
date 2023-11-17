import sys
sys.path.append('../../../')
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)
from gmapy.mcmc_inference import compute_effective_sample_size
from gmapy.mappings.tf.compound_map_tf import CompoundMap
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.gmap import run_gmap_simplified
from gmapy.legacy.legacy_gmap import run_gmap

thisdir = '/home/gschnabel/Seafile/OmegaSpace/development/codeproj/gitrepos/gmapy/examples/tensorflow/example-005-std2017/output_backup/'
# thisdir = '/home/gschnabel/Seafile/OmegaSpace/development/codeproj/gitrepos/gmapy/examples/tensorflow/example-009/output/'

# load the priortable
priortable, exptable, expcov, is_adj = load_objects(thisdir + '01_model_preparation_output.pkl', 'priortable', 'exptable', 'expcov', 'is_adj')
red_priortable = priortable.loc[is_adj].reset_index(drop=True)

tmp = red_priortable[['DESCR', 'REAC']].drop_duplicates()
tmp = tmp[~tmp.DESCR.isna()]
mtdic = {x: y for x, y in zip(tmp.DESCR, tmp.REAC)}

# 1) load all the standard data from the standards 2017
#    to create a pseudo-experimental dataset
descr_list = []
outdt_list = []

tmp = pd.read_csv('std17-003_Li_006.txt', comment='#', index_col=None, sep=r'\s+')
descr = '6Li(n,a)'
descr_list.append(descr)
outdt_list.append(pd.DataFrame({'NODE': 'exp_1000', 'REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp.CS.to_numpy(), 'UNC': tmp.DCS.to_numpy(), 'DESCR': descr}))

tmp = pd.read_csv('std17-005_B_010.txt', comment='#', index_col=None, sep=r'\s+')
descr = '10B(n,a1)'
descr_list.append(descr)
outdt_list.append(pd.DataFrame({'NODE': 'exp_1001','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp.CS.to_numpy(), 'UNC': tmp.DCS.to_numpy(), 'DESCR': descr}))
descr = '10B(n,a)'
outdt_list.append(pd.DataFrame({'NODE': 'exp_1002','REAC': 'MT:5-R1:3-R2:4', 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS.1'].to_numpy(), 'UNC': tmp['DCS.1'].to_numpy(), 'DESCR': descr}))

tmp = pd.read_csv('std17-079_Au_197.txt', comment='#', index_col=None, sep=r'\s+')
descr = 'Au(n,g)'
descr_list.append(descr)
outdt_list.append(pd.DataFrame({'NODE': 'exp_1003','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))

tmp = pd.read_csv('std17-092_U_235.txt', comment='#', index_col=None, sep=r'\s+')
descr = 'U5(n,f)'
descr_list.append(descr)
outdt_list.append(pd.DataFrame({'NODE': 'exp_1004','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))

tmp = pd.read_csv('std17-092_U_238.txt', comment='#', index_col=None, sep=r'\s+')
descr = 'U8(n,f)'
descr_list.append(descr)
outdt_list.append(pd.DataFrame({'NODE': 'exp_1005','REAC': mtdic[descr], 'ENERGY': tmp.En.to_numpy(), 'DATA': tmp['CS'].to_numpy(), 'UNC': tmp['DCS'].to_numpy(), 'DESCR': descr}))


# now for the thermal neutron constants
tmp = pd.read_csv('Standards2017_TNC.txt', comment='#', index_col=0, sep=r'\s+')

isos = tmp.columns
isos = isos[~isos.str.endswith('UNC')]
quants = tmp.index
exp_cnt = 1005
for iso in isos:
    for q in quants:
        exp_cnt += 1
        descr = f'{q}-{iso}'
        if descr == 'SF-U5':
            descr = 'U5(n,f)'
        elif descr == 'SF-PU9':
            descr = 'PU9(n,f)'
        print(descr)
        # create dataframe
        outdt_list.append(pd.DataFrame({
            'NODE': 'exp_' + str(exp_cnt), 'REAC': [mtdic[descr]],
            'ENERGY': 2.53e-8, 'DATA': tmp.loc[q, iso],
            'UNC': tmp.loc[q, iso + '-UNC'], 'DESCR': descr
        }))

outdt = pd.concat(outdt_list, ignore_index=True)
outdt = outdt.sort_values(['REAC', 'ENERGY'], ignore_index=True)

# create the mapping object
compmap = CompoundMap((priortable, outdt), reduce=True)
restrmap = RestrictedMap(len(is_adj), compmap.propagate, compmap.jacobian,
        fixed_params=priortable.loc[~is_adj, 'PRIOR'].to_numpy(copy=True),
        fixed_params_idcs=np.where(~is_adj)[0]
)
restrmap_prop = tf.function(restrmap.propagate)

# load a chain
chain, = load_objects(thisdir + '03_mcmc_sampling_output.pkl', 'chain')
optres, = load_objects(thisdir + '02_parameter_optimization_output.pkl', 'optres')

# load the maxlike estimate

prop_chain = np.zeros((chain.shape[0], len(outdt)), dtype=np.float64)
for idx in range(chain.shape[0]):
    curchain = chain[idx, :len(red_priortable)]
    prop_chain[idx, :] = restrmap_prop(curchain)

eval_mcmc_raw = np.mean(chain, axis=0)
eval_maxlike_raw = optres.position.numpy()

# calculate the mean values
eval_mcmc = np.mean(prop_chain, axis=0)
eval_mcmc_unc = np.std(prop_chain, axis=0)
eval_maxlike = restrmap_prop(optres.position[:len(red_priortable)])

# some plotting
outdt['PRED_ML'] = eval_maxlike
outdt['PRED_MCMC'] = eval_mcmc
outdt['PRED_MCMC_UNC'] = eval_mcmc_unc
outdt['PRED_ML_vs_STD2017'] = eval_maxlike / outdt['DATA'] - 1.
outdt['PRED_MCMC_vs_STD2017'] = eval_mcmc / outdt['DATA'] - 1.

# look at the thermal constants
selcrit = (outdt.DESCR.str.match('S.-') | 
    ((outdt.ENERGY == 2.53e-8) & outdt.DESCR.isin(('U5(n,f)', 'U8(n,f)'))))
outdt[selcrit]


gmap_res = run_gmap_simplified(
    dbfile='../../../legacy-tests/test_002/input/data.gma',
    num_iter=8
)
gmap_restable = gmap_res['table']
red_gmap_restable = gmap_restable[gmap_restable.NODE.str.match('xsid_|norm_')].reset_index(drop=True)

gmap_res = restrmap_prop(red_gmap_restable.POST.to_numpy())
outdt['GMAP'] = gmap_res
# outdt['GMAP_vs_STD2017'] = gmap_res / outdt['DATA']


legacy_gmap_res = run_gmap(
    dbfile='../../../legacy-tests/test_002/input/data.gma',
    fix_sacs_jacobian=False,
    legacy_integration=True,
    remove_dummy=False, fix_ppp_bug=False, 
    num_iter=5
)
leg_gmap_restable = legacy_gmap_res['table']
red_leg_gmap_restable = leg_gmap_restable[leg_gmap_restable.NODE.str.match('xsid_|norm_')].reset_index(drop=True)
leg_gmap_res = restrmap_prop(red_leg_gmap_restable.POST.to_numpy())
outdt['GMAP_leg'] = leg_gmap_res
outdt['GMAP_leg/STD2017'] = leg_gmap_res / outdt['DATA'] - 1.
outdt['GMAP/GMAP_leg'] = gmap_res / leg_gmap_res - 1.

outdt['GMAP/STD2017'] = gmap_res / outdt['DATA'] - 1.


# explore the SS-PU1 case
# SS-PU1
selidx = outdt[(outdt.DESCR == 'SS-PU1')].index
curchain = prop_chain[:, selidx]
compute_effective_sample_size(curchain)
plt.hist(curchain, bins=50)
plt.show()

# SS-PU9
curchain = prop_chain[:, 479]
compute_effective_sample_size(curchain)
plt.hist(curchain, bins=50)
plt.show()

np.where(curchain > 40)[0]
testres = restrmap_prop(chain[10763, :])
outdt['TESTRES'] = testres



# show the energy-dependent quantities
for curdescr in descr_list:
    cdt = outdt[outdt.DESCR == curdescr]
    # plt.plot(cdt.ENERGY, cdt.PRED_ML)
    # plt.plot(cdt.ENERGY, cdt.PRED_MCMC)
    plt.plot(cdt.ENERGY, cdt.PRED_ML / cdt.DATA, color='green', label='ML')
    plt.plot(cdt.ENERGY, cdt.PRED_MCMC / cdt.DATA, color='blue', label='MCMC')
    plt.plot(cdt.ENERGY, cdt.GMAP / cdt.DATA, color='red', label='GMAP')
    # plt.plot(cdt.ENERGY, cdt.TESTRES / cdt.DATA, color='black', label='GMAP')
    plt.axvline(x=30)
    plt.title(curdescr)
    plt.legend(loc='upper right')
    plt.show()


outdt[outdt.DESCR=='U8(n,f)']

# U8(n,f) at 192 MeV
selidx = outdt[(outdt.DESCR == 'U8(n,f)') & (outdt.ENERGY==192.)].index
curchain = prop_chain[:, selidx]
compute_effective_sample_size(curchain)
plt.hist(curchain, bins=50)
plt.show()

# show what type of data we that refers to U5(n,f) 
exp_with_U5 = exptable[exptable.REAC.str.match('.*-R.:8')]
result = exp_with_U5.groupby('REAC').agg(MaxEn=('ENERGY', np.max), NumPts=('ENERGY', len))
result = result.sort_values('MaxEn')
result

# rescale expdata using normalization
exptable['RENORM_DATA'] = exptable['DATA'].copy()
normdt = gmap_restable[gmap_restable.NODE.str.startswith('norm_')]
normdt = red_priortable[red_priortable.NODE.str.startswith('norm_')].copy()
# for MCMC result
rescale_vec = np.mean(chain, axis=0)[normdt.index]
# for maxlike result
# rescale_vec = optres.position.numpy()[normdt.index]
normdt['POST'] = rescale_vec 

for idx, row in normdt.iterrows():
    expid = row.NODE[5:]
    expname = 'exp_' + str(expid)
    if not (exptable.NODE == expname).any():
        raise ValueError()
    sel = exptable.NODE==expname
    exptable.loc[sel, 'RENORM_DATA'] = exptable.loc[sel, 'DATA'] / row['POST']

curreac = 'MT:2-R1:8'
# curreac = 'MT:3-R1:9-R2:8'
# curreac = 'MT:3-R1:10-R2:8'
rtbl = exptable[exptable.REAC==curreac]
grouped = rtbl.groupby('NODE')
for gname, gdata in grouped:
    # plt.scatter(gdata['ENERGY'], gdata['RENORM_DATA'], s=5)
    plt.errorbar(gdata['ENERGY'], gdata['RENORM_DATA'], yerr=gdata['UNC'],
            fmt='o', markersize=3) 
    if curreac == 'MT:2-R1:8':
        plt.ylim(1, 3)
        tmp = outdt[outdt.REAC=='MT:1-R1:8']
        plt.plot(tmp.ENERGY, tmp.PRED_MCMC, color='black')
        plt.errorbar(tmp.ENERGY, tmp.PRED_MCMC, yerr=tmp.PRED_MCMC_UNC, color='black')
        plt.plot(tmp.ENERGY, tmp.PRED_ML, color='black', linestyle='--')
    elif curreac == 'MT:3-R1:9-R2:8':
        pass
    elif curreac == 'MT:3-R1:10-R2:8':
        plt.ylim(0.7, 1.1)

plt.show()


# marginal distribution along a non-axis direction
d = eval_mcmc_raw - eval_maxlike_raw
d = d / np.sqrt(np.sum(d**2))
project_chain = np.squeeze(chain.numpy() @ d.reshape(-1, 1))
# get mean and mode
tmp_mean = np.mean(project_chain)
hist, bin_edges = np.histogram(project_chain, bins=70)
tmp_mode = bin_edges[np.argmax(hist)]
# plot the histogram
hist = plt.hist(project_chain, bins=50)
plt.axvline(x=np.mean(project_chain), color='black')
plt.axvline(x=tmp_mode, color='blue')
plt.show()
# difference of expectation versus mean
(tmp_mean - tmp_mode) / tmp_mean 




import plotly.express as px
fig = px.scatter(rtbl, x='ENERGY', y='DATA', color='NODE')
fig.show()


exptable[exptable.NODE=='exp_1028']

plt.hist(chain.numpy()[:, idx1])
plt.show()
plt.hist(chain.numpy()[:, idx2])
plt.show()

# show the difference of the normalization factor
idx1 = red_priortable[red_priortable.NODE=='norm_1028'].index
optres.position.numpy()[idx1]
np.mean(chain.numpy()[:, idx1])

idx2 = red_priortable[red_priortable.NODE=='norm_1003'].index
optres.position.numpy()[idx2]
np.mean(chain.numpy()[:, idx2])

# does it return to the other solution?
