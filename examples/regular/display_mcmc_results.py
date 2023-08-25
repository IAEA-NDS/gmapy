import sys
sys.path.append('..')
from gmapy.mcmc_inference import load_mcmc_result
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.energy_dependent_usu_map import attach_endep_usu_df
from gmapy.mappings.priortools import (
    prepare_prior_and_likelihood_quantities,
    create_propagate_source_mask
)
from gmapy.posterior_usu import PosteriorUSU
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mcmc_inference import compute_effective_sample_size
import scipy.sparse as sps
import pandas as pd


gmadb = GMADatabase('../legacy-tests/test_004/input/data.gma')
covmat = gmadb.get_covmat()
gmadb.evaluate(correct_ppp=True)
postcov = gmadb.get_postcov()
dt = gmadb.get_datatable()
dt.rename(columns={'POST': 'GLS_POST'}, inplace=True)
dt['GLS_POSTUNC'] = np.sqrt(postcov.diagonal())

# this code must be changed depending on USU uncertainty assumptions taken
cdt = dt.copy()
cdt = attach_endep_usu_df(cdt, ['MT:1-R1:9'], [0, 7, 12.5, 20], [0.1]*4)
cdt = attach_endep_usu_df(cdt, ['MT:3-R1:9-R2:8'], [0, 7, 12.5, 20], [0.1]*4)
cdt = attach_endep_usu_df(cdt, ['MT:4-R1:9-R2:8'], [0, 7, 12.5, 20], [0.1]*4)
usu_uncs = cdt.loc[cdt.NODE.str.match('endep_'), 'UNC'].to_numpy()
usu_cov = sps.diags(usu_uncs**2)
ccovmat = sps.block_diag([covmat, usu_cov], format='csr')

q = prepare_prior_and_likelihood_quantities(cdt, ccovmat)
priordt = q['priortable']
priorvals = q['priorvals']
priorcov = q['priorcov']
expvals = q['expvals']
expcov = q['expcov']
exptable = q['exptable']

unc_dt = priordt.loc[priordt.NODE.str.match('endep_usu'), ['NODE', 'REAC', 'ENERGY']]
unc_idcs = unc_dt.index.to_numpy()
unc_group_assoc = unc_dt.REAC + '_EN:' +  unc_dt.ENERGY.astype(str)

m = CompoundMap(cdt, reduce=True)
source_mask = create_propagate_source_mask(priordt)
postdist = PosteriorUSU(
    priorvals, priorcov, m, expvals, expcov,
    relative_exp_errors=True, source_mask=source_mask,
    unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
)

# NEEDS to be done better!
usu_group_dt = pd.DataFrame({'usu_group': postdist._groups})


# quick timeing
import time
st = time.time()
for i in range(100):
    u = np.random.uniform(size=6600)
    x = priorvals * np.random.uniform(size=priorvals.shape, low=0.9, high=1.1)
    r = m.vec_times_jacobian(u, x)


ft = time.time()
print(ft-st)



# save for plotting in R
# mh_res = load_mcmc_result('mh_res_000_', '.pkl', 'results-008e')
# np.savetxt('mini_csewg_2023_stuff/samples_8e.csv', mh_res['samples'], delimiter=',')
# np.savetxt('mini_csewg_2023_stuff/logprobhist_8e.csv', mh_res['logprob_hist'], delimiter=',')
# cdt.to_csv('mini_csewg_2023_stuff/datatable_8e.csv')
# priordt.to_csv('mini_csewg_2023_stuff/priortable_8e.csv')
# usu_group_dt.to_csv('mini_csewg_2023_stuff/usu_group_datatable_8e.csv')
# exptable.to_csv('mini_csewg_2023_stuff/exptable_8e.csv')
# np.savetxt('mini_csewg_2023_stuff/gls_postcov.csv', postcov.toarray(), delimiter=',')

# script to prepare quantities for plotting in R ends here

exp_dt = dt[dt.NODE.str.match('exp_')]
all_reacs = exp_dt.REAC.drop_duplicates()
usu_reacs = all_reacs[all_reacs.str.match('MT:[1357].*-R.:9')].tolist()

cdt = dt.copy()
usu_reacs.append('MT:2-R1:9')
cdt = attach_endep_usu_df(cdt, usu_reacs, [0, 7, 12.5, 20], [0.1]*4)




mh_res1 = load_mcmc_result('mh_res_000_', '.pkl', 'results-008e')
mh_res2 = load_mcmc_result('mh_res_001_', '.pkl', 'results-008e')
smpl1 = mh_res1['samples']
smpl2 = mh_res2['samples']
num_burn = 20000
red_smpl1 = smpl1[:, num_burn:]
red_smpl2 = smpl2[:, num_burn:]

plt.plot(np.arange(len(mh_res1['logprob_hist'])), mh_res1['logprob_hist'])
plt.plot(np.arange(len(mh_res2['logprob_hist'])), mh_res2['logprob_hist'])
plt.show()


curidx = -9

plt.plot(np.arange(smpl1[:,num_burn:].shape[1]), red_smpl1[curidx,:], alpha=0.4)
plt.plot(np.arange(smpl2[:,num_burn:].shape[1]), red_smpl2[curidx, :], alpha=0.4)
plt.show()

binbreaks = np.linspace(0, 0.1, 20)
plt.hist(red_smpl1[curidx,:], alpha=0.4, bins=binbreaks, color='r')
plt.hist(red_smpl2[curidx,:], alpha=0.4, bins=binbreaks)
plt.show()

compute_effective_sample_size(red_smpl1[curidx,:])
compute_effective_sample_size(red_smpl2[curidx,:])

meanvec1 = np.mean(red_smpl1, axis=1)
meanvec2 = np.mean(red_smpl2, axis=1)
stdvec1 = np.std(red_smpl1, axis=1)
stdvec2 = np.std(red_smpl2, axis=1)

meanvec1[curidx]
meanvec2[curidx]

# study distributions of experimental quantities

expsmpl1 = m.propagate(red_smpl1[:-len(usu_group_dt), :10]) 
expsmpl2 = m.propagate(meanvec2[:-len(usu_group_dt)]) 

thin_idcs = np.linspace(1, red_smpl1.shape[1]-1, 2000).astype('int')
thin_red_smpl1 = red_smpl1[:, thin_idcs]
expsmpl1 = np.empty((len(exptable), thin_red_smpl1.shape[1]), dtype=float)
for i in range(expsmpl1.shape[1]):
    expsmpl1[:,i] = m.propagate(thin_red_smpl1[:-len(usu_group_dt), i]) 

expmean1 = np.mean(expsmpl1, axis=1)
expstd1 = np.std(expsmpl1, axis=1)

exptable['RELUNC'] = exptable['UNC'] / exptable['DATA']
exptable['MC_POST'] = expmean1
exptable['MC_POSTUNC'] = expstd1
exptable['MC_RELPOSTUNC'] = expstd1 / expmean1
exptable['GLS_RELPOSTUNC'] = exptable['GLS_POSTUNC'] / exptable['GLS_POST']

# show exp datatable with posterior
exp_idcs = exptable.index[exptable.REAC.str.match('MT:6-R1:9')]
exptable.loc[exp_idcs]

# show correlation matrix
cormat = expcov.copy()
cormat /= np.sqrt(expcov.diagonal().reshape(-1, 1) * expcov.diagonal().reshape(1, -1))
red_cormat = cormat[np.ix_(exp_idcs, exp_idcs)]
red_cormat[np.abs(red_cormat) < 1e-3] = 0
red_cormat

# plot some experimental histograms

plt.hist(expsmpl1[2603, :], bins=20)
plt.show()

# plot some cross sections

cur_node = 'xsid_9'
reac_idcs = priordt.index[(priordt.NODE == cur_node) & (priordt.ENERGY > 0.15) & (priordt.ENERGY < 20)]
priordt.loc[reac_idcs]
priordt['MC_POST'] = meanvec1[:-len(usu_group_dt)]
priordt['MC_POSTUNC'] = stdvec1[:-len(usu_group_dt)]
reac_energies = priordt.loc[reac_idcs, 'ENERGY'].to_numpy(copy=True)
reac_uncs = priordt.loc[reac_idcs, 'MC_POSTUNC']
reac_xs = priordt.loc[reac_idcs, 'MC_POST']
reac_uncs / reac_xs
plt.errorbar(reac_energies, reac_xs, reac_uncs)
plt.xlim(0.15, 20)
plt.title(cur_node)
plt.show()
plt.title(cur_node)
plt.plot(reac_energies, reac_uncs / reac_xs)
plt.show()



reac_dt = cdt.loc[cdt.NODE == 'xsid_9']
idcs = reac_dt.index
sel_dt = cdt.loc[idcs]
sel_dt['POST1'] = meanvec1[idcs]
sel_dt['POST2'] = meanvec2[idcs] 
sel_dt['RELUNC1'] = stdvec1[idcs] / meanvec1[idcs]
sel_dt['RELUNC2'] = stdvec2[idcs] / meanvec2[idcs] 
sel_dt[(sel_dt.ENERGY > 1) & (sel_dt.ENERGY < 20)]

idx = -1
postdist._groups[idx]
s1 = red_smpl1[idx, :]
s2 = red_smpl2[idx, :]
bins = np.arange(0, max(np.max(s1), np.max(s2)), 0.01)
plt.hist(red_smpl1[idx, :], bins=bins, alpha=0.5)
plt.hist(red_smpl2[idx, :], bins=bins, alpha=0.5)
plt.axvline(x=meanvec1[idx])
plt.axvline(x=meanvec2[idx])
plt.title(postdist._groups[idx])
# plt.yscale('log')
# plt.ylim(0, 500)
plt.show()

meanvec1[idx]
meanvec2[idx]
stdvec1[idx]
stdvec2[idx]


# print reaction xsid mapping
dt.loc[dt.NODE.str.match('xsid_'), ['NODE', 'REAC', 'DESCR']].drop_duplicates()

# analyze number of experiments in reactions
exp_dt = dt[dt.NODE.str.match('exp_')]
red_exp_dt = exp_dt[exp_dt.REAC.str.match('MT:[1357]-.*-?R.:9')]
red_exp_dt[['NODE', 'REAC']].drop_duplicates().groupby('REAC').count()

red_exp_dt = exp_dt[exp_dt.REAC.str.match('MT:[123456789]-.*-?R.:9')]
red_exp_dt[['NODE', 'REAC']].drop_duplicates().groupby('REAC').count()

exp_dt.loc[exp_dt.REAC.str.match('MT:[1234567]-.*-?R.:9'), 'REAC'].drop_duplicates()

exp_dt.loc[exp_dt.REAC.str.match('MT:[123456789]-.*-?R.:4'), 'REAC'].drop_duplicates()

# plot a channel
# red_exp_dt = exp_dt[exp_dt.REAC == 'MT:9-R1:9-R2:3-R3:4']
red_exp_dt = exp_dt[exp_dt.REAC == 'MT:2-R1:9']
# red_exp_dt = exp_dt[exp_dt.REAC == 'MT:1-R1:9']
# red_exp_dt = exp_dt[exp_dt.REAC == 'MT:4-R1:7-R2:9']
# red_exp_dt = exp_dt[exp_dt.REAC == 'MT:1-R1:7']
red_exp_dt[['NODE', 'REAC']].drop_duplicates().groupby('REAC').count()

ax = sns.scatterplot(data=red_exp_dt, x='ENERGY', y='DATA', hue='NODE') 
ax.errorbar(red_exp_dt.ENERGY, red_exp_dt.DATA, red_exp_dt.UNC, fmt='', ls='none', color='gray')
plt.xscale('log') # for MT:1-R1:9
# plt.ylim(1, 3)
plt.legend([],  [], frameon=False)
plt.show()


######################################################################
#   EXCLUSION STUDIES
######################################################################

from copy import deepcopy
ref_gmadb = GMADatabase('../legacy-tests/test_004/input/data.gma')
ref_dt = ref_gmadb.get_datatable()
coupled_reacs = ref_dt.loc[ref_dt.REAC.str.match('.*-R.:9'), 'REAC'].drop_duplicates().to_numpy()

gmadb = deepcopy(ref_gmadb) 
dt = gmadb.get_datatable()
remove_reacs = coupled_reacs[3:] # we still are around (slightly below) 1%
# remove_reacs = coupled_reacs[0:3]  # we get 4%
# remove_reacs = coupled_reacs[[0, 2]] # we get around 1.1% # remove MT:3-R1:9-R2:8
# remove_reacs = coupled_reacs[[1, 2]] # we get 2%  # remove MT:1-R1:9 
# remove_reacs = coupled_reacs[[0, 1]]  # we get 4% # remove MT:4-R1:9-R2:8

remove_idcs = dt.index[(dt.NODE.str.match('exp_')) & (dt.REAC.isin(remove_reacs))]
covmat = gmadb.get_covmat()
covmat[remove_idcs, remove_idcs] += 1e3
gmadb.set_covmat(covmat)
try:
    gmadb.evaluate()
    dt = gmadb.get_datatable()
    idcs = dt.index[dt.NODE.str.match('xsid_9$')]
    uncs = gmadb.get_postcov(idcs=idcs, unc_only=True)
    dt.loc[idcs, 'POSTUNC'] = uncs
    dt.loc[idcs, 'POSTRELUNC'] = uncs / dt.loc[idcs, 'POST']
    red_dt = dt.loc[idcs]
    red_dt.loc[(red_dt.ENERGY > 0.15) & (red_dt.ENERGY < 20)]
except:
    print('error occurred')

# red_dt.to_csv('mini_csewg_2023_stuff/result_only_MT-1-R1-9-MT-4-R1-9-R2-8-MT-3-R1-9-R2-8.csv')

#                    counts
# MT:1-R1:9              17
# MT:3-R1:9-R2:8         15
# MT:4-R1:9-R2:8          7


# plot a channel
# red_exp_dt = exp_dt[(exp_dt.REAC == 'MT:1-R1:9') & (exp_dt.ENERGY > 1e-6)]
# red_exp_dt = exp_dt[(exp_dt.REAC == 'MT:3-R1:9-R2:8') & (exp_dt.ENERGY > 1e-6)]
red_exp_dt = exp_dt[(exp_dt.REAC == 'MT:4-R1:9-R2:8') & (exp_dt.ENERGY > 1e-6)]
red_exp_dt[['NODE', 'REAC']].drop_duplicates().groupby('REAC').count()
ax = sns.scatterplot(data=red_exp_dt, x='ENERGY', y='DATA', hue='NODE') 
ax.errorbar(red_exp_dt.ENERGY, red_exp_dt.DATA, red_exp_dt.UNC, fmt='', ls='none', color='gray')
# plt.xscale('log') # for MT:1-R1:9
# plt.ylim(0, 3)
plt.legend([],  [], frameon=False)
plt.show()



