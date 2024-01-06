import sys
sys.path.append('../../..')
from gmapy.data_management.object_utils import load_objects
from gmapy.mcmc_inference import compute_effective_sample_size
import numpy as np
import matplotlib.pyplot as plt



priortable, red_usu_df, is_adj, exptable, restrimap = \
    load_objects('example-009-ext-usu/output/01_model_preparation_output.pkl',
                 'priortable', 'red_usu_df', 'is_adj', 'exptable', 'restrimap')
chain, = load_objects('example-009-ext-usu/output/03_mcmc_sampling_output.pkl',
                      'chain')




for i in range(1, len(red_usu_df)+1):
    cur_idx = -i
    ridcs = red_usu_df.index[cur_idx]
    currow = red_usu_df.loc[ridcs,]
    compute_effective_sample_size(np.abs(chain[:, cur_idx]))
    plt.title(f'{currow.REAC} --- {currow.ENERGY} MeV')
    plt.hist(np.abs(chain[:, cur_idx]), bins=30)
    plt.show()


postmeans = np.mean(chain, axis=0)
covmat = np.cov(chain.numpy().T)
postuncs = np.sqrt(covmat.diagonal())


red_priortable = priortable.loc[is_adj,].copy()
red_priortable.loc[:, 'POST'] = postmeans[:-len(red_usu_df)] 
red_priortable.loc[:, 'POSTUNC'] = postuncs[:-len(red_usu_df)] 
red_priortable.loc[:, 'RELPOSTUNC'] = red_priortable.loc[:, 'POSTUNC'] / red_priortable.loc[:, 'POST'] 

plotdt = red_priortable.query('REAC == "MT:1-R1:9"')
explodt = exptable.query('REAC == "MT:1-R1:9"')

ylim = (1.5, 4)
xlim = (1.5e-4, 0.02)
fig, axes = plt.subplots(nrows=2)
axes[0].plot(plotdt.ENERGY, plotdt.POST)
axes[0].plot(plotdt.ENERGY, plotdt.POST-plotdt.POSTUNC, c='gray')
axes[0].plot(plotdt.ENERGY, plotdt.POST+plotdt.POSTUNC, c='gray')
axes[0].errorbar(explodt.ENERGY, explodt.DATA, yerr=explodt.UNC, fmt='o')
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
axes[1].plot(plotdt.ENERGY, plotdt.RELPOSTUNC)
axes[1].set_xlim(xlim)
axes[1].set_ylim([0,0.07])
plt.show()


# compare with experimental data
propvals = restrimap.propagate(postmeans[:-len(red_usu_df)])
exptable.loc[:, 'PROPVAL'] = propvals.numpy()
