import sys
sys.path.append('../../..')
from gmapy.data_management.object_utils import load_objects
from gmapy.mcmc_inference import compute_effective_sample_size
import numpy as np
import matplotlib.pyplot as plt



priortable, red_usu_df, is_adj = load_objects('example-009-ext-usu/output/01_model_preparation_output.pkl',
                                  'priortable', 'red_usu_df', 'is_adj')
chain, = load_objects('example-009-ext-usu/output/03_mcmc_sampling_output.pkl',
                      'chain')



red_priortable = priortable.loc[is_adj,].copy()
red_priortable.shape
red_usu_df.shape


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


red_priortable.loc[:, 'POST'] = postmeans[:-len(red_usu_df)] 
red_priortable.loc[:, 'POSTUNC'] = postuncs[:-len(red_usu_df)] 
red_priortable.loc[:, 'RELPOSTUNC'] = red_priortable.loc[:, 'POSTUNC'] / red_priortable.loc[:, 'POST'] 

plotdt = red_priortable.query('REAC == "MT:1-R1:9"')


fig, axes = plt.subplots(nrows=2)
axes[0].plot(plotdt.ENERGY, plotdt.POST)
axes[0].set_xlim([1,30])
axes[0].set_ylim([1,5])
axes[1].plot(plotdt.ENERGY, plotdt.RELPOSTUNC)
axes[1].set_xlim([0,30])
axes[1].set_ylim([0,0.05])
plt.show()
