import sys
sys.path.append('../..')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gmapy.data_management.object_utils import (
    load_objects, save_objects
)

# show the prior data

nodes = pd.unique(exptable.NODE)
groupidx = np.arange(len(nodes))
mapping = {k: f for k, f in zip(nodes, groupidx)}
exptable['exp_fact'] = exptable.NODE.map(mapping)


grouped = exptable.groupby('NODE')
ax = None
for name, group in grouped:
    ax = group.plot(x='ENERGY', y='DATA', yerr='UNC', ax=ax, alpha=0.7, legend=False)

plt.show()


# show the results of the MCMC
chain, = load_objects('output/03_mcmc_sampling_output.pkl', 'chain')
priortable, exptable, usu_df = load_objects(
    'output/01_model_preparation_output.pkl',
    'priortable', 'exptable', 'usu_df'
)

plt.hist(np.abs(chain[:,11]), bins=30, density=True)
plt.show()


from gmapy.mcmc_inference import compute_effective_sample_size
compute_effective_sample_size(np.abs(chain[:,11]))



