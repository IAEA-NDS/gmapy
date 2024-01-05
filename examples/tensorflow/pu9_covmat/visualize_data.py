import sys
sys.path.append('../../..')
import matplotlib
# matplotlib.use('Qtagg')
import matplotlib.pyplot as plt
import numpy as np
from gmapy.data_management.database_IO import read_gma_database
from gmapy.mappings.priortools import remove_dummy_datasets
from gmapy.data_management.tablefuns import create_experiment_table
from gmapy.data_management.uncfuns import create_experimental_covmat
import os
import re

db_path = '../../../tests/testdata/data_and_sacs.json'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

expcov = create_experimental_covmat(db['datablock_list'])

exptable = create_experiment_table(db['datablock_list'])
exptable['UNC'] = np.sqrt(expcov.diagonal())

def reac_trans(reacstr):
    reacdic = {8: 'U5(n,f)', 9: 'PU9(n,f)', 10: 'U8(n,f)'}
    typedic = {1: 'abs', 2: 'shape', 3: 'ratio', 4: 'ratio shape'}
    toks = reacstr.split('-')
    toks = tuple(s.split(':') for s in toks)
    try:
        obstype = typedic[int(toks[0][1])]
        reacstrs = tuple(reacdic[int(s[1])] for s in toks[1:])
        return f'{obstype} : {" - ".join(reacstrs)}'
    except:
        return reacstr

energy_ranges = ((0, 0.01), (0.01, 0.1), (0.1, 1), (1, 30), (30, 200)) # , (0.01, 0.1), (0.1, 1), (1, 30), (30, 200))
reac_grouped = exptable.groupby('REAC')
for reac_group, reacdt in reac_grouped:
    if not re.search('R?:(8|9|10)($|-)', reac_group):
        continue
    fig, axs = plt.subplots(len(energy_ranges), squeeze=False)
    fig.set_size_inches(10, 12)
    reacstr = reac_trans(reac_group)
    fig.suptitle(reacstr)
    for idx, en in enumerate(energy_ranges):
        curdt = reacdt.query(f'ENERGY >= {en[0]} & ENERGY <= {en[1]}')
        expdata_grouped = curdt.groupby('NODE')
        for expid, expdt in expdata_grouped:
            pass
            axs[idx,0].errorbar(expdt.ENERGY, expdt.DATA, yerr=expdt.UNC, fmt='o')
            axs[idx,0].set_xlim([en[0], en[1]])
            axs[idx,0].set_title(f'{en[0]} - {en[1]} MeV')
    fig.tight_layout(pad=3.0)
    curfpath = os.path.join('plots', reac_group.replace(':','-'))
    #plt.show()
    plt.savefig(curfpath)
