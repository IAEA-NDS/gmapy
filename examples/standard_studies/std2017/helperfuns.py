import pandas as pd
import numpy as np


def translate_to_absreacs(reacs):
    absreacs = []
    for reac in reacs:
        if reac.startswith('MT:2-'):
            absreac = 'MT:1-' + reac[5:]
        elif reac.startswith('MT:4-'):
            absreac = 'MT:3-' + reac[5:]
        elif reac.startswith('MT:8-'):
            absreac = 'MT:5-' + reac[5:]
        elif reac.startswith('MT:9-'):
            absreac = 'MT:7-' + reac[5:]
        else:
            absreac = reac
        absreacs.append(absreac)
    return np.array(absreacs)


def renormalize_data(priortable, exptable, priorvals, expvals):
    renorm_vals = np.array(expvals) 
    expids = exptable.loc[exptable.REAC.str.match('MT:[2489]-'), 'NODE'].unique()
    for expid in expids:
        norm_node = 'norm_' + expid[4:]
        norm_index = priortable.index[priortable.NODE == norm_node]
        if len(norm_index) != 1:
            raise IndexError(
                f'exactly one normalization error must be present for {expid}'
            )
        norm_index = norm_index[0]
        norm_fact = priorvals[norm_index]
        exp_idcs = exptable.index[exptable.NODE == expid]
        renorm_vals[exp_idcs] /= norm_fact
    return renorm_vals


import sys
sys.path.append('../../..')
from gmapy.data_management.object_utils import load_objects

priortable, exptable, is_adj = load_objects('../../tensorflow/example-005/output/01_model_preparation_output.pkl', 'priortable', 'exptable', 'is_adj')
optres, = load_objects('../../tensorflow/example-005/output/02_parameter_optimization_output.pkl', 'optres')
red_priortable = priortable[is_adj].reset_index(drop=True)

renorm_data = renormalize_data(red_priortable, exptable, optres.position.numpy(), exptable.DATA)
absreacs = translate_to_absreacs(exptable.REAC)

exptable['ABSREAC'] = absreacs
exptable['RENORM_DATA'] = renorm_data












