import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from gmapy.mappings.compound_map import CompoundMap
import numpy as np
import pandas as pd


def gmapi_propagate(dt, refvals):
    refvals = np.array(refvals)
    compmap = CompoundMap()
    ret = compmap.propagate(refvals, dt)
    return ret


def gmapi_jacobian(dt, refvals):
    refvals = np.array(refvals)
    compmap = CompoundMap()
    ret = compmap.jacobian(refvals, dt)
    return ret

