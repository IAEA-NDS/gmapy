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
    ret = compmap.propagate(dt, refvals)
    return ret


def gmapi_jacobian(dt, refvals):
    refvals = np.array(refvals)
    compmap = CompoundMap()
    ret = compmap.jacobian(dt, refvals)
    return ret

