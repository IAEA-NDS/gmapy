import numpy as np
from .basic_maps import basic_propagate, get_basic_sensmat
from .helperfuns import compute_romberg_integral


def basic_integral_propagate(x, y, interp_type='lin-lin',
                             maxord=16, rtol=1e-8):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type) 
    ret = compute_romberg_integral(xref, propfun, maxord=maxord, rtol=rtol)
    ret = np.array(ret, float)
    return ret

