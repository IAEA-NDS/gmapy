import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
                         basic_extract_Sdic_coeffs)
from .helperfuns import compute_romberg_integral


def basic_integral_propagate(x, y, interp_type='lin-lin',
                             maxord=16, rtol=1e-8):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type)
    ret = compute_romberg_integral(xref, propfun, maxord=maxord, rtol=rtol)
    ret = np.array(ret, float)
    return ret


def get_basic_integral_sensmat(x, y, interp_type='lin-lin',
                               maxord=16, rtol=1e-8):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type)
    def dpropfun(x):
        Sdic = get_basic_sensmat(xref, yref, x, interp_type, ret_mat=False)
        coeffs1, coeffs2 = basic_extract_Sdic_coeffs(Sdic)
        return (coeffs1, coeffs2)
    ret = compute_romberg_integral(xref, propfun, dfun=dpropfun,
                                   maxord=maxord, rtol=rtol)
    ret = np.array([ret])
    return ret

