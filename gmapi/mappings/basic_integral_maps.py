import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
                         basic_extract_Sdic_coeffs,
                         basic_product_propagate)
from .helperfuns import compute_romberg_integral


def basic_integral_propagate(x, y, interp_type='lin-lin', **kwargs):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type)
    ret = compute_romberg_integral(xref, propfun, **kwargs)
    ret = np.array(ret, float)
    return ret


def get_basic_integral_sensmat(x, y, interp_type='lin-lin', **kwargs):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type)
    def dpropfun(x):
        Sdic = get_basic_sensmat(xref, yref, x, interp_type, ret_mat=False)
        coeffs1, coeffs2 = basic_extract_Sdic_coeffs(Sdic)
        return (coeffs1, coeffs2)
    ret = compute_romberg_integral(xref, propfun, dfun=dpropfun, **kwargs)
    ret = np.array([ret])
    return ret


def basic_integral_of_product_propagate(xlist, ylist, interplist, **kwargs):
    def propfun(x):
        return basic_product_propagate(xlist, ylist, x,
                                       interplist, zero_outside=True)
    xref = np.unique(np.concatenate(xlist))
    ret = compute_romberg_integral(xref, propfun, **kwargs)
    ret = np.array([ret])
    return ret

