import numpy as np
from functools import lru_cache
from .basic_maps import (basic_propagate, get_basic_sensmat,
                         basic_extract_Sdic_coeffs,
                         basic_product_propagate,
                         get_basic_product_sensmats)
from .helperfuns import compute_romberg_integral



def basic_integral_propagate(x, y, interp_type='lin-lin',
                             zero_outside=False, **kwargs):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type, zero_outside)
    ret = compute_romberg_integral(xref, propfun, **kwargs)
    ret = np.array(ret, float)
    return ret



def get_basic_integral_sensmat(x, y, interp_type='lin-lin',
                               zero_outside=False, **kwargs):
    xref = x; yref = y
    def propfun(x):
        return basic_propagate(xref, yref, x, interp_type, zero_outside)
    def dpropfun(x):
        Sdic = get_basic_sensmat(xref, yref, x, interp_type,
                                 zero_outside, ret_mat=False)
        coeffs1, coeffs2 = basic_extract_Sdic_coeffs(Sdic)
        return (coeffs1, coeffs2)
    ret = compute_romberg_integral(xref, propfun, dfun=dpropfun, **kwargs)
    ret = np.array([ret])
    return ret



def basic_integral_of_product_propagate(xlist, ylist, interplist,
                                        zero_outside=False, **kwargs):
    def propfun(x):
        return basic_product_propagate(xlist, ylist, x,
                                       interplist, zero_outside)
    xref = np.unique(np.concatenate(xlist))
    ret = compute_romberg_integral(xref, propfun, **kwargs)
    ret = np.array([ret])
    return ret



def get_basic_integral_of_product_sensmats(xlist, ylist, interplist,
                                           zero_outside=False, **kwargs):

    def propfun(x):
        return basic_product_propagate(xlist, ylist, x,
                                       interplist, zero_outside)
    # The function sensfun is introduced in order to
    # cache the results of get_basic_product_sensmats called
    # in cur_dpropfun because it will be called several times
    # with identical arguments there.
    @lru_cache(maxsize=64)
    def sensfun(x):
        x = np.array(x)
        return get_basic_product_sensmats(xlist_ref, ylist_ref, x, interplist,
                                          zero_outside, ret_mat=False)
    def generate_dpropfun(i):
        def cur_dpropfun(x):
            # convert to tuple because ndarrays are
            # not hashable and lru_cache would fail
            tx = tuple(x)
            Sdic_list = sensfun(tx)
            Sdic = Sdic_list[i]
            coeffs1, coeffs2 = basic_extract_Sdic_coeffs(Sdic)
            return (coeffs1, coeffs2)
        return cur_dpropfun
    # determine a common mesh
    min_x_list = np.array([np.min(tx) for tx in xlist])
    max_x_list = np.array([np.max(tx) for tx in xlist])
    if not zero_outside:
        if (not np.all(min_x_list == min_x_list[0]) or
            not np.all(max_x_list == max_x_list[0])):
            raise ValueError('The x-limits of the meshes do not coincide')

    min_x = np.max([np.min(tx) for tx in xlist])
    max_x = np.min([np.max(tx) for tx in xlist])
    xref = np.unique(np.concatenate(xlist))
    xref = xref[np.logical_and(xref >= min_x, xref <= max_x)]
    # use the common mesh for each basic mapping
    xlist_ref = []
    ylist_ref = []
    interplist_ref = []
    for x, y, interp in zip(xlist, ylist, interplist):
        cury = basic_propagate(x, y, xref, interp, zero_outside)
        tmpord = np.argsort(x)
        idcs = np.searchsorted(x[tmpord], xref)
        idcs = tmpord[idcs]
        if isinstance(interp, str):
            interp = np.full(len(x), interp)
        curinterp = np.array(interp, copy=False)[idcs]
        xlist_ref.append(xref)
        ylist_ref.append(cury)
        interplist_ref.append(curinterp)
    # compute the sensitivity matrix for each inddividual contribution
    Smat_list = []
    for i in range(len(xlist)):
        cur_dfun = generate_dpropfun(i)
        cur_Smat =  compute_romberg_integral(xref, propfun,
                dfun=cur_dfun, **kwargs)
        curSref = get_basic_sensmat(xlist[i], ylist[i], xref, interplist[i],
                                    zero_outside, ret_mat=True)
        cur_Smat = np.array([cur_Smat]) @ curSref
        Smat_list.append(cur_Smat)

    return Smat_list

