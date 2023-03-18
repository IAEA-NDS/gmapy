import numpy as np
from functools import lru_cache
from .basic_maps import (basic_propagate, get_basic_sensmat,
                         basic_product_propagate,
                         get_basic_product_sensmats)
from .helperfuns import compute_romberg_integral


def extract_partial_derivatives(S, xmesh, xout):
    S = S.tocsr()
    S.eliminate_zeros()
    z = S.tolil()
    n = len(z.rows)
    minx = min(xmesh)
    maxx = max(xmesh)
    coeff1 = np.full(S.shape[0], 0., dtype=float)
    coeff2 = np.full(S.shape[0], 0., dtype=float)
    for row, idcs, data in zip(range(n), z.rows, z.data):
        if len(idcs) == 2:
            idx1 = idcs[0]
            idx2 = idcs[1]
            x1 = xmesh[idx1]
            x2 = xmesh[idx2]
            if x1 < x2:
                coeff1[row] = data[0]
                coeff2[row] = data[1]
            else:
                coeff1[row] = data[1]
                coeff2[row] = data[0]
        elif len(idcs) == 1:
            idx = idcs[0]
            x = xmesh[idx]
            if x < minx or x > maxx:
                coeff1[row] = 0.
                coeff2[row] = 0.
            elif x == minx:
                coeff1[row] = data[0]
                coeff2[row] = 0.
            else:
                coeff1[row] = 0.
                coeff2[row] = data[0]
        else:
            raise ValueError('this must not happen')
    return coeff1, coeff2


def _integrate_lin_lin(x, y):
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    part1 = x2*y1 - x1*y2
    part2 = 0.5*(x2*x2 - x1*x1) * (y2-y1)/(x2-x1)
    return np.sum(part1 + part2)


def _integrate_lin_lin_sensmat(x, y):
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    sens = np.zeros(len(x), dtype=float)
    h = 0.5 * (x2*x2 - x1*x1) / (x2-x1)
    sens1 = x2 - h
    sens2 = h - x1
    sens = np.zeros(len(x), dtype=float)
    sens[:-1] += sens1
    sens[1:] += sens2
    return sens


def basic_integral_propagate(x, y, interp_type='lin-lin',
                             zero_outside=False, **kwargs):
    if np.all(interp_type == 'lin-lin'):
        p = np.argsort(x)
        x = np.array(x)[p]
        y = np.array(y)[p]
        ret = _integrate_lin_lin(x, y)
        return np.array(ret, float)
    else:
        xref = x; yref = y
        def propfun(x):
            return basic_propagate(xref, yref, x, interp_type, zero_outside)
        ret = compute_romberg_integral(xref, propfun, **kwargs)
        ret = np.array(ret, float)
        return ret


def get_basic_integral_sensmat(x, y, interp_type='lin-lin',
                               zero_outside=False, **kwargs):
    if np.all(interp_type == 'lin-lin'):
        p = np.argsort(x)
        x = np.array(x)[p]
        y = np.array(y)[p]
        pret = _integrate_lin_lin_sensmat(x, y)
        ret = np.empty((1, len(x)), dtype=float)
        ret[0, p] = pret
        return ret
    else:
        sortord = np.argsort(x)
        xref = np.array(x)[sortord]
        yref = np.array(y)[sortord]
        if type(interp_type) != str:
            interp_type = np.array(interp_type)[sortord]

        def propfun(x):
            return basic_propagate(xref, yref, x, interp_type, zero_outside)

        def dpropfun(x):
            S = get_basic_sensmat(xref, yref, x, interp_type,
                                  zero_outside)
            coeffs1, coeffs2 = extract_partial_derivatives(S, xref, x)
            return (coeffs1, coeffs2)

        ordered_ret = compute_romberg_integral(
            xref, propfun, dfun=dpropfun, **kwargs
        )
        orig_ret = np.empty(len(ordered_ret))
        orig_ret[sortord] = ordered_ret
        ret = np.array([orig_ret])
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
                                          zero_outside)
    def generate_dpropfun(i):
        def cur_dpropfun(x):
            # convert to tuple because ndarrays are
            # not hashable and lru_cache would fail
            tx = tuple(x)
            S_list = sensfun(tx)
            S = S_list[i]
            coeffs1, coeffs2 = extract_partial_derivatives(S, xlist_ref[i], x)
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
                                    zero_outside)
        cur_Smat = np.array([cur_Smat]) @ curSref
        Smat_list.append(cur_Smat)

    return Smat_list
