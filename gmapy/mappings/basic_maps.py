import warnings
import numpy as np
from .helperfuns import return_matrix, return_matrix_new


def _interpolate_lin_lin(x1, y1, x2, y2, xout):
    return (y1*(x2-xout) + y2*(xout-x1)) / (x2-x1)


def _interpolate_log_lin(x1, y1, x2, y2, xout):
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)
    log_xd = log_x2 - log_x1
    log_xout = np.log(xout)
    return (y1*(log_x2-log_xout) + y2*(log_xout-log_x1)) / log_xd


def _interpolate_lin_log(x1, y1, x2, y2, xout):
    if y1 <= 0 or y2 <= 0:
        raise ValueError('attempted to take logarithm of negative value')
    xd = x2 - x1
    log_y1 = np.log(y1)
    log_y2 = np.log(y2)
    return np.exp((log_y1*(x2-xout) + log_y2*(xout-x1)) / xd)


def _interpolate_log_log(x1, y1, x2, y2, xout):
    log_x1 = np.log(x1)
    log_x2 = np.log(x2)
    log_y1 = np.log(y1)
    log_y2 = np.log(y2)
    log_xd = log_x2 - log_x1
    log_xout = np.log(xout)
    return np.exp((log_y1*(log_x2-log_xout) + log_y2*(log_xout-log_x1)) / log_xd)


def _interpolate1d(x1, y1, x2, y2, xout, interp_type, zero_outside):
    if (xout < x1 or xout > x2):
        if zero_outside:
            return 0.
        else:
            raise ValueError(f'xout={xout} outside interval '
                             f'from x1={x1} to x2={x2}')
    if interp_type == 'lin-lin':
        interpfun = _interpolate_lin_lin
    elif interp_type == 'lin-log':
        interpfun = _interpolate_lin_log
    elif interp_type == 'log-lin':
        interpfun = _interpolate_log_lin
    elif interp_type == 'log-log':
        interpfun = _interpolate_log_log
    else:
        raise TypeError('unsupported interpolation type {interp_type}')
    return interpfun(x1, y1, x2, y2, xout)


def _findintervals(xmesh, xvals):
    n = len(xmesh)
    tmp = np.searchsorted(xmesh, xvals, side='left')
    tmp = [t if t > 0 else 1 for t in tmp]
    tmp = [t if t < n else n-1 for t in tmp]
    idcs2 = np.array(tmp)
    idcs1 = idcs2 - 1
    return idcs1, idcs2


def basic_propagate(x, y, xout, interp_type='lin-lin', zero_outside=False):
    """Propagate from one mesh to another one."""
    if len(x) != len(y):
        raise ValueError('x and y must be of same length')

    xout = np.array(xout)
    x = np.array(x)
    y = np.array(y)
    if len(x) == 1:
        if not np.all(xout == x):
            raise ValueError('xout must match x for a single mesh point')
        return np.full(len(xout), y, dtype=float)

    myord = np.argsort(x)
    x = x[myord]
    y = y[myord]
    if isinstance(interp_type, str):
        interp_type = np.full(len(x), interp_type, dtype='<U7')
    interp_type = np.array(interp_type)[myord]

    idcs1, idcs2 = _findintervals(x, xout)
    x1v = x[idcs1]
    x2v = x[idcs2]
    y1v = y[idcs1]
    y2v = y[idcs2]
    ityp = interp_type[idcs1]

    interp_yout = np.array(tuple(
            _interpolate1d(x1, y1, x2, y2, xout, inttype, zero_outside)
            for x1, y1, x2, y2, xout, inttype
            in zip(x1v, y1v, x2v, y2v, xout, ityp)
        ))
    return interp_yout


def get_basic_sensmat(x, y, xout, interp_type='lin-lin',
                      zero_outside=False, ret_mat=True):
    """Compute sensitivity matrix for basic mappings."""
    orig_x = np.array(x)
    x = orig_x.copy()
    x = np.array(x)
    y = np.array(y)
    xout = np.array(xout)
    myord = np.argsort(x)
    x = x[myord]
    y = y[myord]
    orig_len_x = len(x)
    orig_len_xout = len(xout)
    if isinstance(interp_type, str):
        interp_type = np.full(len(x), interp_type, dtype='<U7')
    interp_type = np.array(interp_type)[myord]

    possible_interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
    if not np.all(np.isin(interp_type, possible_interp_types)):
        ValueError('Unspported interpolation type')

    idcs2 = np.searchsorted(x, xout, side='left')
    idcs1 = idcs2 - 1
    idcs_out = np.arange(len(xout))
    # special case: where the values of xout are exactly on
    # the limits of the mesh in x
    limit_sel = np.logical_or(xout == x[0], xout == x[-1])
    not_limit_sel = np.logical_not(limit_sel)
    edge_idcs = idcs2[limit_sel]
    edge_idcs_out = idcs_out[limit_sel]
    idcs2 = idcs2[not_limit_sel]
    idcs1 = idcs1[not_limit_sel]
    xout = xout[not_limit_sel]
    idcs_out = idcs_out[not_limit_sel]

    # initialize the index lists and
    # value list of te final sensitivity matrix
    # i ... column indices
    # j ... row indices of final sensitivity matrix
    i = []; j = []; c = []

    # Make sure that we actually have points that
    # need to be interpolated
    if len(idcs2) > 0:
        if np.any(idcs2 >= len(x)) and not zero_outside:
            raise ValueError('some value in xout larger than largest value in x')
        if np.any(idcs2 < 1) and not zero_outside:
            raise ValueError('some value in xout smaller than smallest value in x')

        inside_sel = np.logical_and(idcs2 < len(x), idcs2 >= 1)
        idcs1 = idcs1[inside_sel]
        idcs2 = idcs2[inside_sel]
        idcs_out = idcs_out[inside_sel]
        xout = xout[inside_sel]

        x1 = x[idcs1]; x2 = x[idcs2]
        y1 = y[idcs1]; y2 = y[idcs2]
        interp = interp_type[idcs1]
        for curint in possible_interp_types:
            cursel = interp == curint
            if not any(cursel):
                continue
            coeffs1 = {}
            coeffs2 = {}
            x1s = x1[cursel]
            x2s = x2[cursel]
            y1s = y1[cursel]
            y2s = y2[cursel]
            xouts = xout[cursel]
            # yout_linlin = (y1*(x2-xout) + y2*(xout-x1)) / xd
            if curint == 'lin-lin':
                xd = x2s - x1s
                coeffs1 = (x2s-xouts) / xd
                coeffs2 = (xouts-x1s) / xd
            # yout_loglin = (y1*(log_x2-log_xout) + y2*(log_xout-log_x1)) / log_xd
            elif curint == 'log-lin':
                log_x1 = np.log(x1s)
                log_x2 = np.log(x2s)
                log_xd = log_x2 - log_x1
                log_xout = np.log(xouts)
                coeffs1 = (log_x2-log_xout) / log_xd
                coeffs2 = (log_xout-log_x1) / log_xd
            elif curint == 'lin-log':
                xd = x2s - x1s
                log_y1 = np.log(y1s)
                log_y2 = np.log(y2s)
                log_yout_linlog = (log_y1*(x2s-xouts) + log_y2*(xouts-x1s)) / xd
                coeffs1 = np.exp(-log_y1 + np.log((x2s-xouts)/xd) + log_yout_linlog)
                coeffs2 = np.exp(-log_y2 + np.log((xouts-x1s)/xd) + log_yout_linlog)
            elif curint == 'log-log':
                log_x1 = np.log(x1s)
                log_x2 = np.log(x2s)
                log_xd = log_x2 - log_x1
                log_xout = np.log(xouts)
                log_y1 = np.log(y1s)
                log_y2 = np.log(y2s)
                log_yout_loglog = (log_y1*(log_x2-log_xout) + log_y2*(log_xout-log_x1)) / log_xd
                coeffs1 = np.exp(-log_y1 + np.log((log_x2-log_xout)/log_xd) + log_yout_loglog)
                coeffs2 = np.exp(-log_y2 + np.log((log_xout-log_x1)/log_xd) + log_yout_loglog)
            else:
                raise TypeError(f'invalid interpolation scheme "{curint}"')

            # coeff1
            i.append(idcs1[cursel])
            j.append(idcs_out[cursel])
            c.append(coeffs1)
            # coeff2
            i.append(idcs2[cursel])
            j.append(idcs_out[cursel])
            c.append(coeffs2)

    # deal with sensitivies for values at the mesh edges
    i.append(np.concatenate([edge_idcs, edge_idcs]))
    j.append(np.concatenate([edge_idcs_out, edge_idcs_out]))
    c.append(np.concatenate([np.full(edge_idcs.shape, 1.),
                             np.full(edge_idcs.shape, 0.)]))

    # better than model casting:
    # flatten the list of arrays to a single array
    i = myord[np.concatenate(i)]
    j = np.concatenate(j)
    c = np.concatenate(c)
    # we sort these arrays according to j to ensure
    # that the coefficients in one row of the sensitivity
    # matrix are consecutive elements in the array c.
    # We have two coefficients per row irrespective of
    # the interpolation law.
    # The function basic_multiply_Sdic_rows relies on
    # this assumption.
    perm = np.argsort(j)
    i = i[perm]
    j = j[perm]
    c = c[perm]
    # We further do swaps of the variables associated
    # with a row j if x[j_k] > x[j_(k+1)], to be sure
    # that the coefficient associated with the lower x-value
    # comes first. The function basic_extract_Sdic_coeffs
    # relies on this structure.
    i_tmp = i.copy()
    c_tmp = c.copy()
    should_swap = orig_x[i[::2]] > orig_x[i[1::2]]
    i[::2] = np.where(should_swap, i_tmp[1::2], i_tmp[::2])
    i[1::2] = np.where(should_swap, i_tmp[::2], i_tmp[1::2])
    c[::2] = np.where(should_swap, c_tmp[1::2], c_tmp[::2])
    c[1::2] = np.where(should_swap, c_tmp[::2], c_tmp[1::2])

    if np.any(np.isnan(c)):
        raise ValueError('NaN values encountered in Jacobian matrix')

    return return_matrix(i, j, c, dims=(orig_len_xout, orig_len_x),
                         how='csr' if ret_mat else 'dic')


def basic_product_propagate(xlist, ylist, xout, interplist,
                            zero_outside=False, **kwargs):
    """Propagate the product of two basic maps."""
    if len(xlist) != len(ylist) or len(ylist) != len(interplist):
        raise IndexError('xlist, ylist and interplist must have ' +
                         'the same number of elements')
    prod = 1.
    for x, y, interp in zip(xlist, ylist, interplist):
        prod *= basic_propagate(x, y, xout, interp, zero_outside, **kwargs)
    return prod


def get_basic_product_sensmats(xlist, ylist, xout, interplist,
                               zero_outside=False, ret_mat=True, **kwargs):
    """Get a list of Jacobians for each factor in a product of basic maps."""
    if len(xlist) != len(ylist) or len(ylist) != len(interplist):
        raise IndexError('xlist, ylist and interplist must have ' +
                         'the same number of elements')
    proplist = []
    for x, y, interp in zip(xlist, ylist, interplist):
        proplist.append(basic_propagate(x, y, xout, interp,
                        zero_outside, **kwargs))
    proparr = np.stack(proplist, axis=0)

    Slist = []
    for i, (x, y, interp) in enumerate(zip(xlist, ylist, interplist)):
        curS = get_basic_sensmat(x, y, xout, interp, zero_outside,
                                 ret_mat=True, **kwargs)
        curfacts = np.prod(proparr[:i,:], axis=0)
        if i+1 < proparr.shape[0]:
            curfacts *= np.prod(proparr[(i+1):,:], axis=0)
        curS = curS.multiply(curfacts.reshape(-1, 1))
        Slist.append(curS)

    for i in range(len(Slist)):
        how = 'csr' if ret_mat else 'dic'
        Slist[i] = return_matrix_new(Slist[i], how)
    return Slist
