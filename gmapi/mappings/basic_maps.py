import warnings
import numpy as np
from scipy.sparse import csr_matrix
from .helperfuns import return_matrix



def basic_propagate(x, y, xout, interp_type='lin-lin', zero_outside=False):
    """Propagate from one mesh to another one."""
    x = np.array(x)
    y = np.array(y)
    xout = np.array(xout)
    myord = np.argsort(x)
    x = x[myord]
    y = y[myord]
    if isinstance(interp_type, str):
        interp_type = np.full(len(x), interp_type, dtype='<U7')
    interp_type = np.array(interp_type)[myord]

    possible_interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
    if not np.all(np.isin(interp_type, possible_interp_types)):
        ValueError('Unspported interpolation type')

    # variable to store the result of this function
    final_yout = np.full(len(xout), 0., dtype=float)

    idcs2 = np.searchsorted(x, xout, side='left')
    idcs1 = idcs2 - 1
    # special case: where the values of xout are exactly on
    # the limits of the mesh in x
    limit_sel = np.logical_or(xout == x[0], xout == x[-1])
    not_limit_sel = np.logical_not(limit_sel)
    edge_idcs = idcs2[limit_sel]
    idcs1 = idcs1[not_limit_sel]
    idcs2 = idcs2[not_limit_sel]
    xout = xout[not_limit_sel]

    # Make sure that we actually have points that
    # need to be interpolated
    if len(idcs2) > 0:
        if np.any(idcs2 >= len(x)) and not zero_outside:
            raise ValueError('some value in xout larger than largest value in x')
        if np.any(idcs2 < 1) and not zero_outside:
            raise ValueError('some value in xout smaller than smallest value in x')

        inside_sel = np.logical_and(idcs2 < len(x), idcs2 >= 1)
        outside_sel = np.logical_not(inside_sel)
        idcs1 = idcs1[inside_sel]
        idcs2 = idcs2[inside_sel]
        xout = xout[inside_sel]

        x1 = x[idcs1]; x2 = x[idcs2]
        y1 = y[idcs1]; y2 = y[idcs2]
        xd = x2 - x1
        # transformed quantities
        log_x = np.log(x)
        log_y = np.log(y)
        log_x1 = log_x[idcs1]; log_x2 = log_x[idcs2]
        log_y1 = log_y[idcs1]; log_y2 = log_y[idcs2]
        log_xd = log_x2 - log_x1
        log_xout = np.log(xout)
        # results
        yout = {}
        yout['lin-lin'] = (y1*(x2-xout) + y2*(xout-x1)) / xd
        yout['log-lin'] = (y1*(log_x2-log_xout) + y2*(log_xout-log_x1)) / log_xd
        with warnings.catch_warnings():
            # We ignore a 'divide by zero in log warning due to y1 or y2
            # possibly containing non-positive values. As long as they
            # are not used, everything is fine. We check at the end of
            # this function explicitly for NaN values caused here.
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            yout['lin-log'] = np.exp((log_y1*(x2-xout) + log_y2*(xout-x1)) / xd)
            yout['log-log'] = np.exp((log_y1*(log_x2-log_xout) + log_y2*(log_xout-log_x1)) / log_xd)

        # fill final array
        interp = interp_type[idcs1]
        interp_yout = np.full(idcs1.shape, 0.)
        for curint in possible_interp_types:
            cursel = interp == curint
            interp_yout[cursel] = yout[curint][cursel]

        tmp = np.empty(len(inside_sel), dtype=float)
        tmp[inside_sel] = interp_yout
        tmp[outside_sel] = 0.
        final_yout[not_limit_sel] = tmp

    # add the edge points
    final_yout[limit_sel] = y[edge_idcs]

    if np.any(np.isnan(final_yout)):
        raise ValueError('NaN values encountered in interpolation result')
    return final_yout



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
        xd = x2 - x1
        # transformed quantities
        log_x = np.log(x)
        log_y = np.log(y)
        log_x1 = log_x[idcs1]; log_x2 = log_x[idcs2]
        log_y1 = log_y[idcs1]; log_y2 = log_y[idcs2]
        log_xd = log_x2 - log_x1
        log_xout = np.log(xout)
        # results
        coeffs1 = {}
        coeffs2 = {}
        # yout_linlin = (y1*(x2-xout) + y2*(xout-x1)) / xd
        coeffs1['lin-lin'] = (x2-xout) / xd
        coeffs2['lin-lin'] = (xout-x1) / xd
        # yout_loglin = (y1*(log_x2-log_xout) + y2*(log_xout-log_x1)) / log_xd
        coeffs1['log-lin'] = (log_x2-log_xout) / log_xd
        coeffs2['log-lin'] = (log_xout-log_x1) / log_xd

        with warnings.catch_warnings():
            # We ignore a 'divide by zero in log warning due to y1 or y2
            # possibly containing non-positive values. As long as they
            # are not used, everything is fine. We check at the end of
            # this function explicitly for NaN values in the sensitivity
            # matrix before it is returned.
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            log_yout_linlog = (log_y1*(x2-xout) + log_y2*(xout-x1)) / xd
            coeffs1['lin-log'] = np.exp(-log_y1 + np.log((x2-xout)/xd) + log_yout_linlog)
            coeffs2['lin-log'] = np.exp(-log_y2 + np.log((xout-x1)/xd) + log_yout_linlog)

            log_yout_loglog = (log_y1*(log_x2-log_xout) + log_y2*(log_xout-log_x1)) / log_xd
            coeffs1['log-log'] = np.exp(-log_y1 + np.log((log_x2-log_xout)/log_xd) + log_yout_loglog)
            coeffs2['log-log'] = np.exp(-log_y2 + np.log((log_xout-log_x1)/log_xd) + log_yout_loglog)

        interp = interp_type[idcs1]
        for curint in possible_interp_types:
            cursel = interp == curint
            # coeff1
            i.append(idcs1[cursel])
            j.append(idcs_out[cursel])
            c.append(coeffs1[curint][cursel])
            # coeff2
            i.append(idcs2[cursel])
            j.append(idcs_out[cursel])
            c.append(coeffs2[curint][cursel])

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



def basic_multiply_Sdic_rows(Sdic, rowfacts):
    """Multiply each row with a multiplication factor.

    Multiply each row of a sensitivity matrix returned
    by get_basic_sensmat function as a dictionary with a
    multiplication factor. The dictionary Sdic is changed
    in place.
    """
    Sdic['x'][::2] *= rowfacts
    Sdic['x'][1::2] *= rowfacts



def basic_extract_Sdic_coeffs(Sdic):
    """Extract partial derivatives from Sdic.

    Sdic represents the sensitivity matrix S
    to map from y-values given on mesh x to
    a mesh xout. Only two non-zero values
    are present in each row of S which are
    the partial derivative with respect to
    the y-value at the lower and upper limit
    of an interval respectively. This function
    facilitates the extraction of these
    partial derivatives.
    """
    df_da = Sdic['x'][::2].copy()
    df_db = Sdic['x'][1::2].copy()
    return [df_da, df_db]



def basic_product_propagate(xlist, ylist, xout, interplist, **kwargs):
    """Propagate the product of two basic maps."""
    prod = 1.
    for x, y, interp in zip(xlist, ylist, interplist):
        prod *= basic_propagate(x, y, xout, interp, **kwargs)
    return prod



def get_basic_product_sensmat(xlist, ylist, xout, interplist,
                              ret_mat=True, **kwargs):
    """Get a list of Jacobians for each factor in a product of basic maps."""
    proplist = []
    for x, y, interp in zip(xlist, ylist, interplist):
        proplist.append(basic_propagate(x, y, xout, interp, **kwargs))
    proparr = np.stack(proplist, axis=0)

    Slist = []
    for i, (x, y, interp) in enumerate(zip(xlist, ylist, interplist)):
        curSdic = get_basic_sensmat(x, y, xout, interp,
                                    ret_mat=False, **kwargs)
        sel = np.logical_and(np.min(x) <= xout, np.max(x) >= xout)
        curfacts = np.prod(proparr[:i,:], axis=0)
        if i+1 < proparr.shape[0]:
            curfacts *= np.prod(proparr[(i+1):,:], axis=0)
        basic_multiply_Sdic_rows(curSdic, curfacts[sel])
        Slist.append(curSdic)

    if ret_mat:
        for i in range(len(Slist)):
            curS = Slist[i]
            curS = return_matrix(curS['idcs1'], curS['idcs2'], curS['x'],
                                 dims=(len(xout), len(xlist[i])), how='csr')
            Slist[i] = curS

    return Slist



def propagate_fisavg(ens, vals, ensfis, valsfis):
    ord = np.argsort(ens)
    ens = ens[ord]
    vals = vals[ord]
    ordfis = np.argsort(ensfis)
    ensfis = ensfis[ordfis]
    valsfis = valsfis[ordfis]

    # we skip one element, because all energy meshes contain
    # as lowest energy 2.53e-8 MeV
    fidx = np.searchsorted(ensfis[1:], ens[1]) + 1

    lentmp = len(ensfis)
    ensfis = np.concatenate([ensfis, np.full(100, 0.)])
    valsfis = np.concatenate([valsfis, np.full(100, 0.)])

    uhypidx = fidx+len(ens)-1
    urealidx = min(fidx+len(ens)-1, len(ensfis))
    ulimidx2 = len(ens) - (uhypidx - urealidx)
    # TODO: This check fails for the 9Pu(n,f) cross section fission average
    #       because the energy 0.235 MeV is missing in 9Pu(n,f) but present
    #       in the fission spectrum
    # if not np.all(np.isclose(ens[1:ulimidx2], ensfis[fidx:urealidx], atol=0, rtol=1e-05)):
    #   raise ValueError('energies of excitation function and fission spectrum do not match')

    fl = 0.
    sfl = 0.
    for i in range(1, ulimidx2-1):
        fl = fl + valsfis[fidx-1+i]
        el1 = 0.5 * (ens[i-1] + ens[i])
        el2 = 0.5 * (ens[i] + ens[i+1])
        de1 = 0.5 * (ens[i] - el1)
        de2 = 0.5 * (el2 - ens[i])
        ss1 = 0.5 * (vals[i] + 0.5*(vals[i-1] + vals[i]))
        ss2 = 0.5 * (vals[i] + 0.5*(vals[i] + vals[i+1]))
        cssli = (ss1*de1 + ss2*de2) / (de1+de2)
        sfl = sfl + cssli*valsfis[fidx-1+i]

    fl = fl + valsfis[0] + valsfis[urealidx-1]
    sfl = sfl + valsfis[0]*vals[0] + valsfis[urealidx-1]*vals[-1]
    sfis = sfl / fl

    if not np.isclose(1., fl, atol=0, rtol=1e-4):
        print('fission normalization: ' + str(fl))
        raise ValueError('fission spectrum not normalized')

    return sfis



def get_sensmat_fisavg(ens, vals, ensfis, valsfis):
    """SACS Jacobian according to legacy code (wrong)."""
    ord = np.argsort(ens)
    ens = ens[ord]
    vals = vals[ord]
    ordfis = np.argsort(ensfis)
    ensfis = ensfis[ordfis]
    valsfis = valsfis[ordfis]

    # we skip one element, because all energy meshes contain
    # as lowest energy 2.53e-8 MeV
    fidx = np.searchsorted(ensfis[1:], ens[1]) + 1

    uhypidx = fidx+len(ens)-1
    urealidx = min(fidx+len(ens)-1, len(ensfis))
    ulimidx2 = len(ens) - (uhypidx - urealidx)
    # TODO: This check fails for the 9Pu(n,f) cross section fission average
    #       because the energy 0.235 MeV is missing in 9Pu(n,f) but present
    #       in the fission spectrum
    # if not np.all(np.isclose(ens[1:ulimidx2], ensfis[fidx:urealidx], atol=0, rtol=1e-05)):
    #     raise ValueError('energies of excitation function and fission spectrum do not match')

    lentmp = len(ensfis)
    ensfis = np.concatenate([ensfis, np.full(100, 0.)])
    valsfis = np.concatenate([valsfis, np.full(100, 0.)])

    sensvec = np.full(len(ens), 0., dtype=float)

    fl = 0.
    for i in range(0, len(ens)): 
        fl = fl + valsfis[fidx+i]
        if i == 0 or i == len(ens)-1:
            cssj = vals[i]
        else:
            el1 = 0.5 * (ens[i-1] + ens[i])
            el2 = 0.5 * (ens[i] + ens[i+1])
            de1 = 0.5 * (ens[i] - el1)
            de2 = 0.5 * (el2 - ens[i])
            ss1 = 0.5 * (vals[i] + 0.5*(vals[i-1] + vals[i]))
            ss2 = 0.5 * (vals[i] + 0.5*(vals[i] + vals[i+1]))
            cssj = (ss1*de1 + ss2*de2) / (de1+de2)

        sensvec[i] = valsfis[fidx+i-1] * cssj / vals[i]

    sensvec[ord] = sensvec.copy()
    return sensvec



def get_sensmat_fisavg_corrected(ens, vals, ensfis, valsfis):
    """Correct SACS Jacobian calculation."""
    ord = np.argsort(ens)
    ens = ens[ord]
    vals = vals[ord]
    ordfis = np.argsort(ensfis)
    ensfis = ensfis[ordfis]
    valsfis = valsfis[ordfis]

    # we skip one element, because all energy meshes contain
    # as lowest energy 2.53e-8 MeV
    fidx = np.searchsorted(ensfis[1:], ens[1]) + 1

    lentmp = len(ensfis)
    ensfis = np.concatenate([ensfis, np.full(100, 0.)])
    valsfis = np.concatenate([valsfis, np.full(100, 0.)])

    uhypidx = fidx+len(ens)-1
    urealidx = min(fidx+len(ens)-1, len(ensfis))
    ulimidx2 = len(ens) - (uhypidx - urealidx)
    # TODO: This check fails for the 9Pu(n,f) cross section fission average
    #       because the energy 0.235 MeV is missing in 9Pu(n,f) but present
    #       in the fission spectrum
    # if not np.all(np.isclose(ens[1:ulimidx2], ensfis[fidx:urealidx], atol=0, rtol=1e-05)):
    #   raise ValueError('energies of excitation function and fission spectrum do not match')

    fl = 0.
    sfl = 0.
    sensvec = np.full(len(vals), 0., dtype=float)
    for i in range(1, ulimidx2-1):
        fl = fl + valsfis[fidx-1+i]
        el1 = 0.5 * (ens[i-1] + ens[i])
        el2 = 0.5 * (ens[i] + ens[i+1])
        de1 = 0.5 * (ens[i] - el1)
        de2 = 0.5 * (el2 - ens[i])
        # For reference: the following lines for propagation
        # appear in propagate_fisavg and are the basis for
        # the sensitivity matrix calculation
        # ss1 = 0.5 * (vals[i] + 0.5*(vals[i-1] + vals[i]))
        # ss2 = 0.5 * (vals[i] + 0.5*(vals[i] + vals[i+1]))
        # cssli = (ss1*de1 + ss2*de2) / (de1+de2)
        # sfl = sfl + cssli*valsfis[fidx-1+i]
        ss1di = 0.75
        ss1dim1 = 0.25
        ss2di = 0.75
        ss2dip1 = 0.25
        coeff1 = de1/(de1+de2) * valsfis[fidx-1+i]
        coeff2 = de2/(de1+de2) * valsfis[fidx-1+i]
        sensvec[i] += ss1di * coeff1 + ss2di * coeff2
        sensvec[i-1] += ss1dim1 * coeff1
        sensvec[i+1] += ss2dip1 * coeff2

    fl = fl + valsfis[0] + valsfis[urealidx-1]
    # For reference: the renormalization of the propagated value
    # due to a potentially not normalized fission spectrum
    # sfl = sfl + valsfis[0]*vals[0] + valsfis[urealidx-1]*vals[-1]
    # sfis = sfl / fl
    sensvec[0] += valsfis[0]
    sensvec[-1] += valsfis[urealidx-1]
    sensvec /= fl

    if not np.isclose(1., fl, atol=0, rtol=1e-4):
        print('fission normalization: ' + str(fl))
        raise ValueError('fission spectrum not normalized')

    sensvec[ord] = sensvec.copy()
    return sensvec

