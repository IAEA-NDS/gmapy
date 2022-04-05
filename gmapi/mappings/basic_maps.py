import warnings
import numpy as np
from scipy.sparse import csr_matrix
from .helperfuns import return_matrix



def basic_propagate(x, y, xout, interp_type='lin-lin'):
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
        if np.any(idcs2 >= len(x)):
            raise ValueError('some value in xout larger than largest value in x')
        if np.any(idcs2 < 1):
            raise ValueError('some value in xout smaller than smallest value in x')

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

        final_yout[not_limit_sel] = interp_yout

    # add the edge points
    final_yout[limit_sel] = y[edge_idcs]

    if np.any(np.isnan(final_yout)):
        raise ValueError('NaN values encountered in interpolation result')
    return final_yout



def get_basic_sensmat(x, y, xout, interp_type='lin-lin', ret_mat=True):
    """Compute sensitivity matrix for basic mappings."""
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
        if np.any(idcs2 >= len(x)):
            raise ValueError('some value in xout larger than largest value in x')
        if np.any(idcs2 < 1):
            raise ValueError('some value in xout smaller than smallest value in x')

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



def romberg_integral_propagate(x, fun, maxord=4):
    """Definite integral by Romberg method.

    Romberg integration is performed for each
    interval defined by the break points in x.
    The reason is that we deal with piecewise
    defined functions, and some interpolation
    law used in-between the mesh points.

    NOTE: The implementation of this function
          is a bit inefficient because it
          calculates the same functions values
          several times. Can be improved at
          some point in the future.
    """
    J = maxord
    # pre-calculate all function values
    # required to perform Romberg integration
    # up to a specified order
    funvals = fun(x)
    ftensor_list = []
    funvals_a = funvals[:-1]
    funvals_b = funvals[1:]
    xdiffs = np.diff(x).reshape((len(x)-1, 1))
    T_list = []
    curh = xdiffs.copy()
    for j in range(1, J+1):

        if j < J:
            curh.shape = (len(curh),1)
            xtensor = (x[:-1].reshape(len(x)-1,1) +
                    curh/2 * np.arange(1, 2**j).reshape((1,2**j-1)))
            curshape = xtensor.shape
            xtensor.shape = (np.prod(curshape),)
            funvals = fun(xtensor)
            funvals.shape = curshape
            ftensor_list.append(funvals)

        curh.shape = (len(curh),)
        # do the Romberg integration simultaneously
        # for all the intervals defined by x;
        # in each interval an independent integration
        # is performed up to order J
        # link to document with good explanation:
        # https://www.math.usm.edu/lambers/mat460/fall09/lecture29.pdf
        T_j1 = curh/2 * (funvals_a + funvals_b)
        if j >= 2:
            T_j1 += curh * np.sum(ftensor_list[j-2], axis=1)
        T_list.append([T_j1])
        # NOTE: this loop is not entered for the case j=1
        for k in range(2, j+1):
            T_jk = T_list[j-1][k-2] + 1/(4**(k-1)-1)*(T_list[j-1][k-2] - T_list[j-2][k-3])
            T_list[j-1].append(T_jk)
        curh /= 2

    intval1 = np.sum(T_list[J-2][J-2])
    intval2 = np.sum(T_list[J-1][J-1])
    # looking at
    # https://math.stackexchange.com/questions/1291613/romberg-integration-accuracy
    # I guess this is an overestimate but in the right ballpark
    est_error = np.sum(intval2-intval1)
    return intval2



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

