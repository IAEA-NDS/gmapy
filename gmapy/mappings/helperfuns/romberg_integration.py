import numpy as np


def compute_romberg_integral(x, fun, maxord=4, atol=1e-8, rtol=1e-5, dfun=None):
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
    if maxord < 2:
        raise ValueError('maxord must be at least two')
    if atol <= 0:
        raise ValueError('atol must be positive')
    if rtol <= 0:
        raise ValueError('rtol must be positive')

    calc_deriv = False
    if dfun is not None:
        calc_deriv=True
        if not callable(dfun):
            raise TypeError('dfun must be a function')

    x = np.sort(np.array(x, dtype=float))
    funvals = fun(x)
    funvals_a = funvals[:-1]
    funvals_b = funvals[1:]
    if calc_deriv:
        # NOTE: The function dfun provided as argument
        # yields for an x-value two partial derivatives
        # with respect to the two enclosing mesh points
        # left and right, respectively. If the x-value
        # coincides with a mesh point, one partial
        # derivative will be zero, hence  we can sum
        # over the two derivatives to get the derivative
        # with repect to the value of the coinciding
        # mesh point.
        dfunvals = np.array(dfun(x))
        dfunvals = np.sum(dfunvals, axis=0)
        dfunvals_a = dfunvals[:-1]
        dfunvals_b = dfunvals[1:]
    # do the Romberg integration simultaneously
    # for all the intervals defined by x;
    # in each interval an independent integration
    # is performed up to order J
    # link to document with good explanation:
    # https://www.math.usm.edu/lambers/mat460/fall09/lecture29.pdf
    ftensor_list = []
    df1tensor_list = []
    df2tensor_list = []
    T_list = []
    dT1_list = []
    dT2_list = []
    curh = np.diff(x).reshape((len(x)-1, 1))
    for j in range(1, maxord+1):

        # pre-calculate all function values obtained
        # by walking curh/2 steps
        # required in the Romberg integration
        if j < maxord:
            curh.shape = (len(curh),1)
            xtensor = (x[:-1].reshape(len(x)-1,1) +
                    curh/2 * np.arange(1, 2**j).reshape((1,2**j-1)))
            curshape = xtensor.shape
            xtensor.shape = (np.prod(curshape),)
            funvals = fun(xtensor)
            funvals.shape = curshape
            ftensor_list.append(funvals)
            if calc_deriv:
                # NOTE: xtensor does not contain
                #       any values of x. This is
                #       important because otherwise
                #       dfun1 and dfun2 may not
                #       yield the correct result
                #       (i.e., 0 instead of 1)
                dfunvals1, dfunvals2 = dfun(xtensor)
                dfunvals1.shape = curshape
                dfunvals2.shape = curshape
                df1tensor_list.append(dfunvals1)
                df2tensor_list.append(dfunvals2)

        curh.shape = (len(curh),)
        T_j1 = curh/2 * (funvals_a + funvals_b)
        T_list.append([T_j1])
        if calc_deriv:
            dT1_j1 = curh/2 * dfunvals_a
            dT2_j1 = curh/2 * dfunvals_b
            dT1_list.append([dT1_j1])
            dT2_list.append([dT2_j1])

        if j >= 2:
            T_j1 += curh * np.sum(ftensor_list[j-2], axis=1)
            if calc_deriv:
                dT1_j1 += curh * np.sum(df1tensor_list[j-2], axis=1)
                dT2_j1 += curh * np.sum(df2tensor_list[j-2], axis=1)

        # NOTE: this loop is not entered for the case j=1
        for k in range(2, j+1):
            T_jk = T_list[j-1][k-2] + 1/(4**(k-1)-1)*(T_list[j-1][k-2] - T_list[j-2][k-3])
            T_list[j-1].append(T_jk)
            if calc_deriv:
                dT1_jk = (dT1_list[j-1][k-2] +
                          1/(4**(k-1)-1)*(dT1_list[j-1][k-2] - dT1_list[j-2][k-3]))
                dT2_jk = (dT2_list[j-1][k-2] +
                          1/(4**(k-1)-1)*(dT2_list[j-1][k-2] - dT2_list[j-2][k-3]))
                dT1_list[j-1].append(dT1_jk)
                dT2_list[j-1].append(dT2_jk)

        if calc_deriv:
            last_dT1 = dT1_list[j-1][j-1]
            last_dT2 = dT2_list[j-1][j-1]
            dT = np.empty(len(x), dtype=float)
            dT[1:-1] = last_dT1[1:] + last_dT2[:-1]
            dT[0] = last_dT1[0]
            dT[-1] = last_dT2[-1]

        est_intval = np.sum(T_list[j-1][j-1])

        if j > 1:
            # looking at
            # https://math.stackexchange.com/questions/1291613/romberg-integration-accuracy
            # I guess this is an overestimate but in the right ballpark
            prev_intval = np.sum(T_list[j-2][j-2])
            est_error = np.abs(np.sum(np.abs(est_intval-prev_intval)))
            # accuracy goal reached?
            if (est_error < np.abs(atol + rtol*est_intval)):
                break

        # half step size for next loop iteration
        curh /= 2

    if est_error >= np.abs(atol + rtol*est_intval):
        raise ValueError(f'Desired accuracy (atol={atol}, rtol={rtol}) could not be reached.\n' +
                         f'The estimated absolute error is {est_error} ' +
                         f'and the estimated integral {est_intval}.\n' +
                          'Try to increase maxord or reduce the accuracy by increasing atol and or rtol.')

    if not calc_deriv:
        return est_intval
    else:
        return dT


