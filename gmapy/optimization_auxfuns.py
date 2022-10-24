import numpy as np
from sksparse.cholmod import cholesky


def create_posterior_funs(mapping, dt, covmat, fnscale=1, print_res=True):
    dt = dt.sort_index(inplace=False)
    meas = np.full(len(dt), 0.)
    meas[dt.index] = dt['DATA']
    # NaN (=not a number) values indicate that quantity was not measured
    not_isobs = np.isnan(meas)
    isobs = np.logical_not(not_isobs)
    # restrict the measurement vector to the real measurement points
    meas = meas[isobs]
    # quantities with zero uncertainty are fixed,
    # hence will be propagated but not adjusted
    has_zerounc = covmat.diagonal() == 0.
    has_nonzerounc = np.logical_not(has_zerounc)
    # not observed quantities with uncertainty are adjustable
    isadj = np.logical_and(has_nonzerounc, not_isobs)

    expcov = covmat[isobs,:][:,isobs]
    priorcov = covmat[isadj,:][:,isadj]
    p0 = dt.loc[isadj, 'PRIOR'].to_numpy()
    expvals = dt.loc[isobs, 'DATA'].to_numpy()
    priorcov_fact = cholesky(priorcov.asformat('csc'))
    expcov_fact = cholesky(expcov.asformat('csc'))
    orig_refvals = dt['PRIOR'].to_numpy() 

    def this_logposterior(pcur):
        refvals = orig_refvals.copy()
        refvals[isadj] = pcur
        preds = mapping.propagate(dt, refvals)[isobs] 
        res = logposterior(p0, priorcov_fact, expvals,
                expcov_fact, pcur, preds)
        if print_res:
            print(res)
        return res * fnscale

    def this_grad_logposterior(pcur):
        refvals = orig_refvals.copy()
        refvals[isadj] = pcur
        preds = mapping.propagate(dt, refvals)[isobs]
        S = mapping.jacobian(dt, refvals, ret_mat=True)
        S = S[isobs,:][:,isadj]
        res = grad_logposterior(p0, priorcov_fact, S,
                expvals, expcov_fact, pcur, preds)
        return res * fnscale

    return {'logposterior': this_logposterior,
            'grad_logposterior': this_grad_logposterior,
            'adj_idcs': np.where(isadj)[0]}


def logposterior(p0, priorcov_fact, expvals, expcov_fact, pcur, preds):
    pd = pcur - p0
    d = expvals - preds 
    res = d.T @ expcov_fact(d) + pd.T @ priorcov_fact(pd)
    return (-0.5*res)


def grad_logposterior(p0, priorcov_fact, S, expvals, expcov_fact, pcur, preds):
    pd = p0 - pcur
    d = expvals - preds
    res = S.T @ expcov_fact(d) + priorcov_fact(pd)
    return res

