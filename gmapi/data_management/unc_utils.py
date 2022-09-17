import numpy as np
from scipy.sparse import issparse


def scale_covmat(covmat, sclvec):
    if issparse(covmat):
        newcovmat = covmat.tocsc(copy=True)
        # the following lines achieve the elementwise
        # product of covmat with sclvec.T * sclvec
        # see answer https://stackoverflow.com/a/16046783/1860946
        newcovmat.data *= sclvec[newcovmat.indices]
        newcovmat = newcovmat.tocsr()
        newcovmat.data *= sclvec[newcovmat.indices]
        newcovmat = newcovmat.tocsc()
    else:
        newcovmat = covmat * sclvec.reshape(1,-1) * sclvec.reshape(-1,1)
    return newcovmat



def cov2cor(covmat):
    invuncs = 1. / np.sqrt(covmat.diagonal())
    cormat = scale_covmat(covmat, invuncs)
    np.fill_diagonal(cormat, 1.)
    return cormat



def cor2cov(cormat, uncs):
    covmat = scale_covmat(cormat, uncs)
    return covmat



def calculate_ppp_factors(datasets, css):
    cur_idx = 0
    factors = []
    for ds in datasets:
        origcss = np.array(ds['CSS'])
        next_idx = cur_idx + len(origcss)
        newcss = css[cur_idx:next_idx]
        cur_idx = next_idx
        # no PPP correction for SACS measurements
        if ds['MT'] == 6:
            factors.extend(np.ones(len(origcss), dtype='d'))
        else:
            factors.extend(newcss/origcss)
    return factors



def fix_cormat(cormat):
    """Fix non positive-definite correlation matrix."""
    if np.any(cormat.diagonal() != 1):
        raise ValueError('All diagonal elements of correlation matrix must be one')
    tries = 0
    cormat = cormat.copy()
    while tries < 15:
        success = True
        try:
            np.linalg.cholesky(cormat)
        except np.linalg.LinAlgError:
            tries += 1
            success = False
            # select all off-diagonal elements
            sel = np.logical_not(np.eye(cormat.shape[0]))
            cormat[sel] /= 1.1
        if success:
            break
    return cormat
