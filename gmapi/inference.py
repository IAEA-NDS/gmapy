import numpy as np
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.linalg.lapack import dpotri, dpotrf

from .mappings.compound_map import CompoundMap



def new_gls_update(priortable, exptable, expcovmat, retcov=False):
    """Calculate updated values and covariance matrix."""
    # prepare quantities required for update
    priorvals = priortable['PRIOR'].to_numpy()
    refvals = priorvals.copy()

    meas = exptable['DATA'].to_numpy()

    comp_map = CompoundMap()
    preds = comp_map.propagate(priortable, exptable, refvals)
    S = comp_map.jacobian(priortable, exptable, refvals, ret_mat=True)

    # for the time being mask out the fisdata block
    isfis = priortable['NODE'] == 'fis'
    not_isfis = np.logical_not(isfis)
    priorvals = priorvals[not_isfis]
    S = S[:,not_isfis].copy()

    # perform the update
    inv_post_cov = S.T @ spsolve(expcovmat, S)
    upd_priorvals = priorvals + spsolve(inv_post_cov, S.T @ (spsolve(expcovmat, meas-preds)))

    # introduce the unmodified fission spectrum in the posterior
    ext_upd_priorvals = refvals.copy()
    ext_upd_priorvals[not_isfis] = upd_priorvals

    post_covmat = None
    if retcov is True:
        # following is equivalent to:
        # post_covmat = np.linalg.inv(inv_post_cov.toarray())
        cholfact, _ = dpotrf(inv_post_cov.toarray(), False, False)
        invres , info = dpotri(cholfact)
        if info != 0:
            raise ValueError('Experimental covariance matrix not positive definite')

        # extend posterior covariance matrix with fission block
        ext_post_covmat = np.full((len(priortable), len(priortable)), 0., dtype=float)
        post_covmat = np.triu(invres) + np.triu(invres, k=1).T
        ext_post_covmat[np.ix_(not_isfis, not_isfis)] = post_covmat

    return {'upd_vals': ext_upd_priorvals, 'upd_covmat': ext_post_covmat}
