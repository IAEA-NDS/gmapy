import numpy as np
from scipy.sparse import block_diag, csr_matrix
from collections import OrderedDict
from .unc_utils import (scale_covmat, cov2cor, calculate_ppp_factors,
        fix_cormat)

from .specialized_uncertainty_funs import \
        legacy_uncertainty_funs as legacy_uncfuns



def create_relunc_vector(datablock_list):
    relunc_list = []
    for datablock in datablock_list:
        if datablock['type'] == 'legacy-experiment-datablock':
            relunc_list.append(legacy_uncfuns.create_relunc_vector([datablock]))
        else:
            TypeError('datablock type not implemented')
    return np.concatenate(relunc_list)



def create_relative_datablock_covmat(datablock):
    if datablock['type'] == 'legacy-experiment-datablock':
        return legacy_uncfuns.create_relative_datablock_covmat(datablock)
    else:
        TypeError('datablock type not implemented')



def create_experimental_covmat(datablock_list, propcss=None,
        fix_ppp_bug=True, fix_covmat=True):
    """Calculate experimental covariance matrix."""
    uncs = create_relunc_vector(datablock_list)
    covmat_list = []
    start_idx = 0
    for db in datablock_list:
        numpts = 0
        curexpcss = []
        for ds in db['datasets']:
            numpts += len(ds['CSS'])
            curexpcss.extend(ds['CSS'])
        next_idx = start_idx + numpts

        curuncs = uncs[start_idx:next_idx]
        curpropcss = propcss[start_idx:next_idx] if propcss is not None else curexpcss
        ppp_factors = calculate_ppp_factors(db['datasets'], curpropcss)
        cureffuncs = curuncs * ppp_factors
        curabsuncs = curexpcss * cureffuncs * 0.01

        curcovmat = create_relative_datablock_covmat(db)

        # This if-else statement is here to be able to
        # reproduce a bug of the Fortran GMAP version
        if (db['type'] == 'legacy-experiment-datablock' and
                'ECOR' not in db and not fix_ppp_bug):
            curcormat = legacy_uncfuns.relcov_to_wrong_cor(curcovmat, db['datasets'], curpropcss)
        else:
            curcormat = cov2cor(curcovmat)

        if fix_covmat:
            curcormat = fix_cormat(curcormat)

        curcovmat = scale_covmat(curcormat, curabsuncs)

        covmat_list.append(csr_matrix(curcovmat))
        start_idx = next_idx

    covmat = block_diag(covmat_list, format='csr')
    return covmat

