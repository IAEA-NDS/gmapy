import numpy as np
from scipy.sparse import block_diag, csr_matrix, diags
from collections import OrderedDict
from .unc_utils import (scale_covmat, cov2cor, calculate_ppp_factors,
        fix_cormat)

from .specialized_uncertainty_funs import (
        legacy_uncertainty_funs as legacy_uncfuns,
        simple_uncertainty_funs as simple_uncfuns
    )



def get_method(blocktype, method):
    if blocktype == 'legacy-experiment-datablock':
        mod = legacy_uncfuns
    elif blocktype == 'simple-experiment-datablock':
        mod = simple_uncfuns
    else:
        TypeError('unknown datablock type')
    special_method = getattr(mod, method)
    return special_method



def create_relunc_vector(datablock_list):
    relunc_list = []
    for datablock in datablock_list:
        curtype = datablock['type']
        curfun = get_method(curtype, 'create_relunc_vector')
        relunc_list.append(curfun(datablock))
    return np.concatenate(relunc_list)



def create_relative_datablock_covmat(datablock):
    curfun = get_method(datablock['type'], 'create_relative_datablock_covmat')
    return curfun(datablock)



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
            try:
                curcormat = fix_cormat(curcormat)
            except Exception:
                ds_ids = ', '.join((str(ds['NS']) for ds in db['datasets']))
                raise ValueError(f'Problem with covariance matrix of datablock '
                                 f'with dataset ids {ds_ids}')

        curcovmat = scale_covmat(curcormat, curabsuncs)

        covmat_list.append(csr_matrix(curcovmat))
        start_idx = next_idx

    covmat = block_diag(covmat_list, format='csr')
    return covmat



def create_prior_covmat(prior_list):
    cov_list = []
    for curprior in prior_list:

        curtype = curprior['type']
        if curtype == 'legacy-prior-cross-section':
            n = len(curprior['EN'])
            curcov = diags(np.full(n, np.inf, dtype='d'))
            cov_list.append(curcov)

        elif curtype == 'legacy-fission-spectrum':
            n = len(curprior['ENFIS'])
            curcov = diags(np.full(n, 0., dtype='d'))
            cov_list.append(curcov)

        elif curtype == 'modern-fission-spectrum':
            en = curprior['energies']
            energies = curprior['covmat']['energies']
            curcov1 = diags(np.full(len(en), 0., dtype='d'))
            curcov2 = np.array(curprior['covmat']['matrix'])
            if curcov2.shape[0] != curcov2.shape[0]:
                raise TypeError('covariance matrix in prior fission spectrum ' +
                                'is not square')
            if len(energies)-1 != curcov2.shape[0]:
                raise IndexError('mismatch between number of energies and ' +
                                 'dimension of covariance matrix')
            cov_list.append(curcov1)
            cov_list.append(curcov2)

        else:
            raise TypeError(f'Unsupported prior block type {curtype}')

    covmat = block_diag(cov_list, format='csr')
    return covmat
