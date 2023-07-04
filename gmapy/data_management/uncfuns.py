import numpy as np
from scipy.sparse import block_diag, csr_matrix, diags
from .unc_utils import (
    scale_covmat,
    cov2cor,
    calculate_ppp_factors,
    fix_cormat
)
from .datablock_api import (
    dataset_iterator,
    get_datablock_type
)
from .dataset_api import (
    get_measured_values,
    get_dataset_identifier
)
from .specialized_uncertainty_funs import (
    legacy_uncertainty_funs as legacy_uncfuns,
    simple_uncertainty_funs as simple_uncfuns
)


def _get_method(datablock, method):
    blocktype = datablock['type']
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
        curfun = _get_method(datablock, 'create_relunc_vector')
        relunc_list.append(curfun(datablock))
    return np.concatenate(relunc_list)


def create_relative_datablock_covmat(datablock):
    curfun = _get_method(datablock, 'create_relative_datablock_covmat')
    return curfun(datablock)


def create_experimental_covmat(datablock_list, propcss=None,
        fix_ppp_bug=True, fix_covmat=True, relative=False):
    """Calculate experimental covariance matrix."""
    uncs = create_relunc_vector(datablock_list)
    covmat_list = []
    start_idx = 0
    for db in datablock_list:
        numpts = 0
        curexpcss = []
        datasets = tuple(dataset_iterator(db))
        for ds in datasets:
            css = get_measured_values(ds)
            numpts += len(css)
            curexpcss.extend(css)
        next_idx = start_idx + numpts

        curuncs = uncs[start_idx:next_idx]
        curpropcss = propcss[start_idx:next_idx] if propcss is not None else curexpcss
        ppp_factors = calculate_ppp_factors(datasets, curpropcss)
        cureffuncs = curuncs * ppp_factors
        curabsuncs = curexpcss * cureffuncs * 0.01

        curcovmat = create_relative_datablock_covmat(db)

        # This if-else statement is here to be able to
        # reproduce a bug of the Fortran GMAP version
        dbtype = get_datablock_type(db)
        if (dbtype == 'legacy-experiment-datablock' and
                'ECOR' not in db and not fix_ppp_bug):
            curcormat = legacy_uncfuns.relcov_to_wrong_cor(curcovmat, datasets, curpropcss)
        else:
            curcormat = cov2cor(curcovmat)

        if fix_covmat:
            try:
                curcormat = fix_cormat(curcormat)
            except Exception:
                ds_ids = ', '.join((
                    str(get_dataset_identifier(ds)) for ds in datasets))
                raise ValueError(f'Problem with covariance matrix of datablock '
                                 f'with dataset ids {ds_ids}')

        if relative:
            curcovmat = scale_covmat(curcormat, curabsuncs/curpropcss)
        else:
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
