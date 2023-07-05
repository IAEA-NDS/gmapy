import numpy as np
from scipy.sparse import block_diag, csr_matrix, diags
from .unc_utils import (
    scale_covmat,
    cov2cor,
    calculate_ppp_factors,
    fix_cormat
)
from .dispatch_utils import generate_method_caller
from . import datablock_api as dbapi
from . import dataset_api as dsapi
from . import priorblock_api as priorapi
from .specialized_datablock_apis import (
    legacy_datablock_uncertainty_api as legacy_uncfuns,
    simple_datablock_uncertainty_api as simple_uncfuns
)


_api_mapping = {
    'legacy-experiment-datablock': legacy_uncfuns,
    'simple-experiment-datablock': simple_uncfuns
}


_call_method = generate_method_caller(dbapi.get_datablock_type, _api_mapping)


def create_relunc_vector(datablock_list):
    relunc_list = []
    for datablock in datablock_list:
        curvec = _call_method(datablock, 'create_relunc_vector')
        relunc_list.append(curvec)
    return np.concatenate(relunc_list)


def create_relative_datablock_covmat(datablock):
    return _call_method(datablock, 'create_relative_datablock_covmat')


def create_experimental_covmat(datablock_list, propcss=None,
        fix_ppp_bug=True, fix_covmat=True, relative=False):
    """Calculate experimental covariance matrix."""
    uncs = create_relunc_vector(datablock_list)
    covmat_list = []
    start_idx = 0
    for db in datablock_list:
        numpts = 0
        curexpcss = []
        datasets = tuple(dbapi.dataset_iterator(db))
        for ds in datasets:
            css = dsapi.get_measured_values(ds)
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
        dbtype = dbapi.get_datablock_type(db)
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
                    str(dsapi.get_dataset_identifier(ds)) for ds in datasets))
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
        uncs = priorapi.get_uncertainties(curprior)
        curcov = diags(np.square(uncs))
        cov_list.append(curcov)
    covmat = block_diag(cov_list, format='csr')
    return covmat
