from .dispatch_utils import generate_method_caller
from . import datablock_api as dbapi
from .specialized_datablock_apis import (
    legacy_datablock_uncertainty_api as legacy_uncfuns,
    simple_datablock_uncertainty_api as simple_uncfuns
)


_api_mapping = {
    'legacy-experiment-datablock': legacy_uncfuns,
    'simple-experiment-datablock': simple_uncfuns
}


_call_method = generate_method_caller(dbapi.get_datablock_type, _api_mapping)


def create_relunc_vector(datablock):
    return _call_method(datablock, 'create_relunc_vector')


def create_relative_datablock_covmat(datablock):
    return _call_method(datablock, 'create_relative_datablock_covmat')
