from .dispatch_utils import generate_method_caller
from .specialized_priorblock_apis import (
    legacy_prior_cross_section,
    legacy_fission_spectrum
)


_api_mapping = {
    'legacy-prior-cross-section': legacy_prior_cross_section,
    'legacy-fission-spectrum': legacy_fission_spectrum
}


def get_priorblock_type(priorblock):
    return priorblock['type']


_call_method = generate_method_caller(get_priorblock_type, _api_mapping)


def get_priorblock_identifier(priorblock):
    return _call_method(priorblock, 'get_priorblock_identifier')


def get_reaction_identifier(priorblock):
    return get_priorblock_identifier(priorblock)


def get_nodename(priorblock):
    return _call_method(priorblock, 'get_nodename')


def get_quantity_type(priorblock):
    return _call_method(priorblock, 'get_quantity_type')


def get_energies(priorblock):
    return _call_method(priorblock, 'get_energies')


def get_values(priorblock):
    return _call_method(priorblock, 'get_values')


def get_description(priorblock):
    return _call_method(priorblock, 'get_description')


def get_uncertainties(priorblock):
    return _call_method(priorblock, 'get_uncertainties')
