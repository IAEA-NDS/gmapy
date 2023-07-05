from .dispatch_utils import generate_method_caller
from .specialized_dataset_apis import (
    legacy_dataset_api
)
from .specialized_dataset_apis import modern_dataset_api


_api_mapping = {
    'legacy-experiment-dataset': legacy_dataset_api,
    'modern-experiment-dataset-v0.1': modern_dataset_api.version_0_1
}


def get_dataset_type(dataset):
    dataset_type = str(dataset['type'])
    if 'api_version' in dataset:
        api_version = str(dataset['api_version'])
        dataset_type += '-v' + api_version
    return dataset_type


_call_method = generate_method_caller(get_dataset_type, _api_mapping)
_required_getter_setter_list = (
    ('get_dataset_identifier', 'add_dataset_identifier'),
    ('get_quantity_type', 'add_quantity_type'),
    ('get_reaction_identifiers', 'add_reaction_identifiers'),
    ('get_measured_values', 'add_measured_values'),
    ('get_incident_energies', 'add_incident_energies'),
    ('get_authors_string', 'add_authors_string'),
    ('get_publication_string', 'add_publication_string'),
    ('get_year', 'add_year')
)


def get_dataset_identifier(dataset):
    return _call_method(dataset, 'get_dataset_identifier')


def add_dataset_identifier(dataset, dataset_identifier):
    _call_method(dataset, 'add_dataset_identifier', dataset_identifier)


def get_nodename(dataset):
    return 'exp_' + str(get_dataset_identifier(dataset))


def get_quantity_type(dataset):
    return _call_method(dataset, 'get_quantity_type')


def add_quantity_type(dataset, quantity_type):
    _call_method(dataset, 'add_quantity_type', quantity_type)


def get_reaction_identifiers(dataset):
    return _call_method(dataset, 'get_reaction_identifiers')


def add_reaction_identifiers(dataset, reaction_identifiers):
    _call_method(dataset, 'add_reaction_identifiers', reaction_identifiers)


def get_measured_values(dataset):
    return _call_method(dataset, 'get_measured_values')


def add_measured_values(dataset, measured_values):
    _call_method(dataset, 'add_measured_values', measured_values)


def get_incident_energies(dataset):
    return _call_method(dataset, 'get_incident_energies')


def add_incident_energies(dataset, incident_energies):
    _call_method(dataset, 'add_incident_energies', incident_energies)


def get_authors_string(dataset):
    return _call_method(dataset, 'get_authors_string')


def add_authors_string(dataset, authors_string):
    _call_method(dataset, 'add_authors_string', authors_string)


def get_publication_string(dataset):
    return _call_method(dataset, 'get_publication_string')


def add_publication_string(dataset, publication_string):
    _call_method(dataset, 'add_publication_string', publication_string)


def get_year(dataset):
    return _call_method(dataset, 'get_year')


def add_year(dataset, year):
    _call_method(dataset, 'add_year', year)


def transfer_dataset_info(source_dataset, target_dataset):
    for getter, setter in _required_getter_setter_list:
        getfun = globals()[getter]
        setfun = globals()[setter]
        setfun(target_dataset, getfun(source_dataset))
