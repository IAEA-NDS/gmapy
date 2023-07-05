from .dispatch_utils import generate_method_caller
from .specialized_dataset_apis import (
    legacy_dataset_api
)

_api_mapping = {
    'legacy-experiment-dataset': legacy_dataset_api
}


def get_dataset_type(dataset):
    dataset_type = str(dataset['type'])
    if 'api_version' in dataset:
        api_version = str(dataset['api_version'])
        dataset_type += '-v' + api_version
    return dataset_type


_call_method = generate_method_caller(get_dataset_type, _api_mapping)


def get_dataset_identifier(dataset):
    return _call_method(dataset, 'get_dataset_identifier')


def get_nodename(dataset):
    return 'exp_' + str(get_dataset_identifier(dataset))


def get_quantity_type(dataset):
    return _call_method(dataset, 'get_quantity_type')


def get_reaction_identifiers(dataset):
    return _call_method(dataset, 'get_reaction_identifiers')


def get_measured_values(dataset):
    return _call_method(dataset, 'get_measured_values')


def get_incident_energies(dataset):
    return _call_method(dataset, 'get_incident_energies')


def get_authors_string(dataset):
    return _call_method(dataset, 'get_authors_string')


def get_publication_string(dataset):
    return _call_method(dataset, 'get_publication_string')
