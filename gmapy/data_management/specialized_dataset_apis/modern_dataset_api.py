from ..dispatch_utils import generate_method_caller
from .modern_dataset_api_versions import version_0_1


TYPE_SPECIFICATION = 'modern-experiment-dataset'


_api_mapping = {
    '0.1':  version_0_1
}


def get_api_version(dataset):
    return str(dataset['api_version'])


def add_api_version(dataset, api_version):
    dataset['api_version'] = str(api_version)


_call_method = generate_method_caller(get_api_version, _api_mapping)


def get_dataset_identifier(dataset):
    return _call_method(dataset, 'get_dataset_identifier')


def add_dataset_identifier(dataset, dataset_identifier):
    return _call_method(dataset, 'add_dataset_identifier', dataset_identifier)


def get_quantity_type(dataset):
    return _call_method(dataset, 'get_quantity_type')


def add_quantity_type(dataset, quantity_type):
    return _call_method(dataset, 'add_quantity_type', quantity_type)


def get_reaction_identifiers(dataset):
    return _call_method(dataset, 'get_reaction_identifiers')


def add_reaction_identifiers(dataset, reaction_identifiers):
    return _call_method(dataset, 'add_reaction_identifiers', reaction_identifiers)


def get_measured_values(dataset):
    return _call_method(dataset, 'get_measured_values')


def add_measured_values(dataset, measured_values):
    return _call_method(dataset, 'add_measured_values', measured_values)


def get_incident_energies(dataset):
    return _call_method(dataset, 'get_incident_energies')


def add_incident_energies(dataset, incident_energies):
    return _call_method(dataset, 'add_incident_energies', incident_energies)


def get_authors_string(dataset):
    return _call_method(dataset, 'get_authors_string')


def add_authors_string(dataset, authors_string):
    return _call_method(dataset, 'add_authors_string', authors_string)


def get_publication_string(dataset):
    return _call_method(dataset, 'get_publication_string')


def add_publication_string(dataset, publication_string):
    return _call_method(dataset, 'add_publication_string', publication_string)


def get_year(dataset):
    return _call_method(dataset, 'get_year')


def add_year(dataset, year):
    return _call_method(dataset, 'add_year', year)
