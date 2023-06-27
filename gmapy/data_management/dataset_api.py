from .specialized_dataset_apis import (
    legacy_dataset_api
)


def _get_method(dataset, method):
    ds_type = dataset['type']
    if ds_type == 'legacy-experiment-dataset':
        module = legacy_dataset_api
    else:
        raise ValueError(f'unknown dataset type `{ds_type}`')
    special_method = getattr(module, method)
    return special_method


def get_dataset_identifier(dataset):
    fun = _get_method(dataset, 'get_dataset_identifier')
    return fun(dataset)


def get_quantity_type(dataset):
    fun = _get_method(dataset, 'get_quantity_type')
    return fun(dataset)


def get_reaction_identifiers(dataset):
    fun = _get_method(dataset, 'get_reaction_identifiers')
    return fun(dataset)


def get_measured_values(dataset):
    fun = _get_method(dataset, 'get_measured_values')
    return fun(dataset)


def get_incident_energies(dataset):
    fun = _get_method(dataset, 'get_incident_energies')
    return fun(dataset)


def get_authors_string(dataset):
    fun = _get_method(dataset, 'get_authors_string')
    return fun(dataset)


def get_publication_string(dataset):
    fun = _get_method(dataset, 'get_publication_string')
    return fun(dataset)
