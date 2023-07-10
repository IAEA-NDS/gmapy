from .dispatch_utils import generate_method_getter
from .specialized_datablock_apis import (
    legacy_datablock_api,
    simple_datablock_api
)
from . import dataset_api as dsapi
import numpy as np


_api_mapping = {
    'legacy-experiment-datablock': legacy_datablock_api,
    'simple-experiment-datablock': simple_datablock_api
}


def get_datablock_type(datablock):
    return datablock['type']


_get_method = generate_method_getter(get_datablock_type, _api_mapping)


def _flatten(listlike):
    curtype = type(listlike)
    listlike = np.array(listlike).flatten()
    return curtype(listlike)


def _apply(datablock, func):
    ds_iter = dataset_iterator(datablock)
    return tuple(func(ds) for ds in ds_iter)


def dataset_iterator(datablock):
    iter_generator = _get_method(datablock, 'dataset_iterator')
    dataset_iterator = iter_generator(datablock)
    return dataset_iterator


def get_dataset_identifiers(datablock):
    return _apply(datablock, dsapi.get_dataset_identifier)


def get_quantity_types(datablock):
    return _apply(datablock, dsapi.get_quantity_type)


def get_reaction_identifiers(datablock):
    return _apply(datablock, dsapi.get_reaction_identifiers)


def get_measured_values(datablock, flatten=False):
    res = _apply(datablock, dsapi.get_measured_values)
    if flatten:
        res = _flatten(res)
    return res


def get_incident_energies(datablock, flatten=False):
    res = _apply(datablock, dsapi.get_incident_energies)
    if flatten:
        res = _flatten(res)
    return res


def get_years(datablock):
    return _apply(datablock, dsapi.get_year)


def get_authors_strings(datablock):
    return _apply(datablock, dsapi.get_authors_string)


def get_publication_strings(datablock):
    return _apply(datablock, dsapi.get_publication_string)
