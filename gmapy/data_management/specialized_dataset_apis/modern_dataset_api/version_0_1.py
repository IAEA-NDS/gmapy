import numpy as np
from ...quantity_types import map_str2mt, map_mt2str


TYPE_SPECIFICATION = 'modern-experiment-dataset'
API_VERSION = '0.1'


def get_dataset_identifier(dataset):
    return int(dataset['identifier'])


def add_dataset_identifier(dataset, dataset_identifier):
    dataset['identifier'] = int(dataset_identifier)


def get_quantity_type(dataset):
    return map_str2mt[dataset['quantity']]


def add_quantity_type(dataset, quantity_type):
    dataset['quantity'] = map_mt2str[quantity_type]


def get_reaction_identifiers(dataset):
    quantstr = dataset['quantity']
    if quantstr in ('xs', 'xs_shape', 'legacy_sacs'):
        reacids = [int(dataset['reaction_identifier'])]

    elif quantstr in ('xs_ratio', 'xs_ratio_shape', 'legacy_sacs_ratio'):
        reacids = [
            int(dataset['reaction_identifier_numerator']),
            int(dataset['reaction_identifier_denominator'])
        ]

    elif quantstr in ('xs_sum', 'xs_sum_shape'):
        reacids = list(dataset['reaction_identifiers'])

    elif quantstr in ('xs_div_xs_sum', 'xs_div_xs_sum_shape'):
        reacids = (
            [int(dataset['reaction_identifier_numerator'])] +
            list(dataset['reaction_identifiers_denominator'])
        )

    else:
        raise ValueError(f'unknown quantity type `{quantstr}`')
    return np.array(reacids, dtype=np.int32)


def add_reaction_identifiers(dataset, reaction_identifiers):
    quantstr = dataset['quantity']
    quantstr = dataset['quantity']
    if quantstr in ('xs', 'xs_shape', 'legacy_sacs'):
        dataset['reaction_identifier'] = int(reaction_identifiers[0])

    elif quantstr in ('xs_ratio', 'xs_ratio_shape', 'legacy_sacs_ratio'):
        dataset['reaction_identifier_numerator'] = int(reaction_identifiers[0])
        dataset['reaction_identifier_denominator'] = int(reaction_identifiers[1])

    elif quantstr in ('xs_sum', 'xs_sum_shape'):
        dataset['reaction_identifiers'] = list(reaction_identifiers)

    elif quantstr in ('xs_div_xs_sum', 'xs_div_xs_sum_shape'):
        dataset['reaction_identifier_numerator'] = int(reaction_identifiers[0])
        dataset['reaction_identifiers_denominator'] = list(reaction_identifiers[1:])
    else:
        raise ValueError(f'unknown quantity type `{quantstr}`')


def get_measured_values(dataset):
    return np.array(dataset['measured_values'], dtype=np.float64)


def add_measured_values(dataset, measured_values):
    dataset['measured_values'] = list(measured_values)


def get_incident_energies(dataset):
    quantstr = dataset['quantity']
    if quantstr in ('legacy_sacs', 'legacy_sacs_ratio'):
        return np.array([np.nan], dtype=np.float64)
    return np.array(dataset['energies'], dtype=np.float64)


def add_incident_energies(dataset, incident_energies):
    quantstr = dataset['quantity']
    if quantstr not in ('legacy_sacs', 'legacy_sacs_ratio'):
        dataset['energies'] = list(incident_energies)


def get_authors_string(dataset):
    return str(dataset['authors'])


def add_authors_string(dataset, authors_string):
    dataset['authors'] = str(authors_string)


def get_publication_string(dataset):
    return str(dataset['publication'])


def add_publication_string(dataset, publication_string):
    dataset['publication'] = str(publication_string)


def get_year(dataset):
    return int(dataset['year'])


def add_year(dataset, year):
    dataset['year'] = int(year)
