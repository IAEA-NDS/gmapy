import numpy as np
from ...quantity_types import map_str2mt


def get_dataset_identifier(dataset):
    return int(dataset['identifier'])


def get_quantity_type(dataset):
    return map_str2mt[dataset['quantity']]


def get_reaction_identifiers(dataset):
    quantstr = dataset['quantity']
    if quantstr in ('xs', 'xs_shape', 'legacy_sacs'):
        reacids = dataset['reaction_identifier']

    elif quantstr in ('xs_ratio', 'xs_ratio_shape', 'legacy_sacs_ratio'):
        reacids = [
            dataset['reaction_identifier_numerator'],
            dataset['reaction_identifier_denominator']
        ]

    elif quantstr in ('xs_sum', 'xs_sum_shape'):
        reacids = [dataset['reaction_identifiers']]

    elif quantstr in ('xs_div_xs_sum', 'xs_div_xs_sum_shape'):
        reacids = [
            dataset['reaction_identifier_numerator'],
            dataset['reaction_identifiers_denominator']
        ]

    else:
        raise ValueError(f'unknown quantity type `{quantstr}`')
    return np.array(reacids, dtype=np.int32)


def get_measured_values(dataset):
    return np.array(dataset['measured_values'], dtype=np.float64)


def get_incident_energies(dataset):
    quantstr = dataset['quantity']
    if quantstr in ('legacy_sacs', 'legacy_sacs_ratio'):
        return np.array([np.nan], dtype=np.float64)
    return np.array(dataset['energies'], dtype=np.float64)


def get_authors_string(dataset):
    return ', '.join(dataset['authors'])


def get_publication_string(dataset):
    return str(dataset['publication'])
