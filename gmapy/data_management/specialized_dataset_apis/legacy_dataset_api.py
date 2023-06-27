import numpy as np


def get_dataset_identifier(dataset):
    return int(dataset['NS'])


def get_quantity_type(dataset):
    return int(dataset['MT'])


def get_reaction_identifiers(dataset):
    return np.array(dataset['NT'], dtype=np.int32)


def get_measured_values(dataset):
    return np.array(dataset['CSS'], dtype=np.float64)


def get_incident_energies(dataset):
    return np.array(dataset['E'], dtype=np.float64)


def get_authors_string(dataset):
    return str(dataset['CLABL'])


def get_publication_string(dataset):
    return str(dataset['BREF'])
