import numpy as np


TYPE_SPECIFICATION = 'legacy-experiment-dataset'
API_VERSION = None


def get_dataset_identifier(dataset):
    return int(dataset['NS'])


def add_dataset_identifier(dataset, dataset_identifier):
    dataset['NS'] = dataset_identifier


def get_quantity_type(dataset):
    return int(dataset['MT'])


def add_quantity_type(dataset, quantity_type):
    dataset['MT'] = int(quantity_type)


def get_reaction_identifiers(dataset):
    return np.array(dataset['NT'], dtype=np.int32)


def add_reaction_identifiers(dataset, reaction_identifiers):
    dataset['NT'] = list(reaction_identifiers)


def get_measured_values(dataset):
    return np.array(dataset['CSS'], dtype=np.float64)


def add_measured_values(dataset, measured_values):
    dataset['CSS'] = list(measured_values)


def get_incident_energies(dataset):
    return np.array(dataset['E'], dtype=np.float64)


def add_incident_energies(dataset, incident_energies):
    dataset['E'] = list(incident_energies)


def get_authors_string(dataset):
    return str(dataset['CLABL'])


def add_authors_string(dataset, authors_string):
    dataset['CLABL'] = str(authors_string)


def get_publication_string(dataset):
    return str(dataset['BREF'])


def add_publication_string(dataset, publication_string):
    dataset['BREF'] = str(publication_string)


def get_year(dataset):
    return int(dataset['YEAR'])


def add_year(dataset, year):
    dataset['YEAR'] = int(year)
