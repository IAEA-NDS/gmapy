import numpy as np


def get_priorblock_identifier(priorblock):
    return int(9999)


def get_nodename(priorblock):
    return 'fis'


def get_quantity_type(priorblock):
    return int(9999)


def get_energies(priorblock):
    return np.array(priorblock['ENFIS'], dtype=np.float64, copy=True)


def get_values(priorblock):
    return np.array(priorblock['FIS'], dtype=np.float64, copy=True)


def get_uncertainties(priorblock):
    n = len(priorblock['ENFIS'])
    return np.full(n, 0., dtype=np.float64)


def get_description(priorblock):
    return 'fission spectrum'
