import numpy as np


def get_priorblock_identifier(priorblock):
    return int(priorblock['ID'])


def get_nodename(priorblock):
    return 'xsid_' + str(get_priorblock_identifier(priorblock))


def get_quantity_type(priorblock):
    return int(1)


def get_energies(priorblock):
    return np.array(priorblock['EN'], dtype=np.float64, copy=True)


def get_values(priorblock):
    return np.array(priorblock['CS'], dtype=np.float64, copy=True)


def get_uncertainties(priorblock):
    n = len(get_energies(priorblock))
    return np.full(n, np.inf, dtype=np.float64)


def get_description(priorblock):
    return priorblock['CLAB'].strip()
