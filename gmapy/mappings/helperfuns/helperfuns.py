import numpy as np
from scipy.sparse import csr_matrix


def return_matrix(idcs1, idcs2, vals, dims, how):
    """Return a matrix defined by triples in desired format."""
    Sdic = {'idcs1': np.array(idcs1, dtype=int),
            'idcs2': np.array(idcs2, dtype=int),
            'x': np.array(vals, dtype=float)}
    if how == 'csr':
        S = csr_matrix((Sdic['x'], (Sdic['idcs2'], Sdic['idcs1'])), shape=dims)
        return S
    elif how == 'dic':
        return Sdic
    else:
        raise ValueError('invalid value of parameter "how"')


def return_matrix_new(mat, how):
    if how == 'csr':
        return mat.tocsr()
    elif how == 'dic':
        tmp = mat.tocoo()
        Sdic = {
            'idcs1': np.array(tmp.col, dtype=int, copy=False),
            'idcs2': np.array(tmp.row, dtype=int, copy=False),
            'x': np.array(tmp.data, dtype=float, copy=False)
        }
        return Sdic
    else:
        raise ValueError('invalid value of parameter "how"')


def get_legacy_to_pointwise_fis_factors(energies):
    # The fission spectrum values in the legacy GMA database
    # are given as a histogram (piecewise rectangular function)
    # where the spectrum value in each bin is divided by the
    # energy bin size. For the new routine, where we interpret
    # the spectrum point-wise, we therefore need to multiply
    # by the energy bin size
    assert len(np.unique(energies)) == len(energies)
    ensfis = np.array(energies)
    sort_idcs = ensfis.argsort()
    sorted_ensfis = ensfis[sort_idcs]
    xdiff = np.diff(sorted_ensfis)
    xmid = sorted_ensfis[:-1] + xdiff/2
    sorted_scl = np.full(len(sorted_ensfis), 1.)
    sorted_scl[1:-1] /= np.diff(xmid)
    sorted_scl[0] /= (xdiff[0]/2)
    sorted_scl[-1] /= (xdiff[-1]/2)
    scl = np.empty(len(sorted_scl), dtype=float)
    scl[sort_idcs] = sorted_scl
    return scl
