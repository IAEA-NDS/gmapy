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
