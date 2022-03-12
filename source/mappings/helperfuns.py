import numpy as np
from scipy.sparse import csr_matrix


SHAPE_MT_IDS = (2,4,8,9)


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
