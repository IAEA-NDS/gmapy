import numpy as np
from scipy.sparse import csr_matrix



def get_sensmat_exact(ens1, ens2, idcs1=None, idcs2=None):
    """Compute sensitivity matrix to map
    values given on energy mesh ens1 to
    the mesh given by ens2. It is assumed
    that the energies in ens2 are a 
    subset of the energies in ens1."""
    ens1 = np.array(ens1)
    ens2 = np.array(ens2)
    ord = np.argsort(ens1)
    ens1 = ens1[ord]
    ridcs = np.searchsorted(ens1, ens2, side='left')
    if not np.all(ens1[ridcs] == ens2):
        raise ValueError('mismatching energies encountered' +
                str(ens1[ridcs]) + ' vs ' + str(ens2))

    curidcs2 = np.arange(len(ens2))
    curidcs1 = ord[ridcs]
    coeff = np.ones(len(ens2))
    if idcs1 is not None:
        curidcs1 = idcs1[curidcs1]
    if idcs2 is not None:
        curidcs2 = idcs2[curidcs2]
    return {'idcs1': np.array(curidcs1, dtype=int),
            'idcs2': np.array(curidcs2, dtype=int),
            'x': np.array(coeff, dtype=float)}



def propagate_exact(ens1, vals1, ens2):
    """Propagate values vals1 given on
    energy mesh ens1 to the mesh given
    by ens2. It is assumed that ens2 is
    a subset of ens1."""
    Sraw = get_sensmat_exact(ens1, ens2)
    S = csr_matrix((Sraw['x'], (Sraw['idcs2'], Sraw['idcs1'])),
              shape = (len(ens2), len(ens1)))
    return S @ vals1

