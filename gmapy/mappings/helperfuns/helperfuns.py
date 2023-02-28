import numpy as np


def mapclass_with_params(origclass, *args, **kwargs):
    class WrapperClass(origclass):
        def __init__(self, datatable):
            super().__init__(datatable, *args, **kwargs)
    return WrapperClass


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
