import numpy as np


def mapclass_with_params(origclass, **kwargs):
    class WrapperClass(origclass):
        def __init__(self, datatable, *args, **kwargs2):
            kwargs2.update(kwargs)
            super().__init__(datatable, *args, **kwargs2)
    return WrapperClass


def get_fission_spectrum_interp(fistable):
    """Determine the interpretation of the fission spectrum rows.

    Returns None if the spectrum values are bin-integrated probabilities
    (legacy binned interpretation), which is also assumed if the table
    lacks an INTERP column. Otherwise, the spectrum values are a
    point-wise density and the per-point vector of interpolation laws
    is returned.
    """
    if 'INTERP' not in fistable.columns:
        return None
    interp = fistable['INTERP'].to_list()
    is_binned = [
        el is None or (isinstance(el, float) and np.isnan(el))
        or el == 'legacy-binned' for el in interp
    ]
    if all(is_binned):
        return None
    if any(is_binned):
        raise ValueError(
            'mixture of legacy-binned and point-wise interpolation ' +
            'laws in fission spectrum'
        )
    return np.array(interp)


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
