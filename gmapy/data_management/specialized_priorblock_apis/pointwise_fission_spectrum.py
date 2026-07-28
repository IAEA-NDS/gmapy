import numpy as np


INTERP_LAWS = ('lin-lin', 'lin-log', 'log-lin', 'log-log')


def expand_interpolation_spec(interp, num_points):
    """Expand an interpolation specification to a per-point law vector.

    The specification may be a single law string, a list with one law
    string per mesh point, or an ENDF-style region list of dictionaries
    with keys `last_index` (0-based index of the last mesh point of the
    region) and `law`. The returned vector contains for each mesh point
    the law governing the segment between this point and the next one;
    the entry of the last point repeats the law of the last region.
    """
    if num_points < 2:
        raise IndexError('mesh must contain at least two points')
    if isinstance(interp, str):
        if interp not in INTERP_LAWS:
            raise ValueError(f'invalid interpolation law `{interp}`')
        return [interp] * num_points
    interp = list(interp)
    if len(interp) == 0:
        raise IndexError('empty interpolation specification')
    if all(isinstance(el, str) for el in interp):
        if len(interp) != num_points:
            raise IndexError(
                'per-point interpolation vector must have as many ' +
                'elements as the energy mesh'
            )
        for law in interp:
            if law not in INTERP_LAWS:
                raise ValueError(f'invalid interpolation law `{law}`')
        return interp
    if not all(isinstance(el, dict) for el in interp):
        raise TypeError(
            'interpolation specification must be a law string, a list ' +
            'of law strings or a list of regions given as dictionaries ' +
            'with keys `last_index` and `law`'
        )
    expanded = []
    start = 0
    for region in interp:
        last_index = int(region['last_index'])
        law = region['law']
        if law not in INTERP_LAWS:
            raise ValueError(f'invalid interpolation law `{law}`')
        if last_index <= start:
            raise ValueError(
                '`last_index` of interpolation regions must be ' +
                'strictly increasing and each region must cover ' +
                'at least one segment'
            )
        # under the left-point convention the region covers the
        # segments whose left mesh point index is in [start, last_index-1];
        # the segment starting at last_index belongs to the next region
        expanded.extend([law] * (last_index - start))
        start = last_index
    if start != num_points - 1:
        raise ValueError(
            '`last_index` of the final interpolation region must equal ' +
            'the index of the last mesh point'
        )
    # by convention, the entry of the last mesh point (which does not
    # govern any segment) repeats the law of the last region
    expanded.append(expanded[-1])
    return expanded


def _validate_priorblock(priorblock):
    energies = np.array(priorblock['energies'], dtype=np.float64)
    values = np.array(priorblock['values'], dtype=np.float64)
    if len(energies) != len(values):
        raise IndexError('`energies` and `values` must be of same length')
    if np.any(np.diff(energies) <= 0):
        raise ValueError('`energies` must be strictly increasing')
    interp = np.array(
        expand_interpolation_spec(priorblock['interpolation'], len(energies))
    )
    seg_laws = interp[:-1]
    xlog_segs = np.isin(seg_laws, ('log-lin', 'log-log'))
    ylog_segs = np.isin(seg_laws, ('lin-log', 'log-log'))
    if np.any(xlog_segs & (energies[:-1] <= 0)):
        raise ValueError(
            'energies must be positive on segments with ' +
            'logarithmic energy interpolation'
        )
    ylog_points = np.zeros(len(values), dtype=bool)
    ylog_points[:-1] |= ylog_segs
    ylog_points[1:] |= ylog_segs
    if np.any(ylog_points & (values <= 0)):
        raise ValueError(
            'spectrum values must be positive on segments with ' +
            'logarithmic value interpolation'
        )


def get_priorblock_identifier(priorblock):
    return int(9999)


def get_nodename(priorblock):
    return 'fis'


def get_quantity_type(priorblock):
    return int(9999)


def get_energies(priorblock):
    return np.array(priorblock['energies'], dtype=np.float64, copy=True)


def get_values(priorblock):
    return np.array(priorblock['values'], dtype=np.float64, copy=True)


def get_uncertainties(priorblock):
    n = len(priorblock['energies'])
    if 'uncertainties' not in priorblock:
        return np.full(n, 0., dtype=np.float64)
    uncs = np.array(priorblock['uncertainties'], dtype=np.float64, copy=True)
    if len(uncs) != n:
        raise IndexError(
            '`uncertainties` must be of same length as `energies`'
        )
    if np.any(uncs < 0):
        raise ValueError('`uncertainties` must be non-negative')
    return uncs


def get_correlation_matrix(priorblock):
    if 'correlations' not in priorblock:
        return None
    if 'uncertainties' not in priorblock:
        raise ValueError(
            '`correlations` given but `uncertainties` missing'
        )
    n = len(priorblock['energies'])
    cormat = np.array(priorblock['correlations'], dtype=np.float64)
    if cormat.shape != (n, n):
        raise IndexError(
            '`correlations` must be a square matrix matching ' +
            'the length of `energies`'
        )
    if not np.allclose(cormat, cormat.T):
        raise ValueError('`correlations` must be symmetric')
    if not np.allclose(np.diag(cormat), 1.):
        raise ValueError('diagonal of `correlations` must be one')
    return cormat


def get_interpolation(priorblock):
    _validate_priorblock(priorblock)
    return expand_interpolation_spec(
        priorblock['interpolation'], len(priorblock['energies'])
    )


def get_description(priorblock):
    return 'pointwise fission spectrum'
