import numpy as np


def concat_coo_triplets(triplet_list):
    rows = np.concatenate([t[0] for t in triplet_list])
    cols = np.concatenate([t[1] for t in triplet_list])
    vals = np.concatenate([t[2] for t in triplet_list])
    return rows, cols, vals


def piecewise_linear_interp_matrix(src_en, tar_en):
    """COO triplets of the linear map equivalent to PiecewiseLinearInterpolation.

    Returns (rows, cols, vals) with rows referring to positions in
    `tar_en` and cols to positions in the (unsorted) `src_en`, such
    that scattering `vals` into a matrix S gives
    S @ y == PiecewiseLinearInterpolation(src_en, tar_en)(y)
    for any vector y living on `src_en`. Target points outside the
    source mesh get an empty row (value zero).
    """
    src_en = np.asarray(src_en, dtype=np.float64)
    tar_en = np.asarray(tar_en, dtype=np.float64)
    if len(src_en) == 0:
        raise IndexError('source mesh must not be empty')
    if len(src_en) == 1:
        # single-point meshes (thermal constants) map target points
        # equal to the mesh point to its value; for other target
        # points PiecewiseLinearInterpolation yields the constant 1
        # (sic), which no linear operator can represent -- that case
        # does not occur in the GMA databases, so refuse it
        if np.any(tar_en != src_en[0]):
            raise NotImplementedError(
                'target points not matching a single-point source '
                'mesh are not supported'
            )
        rows = np.arange(len(tar_en))
        cols = np.zeros(len(tar_en), dtype=np.int64)
        vals = np.ones(len(tar_en))
        return rows, cols, vals
    sort_idcs = np.argsort(src_en, kind='stable')
    sxin = src_en[sort_idcs]
    idcs = np.searchsorted(sxin, tar_en, side='right') - 1
    idcs = np.clip(idcs, 0, len(sxin) - 2)
    x0 = sxin[idcs]
    x1 = sxin[idcs + 1]
    w = (tar_en - x0) / (x1 - x0)
    inside = (tar_en >= sxin[0]) & (tar_en <= sxin[-1])
    rows = np.arange(len(tar_en))
    rows = np.concatenate([rows[inside], rows[inside]])
    cols = np.concatenate([sort_idcs[idcs[inside]],
                           sort_idcs[idcs[inside] + 1]])
    vals = np.concatenate([1. - w[inside], w[inside]])
    nonzero = vals != 0.
    return rows[nonzero], cols[nonzero], vals[nonzero]
