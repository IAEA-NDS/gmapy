import numpy as np


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
    if len(src_en) < 2:
        # PiecewiseLinearInterpolation has a special branch for a
        # 1-point mesh that yields 1 (sic) for non-matching target
        # points; refuse instead of replicating it
        raise NotImplementedError('source mesh must have >= 2 points')
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
