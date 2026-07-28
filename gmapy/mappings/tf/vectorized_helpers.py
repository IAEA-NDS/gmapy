import numpy as np


def concat_coo_triplets(triplet_list):
    rows = np.concatenate([t[0] for t in triplet_list])
    cols = np.concatenate([t[1] for t in triplet_list])
    vals = np.concatenate([t[2] for t in triplet_list])
    return rows, cols, vals


def row_pair_pattern(x_rows, x_cols, x_vals, y_rows, y_cols, y_vals):
    """Cartesian pairing of same-row entries of two COO matrices.

    For all index pairs (j, j') with x_rows[j] == y_rows[j'] return
    (rows, out_k, out_l, base) with rows the shared row index,
    out_k = x_cols[j], out_l = y_cols[j'] and base = x_vals[j] *
    y_vals[j'], i.e. the constant part of sum_i c_i * X_i^T Y_i over
    outer products of matrix rows; the y triplets must be sorted by
    row. Used to precompute the sparsity pattern of expressions like
    X^T diag(c) Y.
    """
    num_rows = max(x_rows.max(), y_rows.max()) + 1 if len(x_rows) else 0
    y_counts = np.bincount(y_rows, minlength=num_rows)
    y_offsets = np.concatenate([[0], np.cumsum(y_counts)])
    rep = y_counts[x_rows]
    idx_x = np.repeat(np.arange(len(x_rows)), rep)
    ends = np.cumsum(rep)
    starts = ends - rep
    total = ends[-1] if len(ends) else 0
    within = np.arange(total) - np.repeat(starts, rep)
    idx_y = y_offsets[x_rows[idx_x]] + within
    rows = x_rows[idx_x]
    out_k = x_cols[idx_x]
    out_l = y_cols[idx_y]
    base = x_vals[idx_x] * y_vals[idx_y]
    return rows, out_k, out_l, base


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
