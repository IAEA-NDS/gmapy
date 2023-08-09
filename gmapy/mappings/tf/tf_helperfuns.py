import tensorflow as tf


def subset_sparse_matrix(spmat, row_idcs):
    # prepare the index map
    min_idx = tf.reduce_min(row_idcs)
    max_idx = tf.reduce_max(row_idcs)
    lgth = max_idx - min_idx + 1
    idcs_map = tf.fill([lgth], tf.constant(-1, tf.int64))
    shifted_row_idcs = row_idcs - min_idx
    idcs_map = tf.tensor_scatter_nd_update(
        idcs_map,
        tf.reshape(shifted_row_idcs, (-1, 1)),
        tf.range(tf.size(shifted_row_idcs), dtype=tf.int64)
    )
    # remove indices being beyond the range of row_idcs
    orig_row_idcs = tf.slice(spmat.indices, [0, 0], [-1, 1])
    orig_col_idcs = tf.slice(spmat.indices, [0, 1], [-1, 1])
    orig_values = spmat.values
    mask = tf.logical_and(
        tf.greater_equal(orig_row_idcs, min_idx),
        tf.less_equal(orig_row_idcs, max_idx)
    )
    orig_row_idcs = tf.boolean_mask(orig_row_idcs, mask)
    orig_col_idcs = tf.boolean_mask(orig_col_idcs, mask)
    orig_values = tf.boolean_mask(orig_values, tf.reshape(mask, (-1,)))
    # now remove the row indices not part of the result
    orig_shifted_row_idcs = orig_row_idcs - min_idx
    mask = tf.greater_equal(
        tf.gather_nd(idcs_map, tf.reshape(orig_shifted_row_idcs, (-1, 1))),
        tf.constant(0, dtype=tf.int64)
    )
    orig_shifted_row_idcs = tf.boolean_mask(orig_shifted_row_idcs, mask)
    orig_row_idcs = tf.boolean_mask(orig_row_idcs, mask)
    orig_col_idcs = tf.boolean_mask(orig_col_idcs, mask)
    orig_values = tf.boolean_mask(orig_values, mask)
    # reorder the row indices so that they match the order in row_idcs
    new_row_idcs = tf.gather_nd(
        idcs_map, tf.reshape(orig_shifted_row_idcs, (-1, 1))
    )
    # assemble final sparse tensor
    new_spmat = tf.sparse.SparseTensor(
        indices=tf.stack((new_row_idcs, orig_col_idcs), axis=1),
        values=orig_values,
        dense_shape=(tf.size(row_idcs), spmat.dense_shape[1])
    )
    new_spmat = tf.sparse.reorder(new_spmat)
    return new_spmat


def scatter_sparse_matrix(spmat, row_idcs, col_idcs, shape):
    if col_idcs is None:
        col_idcs_tf = tf.range(spmat.dense_shape[1], dtype=tf.int64)
    else:
        if isinstance(col_idcs, tf.Tensor):
            col_idcs_tf = col_idcs
        else:
            col_idcs_tf = tf.constant(col_idcs, dtype=tf.int64)
    col_slc = tf.slice(spmat.indices, [0, 1], [-1, 1])
    s = tf.gather(col_idcs_tf, col_slc)
    if row_idcs is None:
        row_idcs_tf = tf.range(spmat.dense_shape[0], dtype=tf.int64)
    else:
        if isinstance(row_idcs, tf.Tensor):
            row_idcs_tf = row_idcs
        else:
            row_idcs_tf = tf.constant(row_idcs, dtype=tf.int64)
    row_slc = tf.slice(spmat.indices, [0, 0], [-1, 1])
    t = tf.gather(row_idcs_tf, row_slc)
    z = tf.concat((t, s), axis=1)
    newmat = tf.sparse.SparseTensor(
        indices=z,
        values=spmat.values,
        dense_shape=shape
    )
    return newmat
