import os
import tensorflow as tf
import dill


def load_objects(filename, *varnames):
    with open(filename, 'rb') as f:
        objdic = dill.load(f)
    return tuple(objdic[v] for v in varnames)


def save_objects(filename, scope, *varnames):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    objdic = {v: scope[v] for v in varnames}
    with open(filename, 'wb') as f:
        dill.dump(objdic, f)


def make_positive_definite(mat, nugget=1e-10):
    tmp = mat
    tmp = (tmp + tf.transpose(tmp)) / 2.
    tmp = tf.linalg.eigh(tmp)
    corr_eigvals = tf.maximum(tmp[0], nugget)
    tmp = tf.matmul(
        tmp[1], (tf.reshape(corr_eigvals, (-1, 1)) * tf.transpose(tmp[1]))
    )
    tmp = 0.5 * (tmp + tf.transpose(tmp))
    return tmp


def invert_symmetric_matrix(mat):
    invmat = tf.linalg.inv(mat)
    invmat = 0.5 * (invmat + tf.transpose(invmat))
    return invmat
