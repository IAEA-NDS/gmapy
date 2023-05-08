import tensorflow as tf


class PiecewiseLinearInterpolation(tf.Module):
    def __init__(self, xin, xout, **kwargs):
        super().__init__(**kwargs)
        xin = tf.constant(xin, dtype=tf.float64)
        xout = tf.constant(xout, dtype=tf.float64)
        self.sorted_indices = tf.argsort(xin, axis=-1)
        self.xin = tf.gather(xin, self.sorted_indices, axis=-1)
        self.xout = tf.constant(xout, dtype=tf.float64)

    def __call__(self, inputs):
        xout = self.xout
        xin = self.xin
        yin = tf.gather(inputs, self.sorted_indices, axis=-1)
        if tf.size(xin) == 1:
            zeros = tf.zeros(inputs.shape, dtype=tf.float64)
            yint = tf.where(xout != xin, zeros, yin)
            return yint
        idcs = tf.searchsorted(xin, xout, side='right')
        idcs = idcs - 1
        idcs = tf.maximum(0, idcs)
        idcs = tf.minimum(idcs, xin.shape[0]-2)
        x0 = tf.gather(xin, idcs, axis=-1)
        x1 = tf.gather(xin, idcs + 1, axis=-1)
        y0 = tf.gather(yin, idcs, axis=-1)
        y1 = tf.gather(yin, idcs + 1, axis=-1)
        slopes = (y1 - y0) / (x1 - x0)
        yint = y0 + slopes * (xout-x0)
        zero_mask = tf.logical_or(xout < xin[0], xout > xin[-1])
        yint = tf.where(zero_mask, tf.zeros_like(yint), yint)
        return yint


class InputSelector(tf.Module):
    def __init__(self, idcs):
        super().__init__()
        self._idcs = tf.constant(idcs, dtype=tf.int32)

    def get_indices(self):
        return self._idcs.copy()

    def __call__(self, inputs):
        return tf.gather(inputs, self._idcs, axis=-1)


class Distributor(tf.Module):
    def __init__(self, idcs, tar_len):
        super().__init__()
        self._idcs = tf.constant(idcs, dtype=tf.int32)
        self._idcs = tf.expand_dims(self._idcs, axis=-1)
        self._tar_len = tf.constant(tar_len, dtype=tf.int32)

    def get_indices(self):
        return self._idcs.copy()

    def __call__(self, inputs):
        res = tf.scatter_nd(self._idcs, inputs, (self._tar_len,))
        return res
