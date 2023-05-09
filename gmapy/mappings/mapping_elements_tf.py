import numpy as np
import tensorflow as tf


class PiecewiseLinearInterpolation(tf.Module):
    def __init__(self, xin, xout, **kwargs):
        super().__init__(**kwargs)
        xin = tf.constant(xin, dtype=tf.float64)
        xout = tf.constant(xout, dtype=tf.float64)
        self.sorted_indices = tf.argsort(xin, axis=-1)
        self.xin = tf.gather(xin, self.sorted_indices, axis=-1)
        self.xout = xout

    def __call__(self, inputs):
        xout = self.xout
        xin = self.xin
        yin = tf.gather(inputs, self.sorted_indices, axis=-1)
        if tf.size(xin) == 1:
            zero = tf.constant((1,), dtype=tf.float64)
            yint = tf.where(xout != xin[0], zero, yin[0])
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


class InputSelectorCollection:

    def __init__(self, listlike=None):
        if listlike is None:
            listlike = []
        self._selector_list = []
        self.add_selectors(listlike)

    def get_indices(self):
        return np.unique(np.concatenate(list(
            obj.get_indices() for obj in self._selector_list
        )))

    def get_selectors(self):
        return self._selector_list

    def add_selectors(self, listlike=None):
        for sel in listlike:
            self.add_selector(sel)

    def add_selector(self, selector):
        if type(selector) != InputSelector:
            raise TypeError('only InputSelector instance allowed')
        selids = {id(sel) for sel in self._selector_list}
        if id(selector) not in selids:
            self._selector_list.append(selector)

    def define_selector(self, idcs):
        for sel in self._selector_list:
            refidcs = sel.get_indices()
            if len(idcs) == len(refidcs):
                if np.all(idcs == refidcs):
                    return sel
        newsel = InputSelector(idcs)
        self._selector_list.append(newsel)
        return newsel


class InputSelector(tf.Module):
    def __init__(self, idcs):
        super().__init__()
        self._idcs = idcs

    def get_indices(self):
        return self._idcs.copy()

    def __call__(self, inputs):
        idcs = tf.constant(self._idcs, dtype=tf.int32)
        return tf.gather(inputs, idcs, axis=-1)


class Distributor(tf.Module):
    def __init__(self, idcs, tar_len):
        super().__init__()
        self._idcs = idcs
        self._tar_len = tar_len

    def get_indices(self):
        return self._idcs

    def __call__(self, inputs):
        idcs = tf.constant(self._idcs, dtype=tf.int32)
        idcs = tf.expand_dims(idcs, axis=-1)
        res = tf.scatter_nd(idcs, inputs, (self._tar_len,))
        return res
