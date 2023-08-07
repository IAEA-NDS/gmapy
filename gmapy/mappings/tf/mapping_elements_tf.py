import numpy as np
import tensorflow as tf


class PiecewiseLinearInterpolation(tf.Module):
    def __init__(self, xin, xout, **kwargs):
        super().__init__(**kwargs)
        self._num_xin = len(xin)
        xin = tf.constant(xin, dtype=tf.float64)
        xout = tf.constant(xout, dtype=tf.float64)
        self.sorted_indices = tf.argsort(xin, axis=-1)
        self.xin = tf.gather(xin, self.sorted_indices, axis=-1)
        self.xout = xout

    def __call__(self, inputs):
        xout = self.xout
        xin = self.xin
        yin = tf.gather(inputs, self.sorted_indices, axis=-1)
        if self._num_xin == 1:
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


class IntegralLinLin(tf.Module):
    def __init__(self, xin, **kwargs):
        super().__init__(**kwargs)
        sxin = np.sort(xin)
        xdiff = np.diff(sxin)
        xout = sxin[:-1] + xdiff / 2
        self._xdiff = tf.constant(xdiff, dtype=tf.float64)
        self._pwlinint = PiecewiseLinearInterpolation(xin, xout)

    def __call__(self, inputs):
        pwlinint = self._pwlinint
        yint = pwlinint(inputs)
        intres = yint * self._xdiff
        intres = tf.reduce_sum(intres, axis=-1, keepdims=True)
        return intres


class IntegralOfProductLinLin(tf.Module):
    def __init__(self, xin1, xin2):
        xmin = max(min(xin1), min(xin2))
        xmax = min(max(xin1), max(xin2))
        xm = np.unique(np.concatenate([xin1, xin2]))
        xm = xm[(xm >= xmin) & (xm <= xmax)]
        self._xm = xm
        self._pwlinint1 = PiecewiseLinearInterpolation(xin1, xm)
        self._pwlinint2 = PiecewiseLinearInterpolation(xin2, xm)

    def __call__(self, inputs1, inputs2):
        x1 = tf.constant(self._xm[:-1], dtype=tf.float64)
        x2 = tf.constant(self._xm[1:], dtype=tf.float64)
        yam = self._pwlinint1(inputs1)
        ybm = self._pwlinint2(inputs2)
        ya1 = yam[:-1]
        ya2 = yam[1:]
        yb1 = ybm[:-1]
        yb2 = ybm[1:]
        d = x2 - x1
        ca1 = (x2*ya1 - x1*ya2) / d
        ca2 = (-1*ya1 + 1*ya2) / d
        cb1 = (x2*yb1 - x1*yb2) / d
        cb2 = (-1*yb1 + 1*yb2) / d
        xp1 = x1
        xp2 = x2
        p1 = ca1*cb1*(xp2-xp1)
        xp1 = xp1 * x1
        xp2 = xp2 * x2
        p2 = (ca2*cb1 + ca1*cb2)*(xp2 - xp1)/2
        xp1 = xp1 * x1
        xp2 = xp2 * x2
        p3 = ca2*cb2*(xp2 - xp1)/3
        intres = tf.reduce_sum(p1 + p2 + p3, axis=-1, keepdims=True)
        return intres


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
