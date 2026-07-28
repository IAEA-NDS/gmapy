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


def _prepare_interp_vector(xin, interp):
    """Sort mesh and per-point law vector, check laws are supported."""
    xin = np.array(xin, dtype=np.float64)
    if len(xin) < 2:
        raise IndexError('mesh must contain at least two points')
    if isinstance(interp, str):
        interp = np.full(len(xin), interp)
    interp = np.array(interp)
    if len(interp) != len(xin):
        raise IndexError(
            'per-point interpolation vector must have as many ' +
            'elements as the mesh'
        )
    sort_idcs = np.argsort(xin)
    sxin = xin[sort_idcs]
    sinterp = interp[sort_idcs]
    unsupported = set(sinterp) - {'lin-lin', 'log-log'}
    if unsupported:
        raise NotImplementedError(
            f'interpolation laws {unsupported} not implemented ' +
            'in the tensorflow-based mappings'
        )
    return sxin, sinterp, sort_idcs


def _stable_expm1_over_x(t):
    # expm1(t)/t with the limit value 1 at t=0; both branches of
    # tf.where receive finite arguments so that no NaN enters the
    # gradient computation
    small = tf.abs(t) < 1e-8
    safe_t = tf.where(small, tf.ones_like(t), t)
    return tf.where(small, 1. + t/2. + t*t/6., tf.math.expm1(safe_t)/safe_t)


def _segment_laws(sxin, sinterp, xvals):
    # law of the segment of the mesh sxin that contains xvals[j],
    # associated with xvals[j] by left-point convention
    idcs = np.searchsorted(sxin, xvals, side='right') - 1
    idcs = np.clip(idcs, 0, len(sxin) - 2)
    return sinterp[idcs], idcs


class PiecewiseInterpolation(tf.Module):
    """Interpolation with per-segment laws ('lin-lin' or 'log-log')."""

    def __init__(self, xin, xout, interp='lin-lin', **kwargs):
        super().__init__(**kwargs)
        sxin, sinterp, sort_idcs = _prepare_interp_vector(xin, interp)
        xout = np.array(xout, dtype=np.float64)
        seg_laws, idcs = _segment_laws(sxin, sinterp, xout)
        is_loglog = (seg_laws == 'log-log')
        x0 = sxin[idcs]
        x1 = sxin[idcs + 1]
        wlin = (xout - x0) / (x1 - x0)
        with np.errstate(divide='ignore', invalid='ignore'):
            wlog = np.log(xout / x0) / np.log(x1 / x0)
        wlog = np.where(is_loglog & np.isfinite(wlog), wlog, 0.5)
        inside = (xout >= sxin[0]) & (xout <= sxin[-1])
        self._sort_idcs = tf.constant(sort_idcs, dtype=tf.int32)
        self._idcs = tf.constant(idcs, dtype=tf.int32)
        self._wlin = tf.constant(wlin, dtype=tf.float64)
        self._wlog = tf.constant(wlog, dtype=tf.float64)
        self._is_loglog = tf.constant(is_loglog)
        self._inside = tf.constant(inside)

    def __call__(self, inputs):
        yin = tf.gather(inputs, self._sort_idcs)
        y0 = tf.gather(yin, self._idcs)
        y1 = tf.gather(yin, self._idcs + 1)
        ylin = y0 + self._wlin * (y1 - y0)
        ones = tf.ones_like(y0)
        sy0 = tf.where(self._is_loglog, y0, ones)
        sy1 = tf.where(self._is_loglog, y1, ones)
        ylog = tf.exp((1. - self._wlog) * tf.math.log(sy0) +
                      self._wlog * tf.math.log(sy1))
        yint = tf.where(self._is_loglog, ylog, ylin)
        yint = tf.where(self._inside, yint, tf.zeros_like(yint))
        return yint


class IntegralInterp(tf.Module):
    """Integral of a function with per-segment interpolation laws."""

    def __init__(self, xin, interp='lin-lin', **kwargs):
        super().__init__(**kwargs)
        sxin, sinterp, sort_idcs = _prepare_interp_vector(xin, interp)
        is_loglog = (sinterp[:-1] == 'log-log')
        x1 = sxin[:-1]
        x2 = sxin[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            lnr = np.log(x2 / x1)
        lnr = np.where(is_loglog & np.isfinite(lnr), lnr, 1.)
        self._sort_idcs = tf.constant(sort_idcs, dtype=tf.int32)
        self._x1 = tf.constant(x1, dtype=tf.float64)
        self._xdiff = tf.constant(x2 - x1, dtype=tf.float64)
        self._lnr = tf.constant(lnr, dtype=tf.float64)
        self._is_loglog = tf.constant(is_loglog)

    def __call__(self, inputs):
        yin = tf.gather(inputs, self._sort_idcs)
        y1 = yin[:-1]
        y2 = yin[1:]
        lin_seg = 0.5 * (y1 + y2) * self._xdiff
        ones = tf.ones_like(y1)
        sy1 = tf.where(self._is_loglog, y1, ones)
        sy2 = tf.where(self._is_loglog, y2, ones)
        # segment with y = c*x^p: integral is y1*x1*lnr*g((p+1)*lnr)
        # with g(t) = expm1(t)/t and lnr = log(x2/x1)
        p = (tf.math.log(sy2) - tf.math.log(sy1)) / self._lnr
        t = (p + 1.) * self._lnr
        log_seg = sy1 * self._x1 * self._lnr * _stable_expm1_over_x(t)
        segvals = tf.where(self._is_loglog, log_seg, lin_seg)
        intres = tf.reduce_sum(segvals, axis=-1, keepdims=True)
        return intres


class IntegralOfProductInterp(tf.Module):
    """Integral of product of a lin-lin function and a function with laws.

    The first mesh belongs to a lin-lin interpolated function, the
    second one to a function with per-segment interpolation laws
    ('lin-lin' or 'log-log').
    """

    def __init__(self, xin1, xin2, interp2='lin-lin', **kwargs):
        super().__init__(**kwargs)
        sxin2, sinterp2, _ = _prepare_interp_vector(xin2, interp2)
        xmin = max(min(xin1), min(xin2))
        xmax = min(max(xin1), max(xin2))
        xm = np.unique(np.concatenate([xin1, xin2]))
        xm = xm[(xm >= xmin) & (xm <= xmax)]
        seg_laws, _ = _segment_laws(sxin2, sinterp2, xm[:-1])
        is_loglog = (seg_laws == 'log-log')
        x1 = xm[:-1]
        x2 = xm[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            lnr = np.log(x2 / x1)
        lnr = np.where(is_loglog & np.isfinite(lnr), lnr, 1.)
        self._pwlinint1 = PiecewiseLinearInterpolation(xin1, xm)
        self._pwint2 = PiecewiseInterpolation(xin2, xm, interp2)
        self._x1 = tf.constant(x1, dtype=tf.float64)
        self._x2 = tf.constant(x2, dtype=tf.float64)
        self._lnr = tf.constant(lnr, dtype=tf.float64)
        self._is_loglog = tf.constant(is_loglog)

    def __call__(self, inputs1, inputs2):
        x1 = self._x1
        x2 = self._x2
        yam = self._pwlinint1(inputs1)
        ybm = self._pwint2(inputs2)
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
        lin_seg = p1 + p2 + p3
        # segment with xs = a + b*x and spectrum = yb1*(x/x1)^p:
        # integral is yb1*(a*x1*lnr*g((p+1)*lnr) + b*x1^2*lnr*g((p+2)*lnr))
        # with g(t) = expm1(t)/t and lnr = log(x2/x1)
        ones = tf.ones_like(yb1)
        syb1 = tf.where(self._is_loglog, yb1, ones)
        syb2 = tf.where(self._is_loglog, yb2, ones)
        p = (tf.math.log(syb2) - tf.math.log(syb1)) / self._lnr
        t1 = (p + 1.) * self._lnr
        t2 = (p + 2.) * self._lnr
        log_seg = syb1 * self._lnr * (
            ca1 * x1 * _stable_expm1_over_x(t1) +
            ca2 * x1 * x1 * _stable_expm1_over_x(t2)
        )
        segvals = tf.where(self._is_loglog, log_seg, lin_seg)
        intres = tf.reduce_sum(segvals, axis=-1, keepdims=True)
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
