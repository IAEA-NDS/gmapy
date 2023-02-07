import numpy as np
from scipy.sparse import csr_matrix, vstack
from .basic_maps import get_basic_sensmat
from .basic_integral_maps import (
    basic_integral_propagate,
    get_basic_integral_sensmat,
    basic_integral_of_product_propagate,
    get_basic_integral_of_product_sensmats
)
from .helperfuns import numeric_jacobian
from ..legacy.legacy_maps import (
    propagate_fisavg,
    get_sensmat_fisavg,
    get_sensmat_fisavg_corrected
)


class MyAlgebra:

    def __add__(self, other):
        return Addition(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other):
        return Multiplication(self, other)

    def __truediv__(self, other):
        return Division(self, other)


class Selector(MyAlgebra):

    def __init__(self, idcs, size):
        self.__idcs = np.array(idcs)
        self.__size = size
        self.__values = None

    def __len__(self):
        return len(self.__idcs)

    def evaluate(self):
        if self.__values is None:
            raise ValueError('please assign numbers')
        return self.__values

    def jacobian(self):
        tar_size = len(self.__idcs)
        coeffs = np.ones(tar_size)
        tar_idcs = np.arange(tar_size)
        return csr_matrix(
            (coeffs, (tar_idcs, self.__idcs)),
            shape=(tar_size, self.__size), dtype=float
        )

    def assign(self, arraylike):
        if len(arraylike) != self.__size:
            raise IndexError('wrong length of vector')
        self.__values = np.array(arraylike)[self.__idcs]


class SelectorCollection:

    def __init__(self, listlike):
        if not all(type(obj) == Selector for obj in listlike):
            raise TypeError('only Selector instances allowed in list')
        self.__selector_list = listlike

    def assign(self, arraylike):
        for obj in self.__selector_list:
            obj.assign(arraylike)


class Const(MyAlgebra):

    def __init__(self, values):
        self.__values = np.array(values)
        self.__shape = (len(values),)*2

    def __len__(self):
        return len(self.__values)

    def evaluate(self):
        return self.__values

    def jacobian(self):
        return 0.


class Distributor(MyAlgebra):

    def __init__(self, obj, idcs, size):
        if len(obj) != len(idcs):
            raise ValueError('size mismatch')
        self.__idcs = np.array(idcs)
        self.__size = size
        self.__obj = obj
        # construct distribution matrix
        src_len = len(self.__obj)
        src_idcs = np.arange(src_len)
        tar_idcs = self.__idcs
        coeffs = np.ones(src_len, dtype=float)
        self.__dist_mat = csr_matrix(
            (coeffs, (tar_idcs, src_idcs)),
            shape=(self.__size, src_len), dtype=float
        )

    def __len__(self):
        return self.__size

    def evaluate(self):
        res = np.zeros(self.__size, dtype=float)
        res[self.__idcs] = self.__obj.evaluate()
        return res

    def jacobian(self):
        return self.__dist_mat @ self.__obj.jacobian()


class Replicator(MyAlgebra):

    def __init__(self, obj, num):
        self.__num = num
        self.__obj = obj

    def __len__(self):
        return len(self.__obj) * self.__num

    def evaluate(self):
        return np.repeat(
            self.__obj.evaluate().reshape(1, -1),
            self.__num, axis=0
        ).flatten()

    def jacobian(self):
        return vstack(
            [self.__obj.jacobian()] * self.__num, format='csr'
        )


class Addition(MyAlgebra):

    def __init__(self, obj1, obj2):
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        return self.__obj1.evaluate() + self.__obj2.evaluate()

    def jacobian(self):
        return self.__obj1.jacobian() + self.__obj2.jacobian()

    def assign(self, arraylike):
        if len(arraylike) != self.__size:
            raise IndexError('wrong length of vector')
        if type(self.__obj1) != Selector or type(self.__obj2) != Selector:
            raise TypeError(
                'assignment only supported for addition of Selector'
            )
        self.__obj1.assign(arraylike)
        self.__obj2.assign(arraylike)


class Multiplication(MyAlgebra):

    def __init__(self, obj1, obj2):
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        return self.__obj1.evaluate() * self.__obj2.evaluate()

    def jacobian(self):
        vals1 = self.__obj1.evaluate().reshape(-1, 1)
        vals2 = self.__obj2.evaluate().reshape(-1, 1)
        S1 = self.__obj1.jacobian().multiply(vals2)
        S2 = self.__obj2.jacobian().multiply(vals1)
        return S1 + S2


class Division(MyAlgebra):

    def __init__(self, obj1, obj2):
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        return self.__obj1.evaluate() / self.__obj2.evaluate()

    def jacobian(self):
        v1 = self.__obj1.evaluate().reshape(-1, 1)
        v2_inv = 1.0 / self.__obj2.evaluate().reshape(-1, 1)
        S1 = self.__obj1.jacobian().multiply(v2_inv)
        S2 = self.__obj2.jacobian().multiply(-v1 * np.square(v2_inv))
        return S1 + S2


class LinearInterpolation(MyAlgebra):

    def __init__(self, obj, src_x, tar_x, zero_outside=False):
        if len(obj) != len(src_x):
            raise ValueError('length mismatch')
        self.__obj = obj
        yzeros = np.zeros(len(src_x), dtype=float)
        self.__jacobian = get_basic_sensmat(
            src_x, yzeros, tar_x, 'lin-lin', zero_outside
        )

    def __len__(self):
        return self.__jacobian.shape[0]

    def evaluate(self):
        return (self.__jacobian @ self.__obj.evaluate()).flatten()

    def jacobian(self):
        return self.__jacobian @ self.__obj.jacobian()


class Integral(MyAlgebra):

    def __init__(self, obj, xvals, interp_type, **kwargs):
        if not isinstance(obj, MyAlgebra):
            raise TypeError('obj must be of class MyAlgebra')
        self.__obj = obj
        self.__xvals = np.array(xvals)
        self.__interp_type = interp_type
        self.__kwargs = kwargs

    def __len__(self):
        return 1

    def evaluate(self):
        yvals = self.__obj.evaluate()
        ret = basic_integral_propagate(
            self.__xvals, yvals, self.__interp_type, **self.__kwargs
        )
        return np.array([ret])

    def jacobian(self):
        yvals = self.__obj.evaluate()
        outer_jac = csr_matrix(get_basic_integral_sensmat(
            self.__xvals, yvals, self.__interp_type, **self.__kwargs
        ))
        inner_jac = self.__obj.jacobian()
        return outer_jac @ inner_jac


class IntegralOfProduct(MyAlgebra):

    def __init__(self, obj_list, xlist, interplist,
                 zero_outside=False, **kwargs):
        if not all(isinstance(obj, MyAlgebra) for obj in obj_list):
            raise TypeError('all objects in obj_list must be of type ' +
                            'obj_list')
        if len(obj_list) != len(xlist):
            raise IndexError('length of obj_list must equal length ' +
                             'of xlist')
        if len(interplist) != len(obj_list):
            raise IndexError('length of interp_list must equal ' +
                             'length of obj_list')
        self.__obj_list = obj_list
        self.__xlist = [np.array(xv) for xv in xlist]
        self.__interplist = interplist
        self.__zero_outside = zero_outside
        self.__kwargs = kwargs

    def __len__(self):
        return 1

    def evaluate(self):
        ylist = [obj.evaluate() for obj in self.__obj_list]
        return basic_integral_of_product_propagate(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )

    def jacobian(self):
        ylist = [obj.evaluate() for obj in self.__obj_list]
        outer_jacs = get_basic_integral_of_product_sensmats(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )
        outer_jacs = [csr_matrix(mat) for mat in outer_jacs]
        inner_jacs = [obj.jacobian() for obj in self.__obj_list]
        jac = 0.
        for outer_jac, inner_jac in zip(outer_jacs, inner_jacs):
            jac += outer_jac @ inner_jac
        return jac


class LegacyFissionAverage(MyAlgebra):

    def __init__(self, en, xsobj, fisen, fisobj, fix_jacobian=True):
        if not isinstance(xsobj, MyAlgebra):
            raise TypeError('xsobj must be of class MyAlgebra')
        if not isinstance(fisobj, MyAlgebra):
            raise TypeError('fisobj must be of class MyAlgebra')
        self.__xsobj = xsobj
        self.__en = en
        self.__fisobj = fisobj
        self.__fisen = fisen
        self.__fix_jacobian = fix_jacobian

    def __len__(self):
        return 1

    def evaluate(self):
        xs = self.__xsobj.evaluate()
        fisvals = self.__fisobj.evaluate()
        ret = propagate_fisavg(self.__en, xs, self.__fisen, fisvals)
        assert type(ret) == float
        return np.array([ret])

    def jacobian(self):
        xs = self.__xsobj.evaluate()
        fisvals = self.__fisobj.evaluate()
        xsjac = self.__xsobj.jacobian()
        if self.__fix_jacobian:
            sensvec = get_sensmat_fisavg_corrected(
                self.__en, xs, self.__fisen, fisvals
            )
        else:
            sensvec = get_sensmat_fisavg(
                self.__en, xs, self.__fisen, fisvals
            )
        out_jac = csr_matrix(sensvec.reshape(1, -1))
        return out_jac @ xsjac


def FissionAverage(MyAlgebra):

    def __init__(self, en, xsobj, fisen, fisobj,
                 legacy=False, fix_jacobian=True):
        if legacy:
            self.__obj = LegacyFissionAverage(
                en, xsobj, fisen, fisobj, fix_jacobian
            )
        else:
            self.__obj = IntegralOfProduct(
                [xsobj, fisobj], [en, fisen], ['lin-lin', 'lin-lin'],
                zero_outside=False, maxord=16, rtol=1e-6
            )

    def evaluate(self):
        return self.__obj.evaluate()

    def jacobian(self):
        return self.__obj.jacobian()
