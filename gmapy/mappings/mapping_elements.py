import numpy as np
from scipy.sparse import csr_matrix, vstack
from .basic_maps import get_basic_sensmat
from .basic_integral_maps import (
    basic_integral_propagate,
    get_basic_integral_sensmat,
    basic_integral_of_product_propagate,
    get_basic_integral_of_product_sensmats
)
from ..legacy.legacy_maps import (
    propagate_fisavg,
    get_sensmat_fisavg,
    get_sensmat_fisavg_corrected
)

# the following two helper functions
# to multiply matrices have a special
# case for one matrix being a float number
# which appear if the jacobian method
# of the Const class is called


def matmul(x, y):
    if type(x) == float or type(y) == float:
        return x * y
    else:
        return x @ y


def elem_mul(x, y):
    if type(x) == float and x == 0.0:
        return 0.0
    else:
        return x.multiply(y)

# Another convenience function that allows
# to reuse InputSelectors (see below) if they
# have already been defined


def reuse_or_create_input_selector(idcs, size, selector_list=None):
    if selector_list is not None:
        for cursel in selector_list:
            curidcs = cursel.get_indices()
            if len(idcs) == len(curidcs):
                if np.all(idcs == curidcs):
                    return cursel
    return InputSelector(idcs, size)

# the following classes are the building
# blocks to construct mathematical expressions
# enabling the automated computation of derivatives


class MyAlgebra:

    def __init__(self):
        self._values_updated = True
        self._jacobian_updated = True
        self._obj_list = []

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

    def islinear(self):
        return False

    def evaluate(self):
        self._values_updated = False

    def jacobian(self):
        self._jacobian_updated = False

    def values_updated(self):
        self._values_updated = (
            self._values_updated or
            any(obj.values_updated() for obj in self._obj_list)
        )
        return self._values_updated

    def jacobian_updated(self):
        self._jacobian_updated = (
            self._jacobian_updated or
            any(obj.jacobian_updated() for obj in self._obj_list)
        )
        if not self.islinear():
            self._jacobian_updated = (
                self._jacobian_updated or
                self.values_updated()
            )
        return self._jacobian_updated


class InputSelector(MyAlgebra):

    def __init__(self, idcs, size):
        super().__init__()
        self.__idcs = np.array(idcs)
        self.__size = size
        self.__values = None

    def __len__(self):
        return len(self.__idcs)

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        if self.__values is None:
            raise ValueError('please assign numbers')
        return self.__values

    def jacobian(self):
        super().jacobian()
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
        self._values_updated = True

    def get_indices(self):
        return self.__idcs.copy()


class Selector(MyAlgebra):

    def __init__(self, inpobj, idcs):
        super().__init__()
        if not isinstance(inpobj, MyAlgebra):
            raise TypeError('please provide instance derived from MyAlgebra')
        self.__inpobj = inpobj
        self.__idcs = np.array(idcs)

    def __len__(self):
        return len(self.__idcs)

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        allvals = self.__inpobj.evaluate()
        return allvals[self.__idcs]

    def jacobian(self):
        super().jacobian()
        src_size = len(self.__inpobj)
        tar_size = len(self.__idcs)
        coeffs = np.ones(tar_size)
        tar_idcs = np.arange(tar_size)
        outerS = csr_matrix(
            (coeffs, (tar_idcs, self.__idcs)),
            shape=(tar_size, src_size), dtype=float
        )
        innerS = self.__inpobj.jacobian()
        S = matmul(outerS,  innerS)
        return S


class InputSelectorCollection:

    def __init__(self, listlike=None):
        if listlike is None:
            listlike = []
        self.__selector_list = []
        self.add_selectors(listlike)

    def assign(self, arraylike):
        for obj in self.__selector_list:
            obj.assign(arraylike)

    def get_indices(self):
        return np.unique(np.concatenate(list(
            obj.get_indices() for obj in self.__selector_list
        )))

    def get_selectors(self):
        return self.__selector_list

    def add_selectors(self, listlike=None):
        if listlike is None:
            return
        if not all(type(obj) == InputSelector for obj in listlike):
            raise TypeError('only InputSelector instances allowed in list')
        if self.__selector_list is None:
            breakpoint()
        sels = self.__selector_list + listlike
        uniq_sels = list({id(cursel): cursel for cursel in sels}.values())
        self.__selector_list = uniq_sels


class Const(MyAlgebra):

    def __init__(self, values):
        super().__init__()
        self.__values = np.array(values)
        self.__shape = (len(values),)*2

    def __len__(self):
        return len(self.__values)

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        return self.__values

    def jacobian(self):
        super().jacobian()
        return 0.0


class Distributor(MyAlgebra):

    def __init__(self, obj, idcs, size):
        super().__init__()
        if len(obj) != len(idcs):
            raise ValueError('size mismatch')
        self.__idcs = np.array(idcs)
        self.__size = size
        self.__obj = obj
        self._obj_list = [obj]
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

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        res = np.zeros(self.__size, dtype=float)
        res[self.__idcs] = self.__obj.evaluate()
        return res

    def jacobian(self):
        super().jacobian()
        return matmul(self.__dist_mat, self.__obj.jacobian())

    def get_indices(self):
        return self.__idcs.copy()


class SumOfDistributors(MyAlgebra):

    def __init__(self, listlike=None):
        if listlike is None:
            listlike = []
        self.__distributor_list = []
        self.add_distributors(listlike)

    def get_indices(self):
        return np.unique(np.concatenate(list(
            obj.get_indices() for obj in self.__distributor_list
        )))

    def __len__(self):
        if len(self.__distributor_list) == 0:
            raise IndexError('empty list of distributors')
        return len(self.__distributor_list[0])

    def evaluate(self):
        super().evaluate()
        res = self.__distributor_list[0].evaluate()
        for obj in self.__distributor_list[1:]:
            res += obj.evaluate()
        return res

    def jacobian(self):
        super().jacobian()
        jac = self.__distributor_list[0].jacobian()
        for obj in self.__distributor_list[1:]:
            jac += obj.jacobian()
        return jac

    def get_distributors(self):
        return self.__distributor_list

    def add_distributors(self, listlike):
        if not all(
            type(obj) in (Distributor, SumOfDistributors) for obj in listlike
        ):
            raise TypeError('only Distributor instances allowed in list')
        self.__distributor_list.extend(listlike)


class Replicator(MyAlgebra):

    def __init__(self, obj, num):
        super().__init__()
        self.__num = num
        self.__obj = obj
        self._obj_list = [obj]

    def __len__(self):
        return len(self.__obj) * self.__num

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        return np.repeat(
            self.__obj.evaluate().reshape(1, -1),
            self.__num, axis=0
        ).flatten()

    def jacobian(self):
        super().jacobian()
        return vstack(
            [self.__obj.jacobian()] * self.__num, format='csr'
        )


class Addition(MyAlgebra):

    def __init__(self, obj1, obj2):
        super().__init__()
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2
        self._obj_list = [obj1, obj2]

    def __len__(self):
        return len(self.__obj1)

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        return self.__obj1.evaluate() + self.__obj2.evaluate()

    def jacobian(self):
        super().jacobian()
        return self.__obj1.jacobian() + self.__obj2.jacobian()


class Multiplication(MyAlgebra):

    def __init__(self, obj1, obj2):
        super().__init__()
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2
        self._obj_list = [obj1, obj2]

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        super().evaluate()
        return self.__obj1.evaluate() * self.__obj2.evaluate()

    def jacobian(self):
        super().jacobian()
        vals1 = self.__obj1.evaluate().reshape(-1, 1)
        vals2 = self.__obj2.evaluate().reshape(-1, 1)
        S1 = elem_mul(self.__obj1.jacobian(), vals2)
        S2 = elem_mul(self.__obj2.jacobian(), vals1)
        return S1 + S2


class Division(MyAlgebra):

    def __init__(self, obj1, obj2):
        super().__init__()
        if len(obj1) != len(obj2):
            raise ValueError('length mismatch')
        self.__obj1 = obj1
        self.__obj2 = obj2
        self._obj_list = [obj1, obj2]

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        super().evaluate()
        return self.__obj1.evaluate() / self.__obj2.evaluate()

    def jacobian(self):
        super().jacobian()
        v1 = self.__obj1.evaluate().reshape(-1, 1)
        v2_inv = 1.0 / self.__obj2.evaluate().reshape(-1, 1)
        S1 = elem_mul(self.__obj1.jacobian(), v2_inv)
        S2 = elem_mul(self.__obj2.jacobian(), -v1 * np.square(v2_inv))
        return S1 + S2


class LinearInterpolation(MyAlgebra):

    def __init__(self, obj, src_x, tar_x, zero_outside=False):
        super().__init__()
        if len(obj) != len(src_x):
            raise ValueError('length mismatch')
        self.__obj = obj
        self._obj_list = [obj]
        yzeros = np.zeros(len(src_x), dtype=float)
        self.__jacobian = get_basic_sensmat(
            src_x, yzeros, tar_x, 'lin-lin', zero_outside
        )

    def __len__(self):
        return self.__jacobian.shape[0]

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        return matmul(self.__jacobian, self.__obj.evaluate()).flatten()

    def jacobian(self):
        super().jacobian()
        return matmul(self.__jacobian, self.__obj.jacobian())


class Integral(MyAlgebra):

    def __init__(self, obj, xvals, interp_type, cache=False, **kwargs):
        super().__init__()
        if not isinstance(obj, MyAlgebra):
            raise TypeError('obj must be of class MyAlgebra')
        self.__obj = obj
        self._obj_list = [obj]
        self.__xvals = np.array(xvals)
        self.__interp_type = interp_type
        self.__kwargs = kwargs
        self.__last_result = None
        self.__last_jacobian = None
        self.__cache = cache

    def __len__(self):
        return 1

    def islinear(self):
        return True

    def evaluate(self):
        if self.__cache and not self.values_updated():
            return self.__last_result
        super().evaluate()
        yvals = self.__obj.evaluate()
        self.__last_result = np.array([basic_integral_propagate(
            self.__xvals, yvals, self.__interp_type, **self.__kwargs
        )])
        return self.__last_result.copy()

    def jacobian(self):
        if self.__cache and not self.jacobian_updated():
            return self.__last_jacobian.copy()
        super().jacobian()
        yvals = self.__obj.evaluate()
        outer_jac = csr_matrix(get_basic_integral_sensmat(
            self.__xvals, yvals, self.__interp_type, **self.__kwargs
        ))
        inner_jac = self.__obj.jacobian()
        self.__last_jacobian = matmul(outer_jac, inner_jac)
        return self.__last_jacobian.copy()


class IntegralOfProduct(MyAlgebra):

    def __init__(self, obj_list, xlist, interplist,
                 zero_outside=False, cache=False, **kwargs):
        super().__init__()
        if not all(isinstance(obj, MyAlgebra) for obj in obj_list):
            raise TypeError('all objects in obj_list must be of type ' +
                            'obj_list')
        if len(obj_list) != len(xlist):
            raise IndexError('length of obj_list must equal length ' +
                             'of xlist')
        if len(interplist) != len(obj_list):
            raise IndexError('length of interp_list must equal ' +
                             'length of obj_list')
        self._obj_list = obj_list
        self.__xlist = [np.array(xv) for xv in xlist]
        self.__interplist = interplist
        self.__zero_outside = zero_outside
        self.__kwargs = kwargs
        self.__last_result = None
        self.__last_jacobian = None
        self.__cache = cache

    def __len__(self):
        return 1

    def evaluate(self):
        if self.__cache and not self.values_updated():
            return self.__last_result.copy()
        super().evaluate()
        ylist = [obj.evaluate() for obj in self._obj_list]
        self.__last_result = basic_integral_of_product_propagate(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )
        return self.__last_result.copy()

    def jacobian(self):
        if self.__cache and not self.jacobian_updated():
            return self.__last_jacobian.copy()
        super().jacobian()
        ylist = [obj.evaluate() for obj in self._obj_list]
        outer_jacs = get_basic_integral_of_product_sensmats(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )
        outer_jacs = [csr_matrix(mat) for mat in outer_jacs]
        inner_jacs = [obj.jacobian() for obj in self._obj_list]
        jac = 0.
        for outer_jac, inner_jac in zip(outer_jacs, inner_jacs):
            jac += matmul(outer_jac, inner_jac)
        self.__last_jacobian = jac
        return self.__last_jacobian.copy()


class LegacyFissionAverage(MyAlgebra):

    def __init__(self, en, xsobj, fisen, fisobj,
                 check_norm=True, fix_jacobian=True):
        super().__init__()
        if not isinstance(xsobj, MyAlgebra):
            raise TypeError('xsobj must be of class MyAlgebra')
        if not isinstance(fisobj, MyAlgebra):
            raise TypeError('fisobj must be of class MyAlgebra')
        self.__xsobj = xsobj
        self._obj_list = [xsobj]
        self.__en = en
        self.__fisobj = fisobj
        self.__fisen = fisen
        self.__fix_jacobian = fix_jacobian
        self.__check_norm = check_norm

    def __len__(self):
        return 1

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        xs = self.__xsobj.evaluate()
        fisvals = self.__fisobj.evaluate()
        ret = propagate_fisavg(self.__en, xs, self.__fisen, fisvals,
                               check_norm=self.__check_norm)
        assert isinstance(ret, float)
        return np.array([ret])

    def jacobian(self):
        super().jacobian()
        xs = self.__xsobj.evaluate()
        fisvals = self.__fisobj.evaluate()
        xsjac = self.__xsobj.jacobian()
        if self.__fix_jacobian:
            sensvec = get_sensmat_fisavg_corrected(
                self.__en, xs, self.__fisen, fisvals,
                check_norm=self.__check_norm
            )
        else:
            sensvec = get_sensmat_fisavg(
                self.__en, xs, self.__fisen, fisvals
            )
        out_jac = csr_matrix(sensvec.reshape(1, -1))
        return matmul(out_jac, xsjac)


class FissionAverage(MyAlgebra):

    def __init__(self, en, xsobj, fisen, fisobj,
                 check_norm=True, legacy=False,
                 fix_jacobian=True, **kwargs):
        super().__init__()
        en = np.array(en)
        fisen = np.array(fisen)
        self.__rtol = kwargs.get('rtol', 1e-5)
        self.__atol = kwargs.get('atol', 1e-6)
        self.__maxord = kwargs.get('maxord', 16)
        if legacy:
            self.__fisavg = LegacyFissionAverage(
                en, xsobj, fisen, fisobj, check_norm, fix_jacobian,
            )
        else:
            if check_norm:
                self.__fisint = Integral(
                    fisobj, fisen, 'lin-lin',
                    atol=self.__atol, rtol=self.__rtol, maxord=self.__maxord
                )
            self.__fisavg = IntegralOfProduct(
                [xsobj, fisobj], [en, fisen], ['lin-lin', 'lin-lin'],
                zero_outside=True, atol=self.__atol, rtol=self.__rtol,
                maxord=self.__maxord
            )
        self._obj_list = [self.__fisavg]
        self.__check_norm = check_norm
        self.__legacy = legacy

    def __len__(self):
        return 1

    def islinear(self):
        return type(self.__fisavg) == LegacyFissionAverage

    def evaluate(self):
        super().evaluate()
        if self.__check_norm and not self.__legacy:
            if not np.isclose(self.__fisint.evaluate()[0], 1.,
                              rtol=self.__rtol, atol=self.__atol):
                raise ValueError('fission spectrum not normalized')
        ret = self.__fisavg.evaluate()
        return ret

    def jacobian(self):
        super().jacobian()
        return self.__fisavg.jacobian()
