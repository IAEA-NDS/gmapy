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

# the following classes are the building
# blocks to construct mathematical expressions
# enabling the automated computation of derivatives


class MyAlgebra:

    def __init__(self):
        self._values_updated = True
        self._jacobian_updated = True
        self._ancestors = []
        self._descendants = []
        # track descendants whose input is
        # a non-linear function of the
        # input of this mapping
        self._nonlinear_descendants = set()
        self._linear_descendants = set()

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

    def __track_nonlinear_deps(self, path):
        path = [self] + path
        for ancestor in self._ancestors:
            ancestor.__track_nonlinear_deps(path)
        for i, curobj in enumerate(path[1:]):
            if not curobj.islinear():
                self._linear_descendants.update(path[1:i+1])
                self._nonlinear_descendants.update(path[i+1:])
                return
        self._linear_descendants.update(path[1:])

    def _add_descendant(self, descendant):
        self._descendants.append(descendant)
        self.__track_nonlinear_deps([descendant])

    def _add_ancestors(self, ancestors):
        self._ancestors.extend(ancestors)

    def _get_ancestors(self):
        return self._ancestors

    def _signal_changes(self):
        self._values_updated = True
        if not self.islinear():
            self._jacobian_updated = True
        for desc in self._nonlinear_descendants:
            desc._values_updated = True
            desc._jacobian_updated = True
        for desc in self._linear_descendants:
            desc._values_updated = True

    def values_updated(self):
        return self._values_updated

    def jacobian_updated(self):
        return self._jacobian_updated


class InputSelector(MyAlgebra):

    def __init__(self, idcs, size):
        super().__init__()
        self.__idcs = np.array(idcs)
        self.__size = size
        self.__values = None
        # pre-compute jacobian
        tar_size = len(self.__idcs)
        coeffs = np.ones(tar_size)
        tar_idcs = np.arange(tar_size)
        self.__jacmat = csr_matrix(
            (coeffs, (tar_idcs, self.__idcs)),
            shape=(tar_size, self.__size), dtype=float
        )

    def __len__(self):
        return len(self.__idcs)

    def islinear(self):
        return True

    def evaluate(self):
        super().evaluate()
        if self.__values is None:
            raise ValueError('please assign numbers')
        return self.__values.copy()

    def jacobian(self):
        super().jacobian()
        return self.__jacmat

    def assign(self, arraylike):
        if len(arraylike) != self.__size:
            raise IndexError('wrong length of vector')
        newvals = np.array(arraylike)[self.__idcs]
        if (self.__values is None
                or np.any(newvals != self.__values)):
            self.__values = newvals
            self._signal_changes()

    def get_indices(self):
        return self.__idcs.copy()


class Selector(MyAlgebra):

    def __init__(self, inpobj, idcs):
        super().__init__()
        if not isinstance(inpobj, MyAlgebra):
            raise TypeError('please provide instance derived from MyAlgebra')
        inpobj._add_descendant(self)
        self._add_ancestors([inpobj])
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
        for sel in listlike:
            self.add_selector(sel)

    def add_selector(self, selector):
        if type(selector) != InputSelector:
            raise TypeError('only InputSelector instance allowed')
        selids = {id(sel) for sel in self.__selector_list}
        if id(selector) not in selids:
            self.__selector_list.append(selector)

    def define_selector(self, idcs, size):
        for sel in self.__selector_list:
            refidcs = sel.get_indices()
            if len(idcs) == len(refidcs):
                if np.all(idcs == refidcs):
                    return sel
        newsel = InputSelector(idcs, size)
        self.__selector_list.append(newsel)
        return newsel


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

    def __init__(self, obj, idcs, size, cache=True):
        super().__init__()
        if len(obj) != len(idcs):
            raise ValueError('size mismatch')
        obj._add_descendant(self)
        self._add_ancestors([obj])
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
        self.__cache = cache
        self.__last_propvals = None
        self.__last_jacobian = None

    def __len__(self):
        return self.__size

    def islinear(self):
        return True

    def evaluate(self):
        res = np.zeros(self.__size, dtype=float)
        if self.__cache and not self.values_updated():
            res[self.__idcs] += self.__last_propvals
            return res
        super().evaluate()
        nonzero_res = self.__obj.evaluate()
        self.__last_propvals = nonzero_res
        res[self.__idcs] += nonzero_res
        return res

    def jacobian(self):
        if self.__cache and not self.jacobian_updated():
            return self.__last_jacobian
        super().jacobian()
        jac = matmul(self.__dist_mat, self.__obj.jacobian())
        self.__last_jacobian = jac
        return jac

    def get_indices(self):
        return self.__idcs.copy()


class SumOfDistributors(MyAlgebra):

    def __init__(self, listlike=None):
        super().__init__()
        self.__distributor_list = []
        if listlike is not None:
            self.add_distributors(listlike)

    def get_indices(self):
        if len(self.__distributor_list) > 0:
            return np.unique(np.concatenate(list(
                obj.get_indices() for obj in self.__distributor_list
            )))
        else:
            return np.empty(0, dtype=int)

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
        for dist in listlike:
            self.add_distributor(dist)

    def add_distributor(self, distributor):
        if type(distributor) not in (Distributor, SumOfDistributors):
            raise TypeError('only Distributor and SumOfDistributor instance ' +
                            'allowed as argument')
        self.__distributor_list.append(distributor)
        distributor._add_descendant(self)
        self._add_ancestors([distributor])


class Replicator(MyAlgebra):

    def __init__(self, obj, num):
        super().__init__()
        obj._add_descendant(self)
        self._add_ancestors([obj])
        self.__num = num
        self.__obj = obj

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
        obj1._add_descendant(self)
        obj2._add_descendant(self)
        self._add_ancestors([obj1, obj2])
        self.__obj1 = obj1
        self.__obj2 = obj2

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
        obj1._add_descendant(self)
        obj2._add_descendant(self)
        self._add_ancestors([obj1, obj2])
        self.__obj1 = obj1
        self.__obj2 = obj2

    def __len__(self):
        return len(self.__obj1)

    def evaluate(self):
        super().evaluate()
        res1 = self.__obj1.evaluate()
        res2 = self.__obj2.evaluate()
        return res1 * res2

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
        obj1._add_descendant(self)
        obj2._add_descendant(self)
        self._add_ancestors([obj1, obj2])
        self.__obj1 = obj1
        self.__obj2 = obj2

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
        obj._add_descendant(self)
        self._add_ancestors([obj])
        self.__obj = obj
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
        obj._add_descendant(self)
        self._add_ancestors([obj])
        self.__obj = obj
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
        for obj in obj_list:
            obj._add_descendant(self)
            self._add_ancestors([obj])
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
        ylist = [obj.evaluate() for obj in self._get_ancestors()]
        self.__last_result = basic_integral_of_product_propagate(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )
        return self.__last_result.copy()

    def jacobian(self):
        if self.__cache and not self.jacobian_updated():
            return self.__last_jacobian.copy()
        super().jacobian()
        ancestors = self._get_ancestors()
        ylist = [obj.evaluate() for obj in ancestors]
        outer_jacs = get_basic_integral_of_product_sensmats(
            self.__xlist, ylist, self.__interplist,
            self.__zero_outside, **self.__kwargs
        )
        outer_jacs = [csr_matrix(mat) for mat in outer_jacs]
        inner_jacs = [obj.jacobian() for obj in ancestors]
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
        xsobj._add_descendant(self)
        self._add_ancestors([xsobj])
        self.__xsobj = xsobj
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
        self.__fisavg._add_descendant(self)
        self._add_ancestors([self.__fisavg])
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
