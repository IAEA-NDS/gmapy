import numpy as np
from scipy.sparse import csr_matrix, vstack
from .basic_maps import get_basic_sensmat
from .helperfuns import numeric_jacobian


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


if __name__ == "__main__":

    def test_jacobian(x0, obj, *inpobjs):
        for inpobj in inpobjs:
            inpobj.assign(x0)
        return obj.jacobian().toarray()

    def reference_jacobian(x0, obj, *inpobjs):
        def eval(x):
            for inpobj in inpobjs:
                inpobj.assign(x)
            return obj.evaluate()
        return numeric_jacobian(eval, x0)

    def is_jacobian_correct(x0, obj, *inpobjs):
        test_jac = test_jacobian(x0, obj, *inpobjs)
        ref_jac = reference_jacobian(x0, obj, *inpobjs)
        return np.allclose(test_jac, ref_jac)

    inpvec = np.array([1, 2, 3, 4, 5, 6])
    # case 1
    x = Selector([0, 1, 2], 6)
    y = Selector([3, 4, 5], 6)
    z1 = LinearInterpolation(x, [1, 2, 3], [1.5, 2, 2.5])
    z2 = LinearInterpolation(y, [1, 2, 3], [1, 2, 3])
    z = z1 + z2
    is_jacobian_correct(inpvec, z, x, y)
    # case 2
    c = Const([1.]*3)
    z = (x+c)/y
    is_jacobian_correct(inpvec, z, x, y)
    # case 3
    x1 = Selector([0, 1, 2], 6)
    x2 = Selector([1, 2, 3], 6)
    x3 = Selector([2, 3, 4], 6)
    z = x1 + x2 + x3
    is_jacobian_correct(inpvec, z, x1, x2, x3)
    # case 4
    z = x1 * x2 * x3
    is_jacobian_correct(inpvec, z, x1, x2, x3)
