import unittest
import numpy as np
from gmapy.mappings.helperfuns import numeric_jacobian
from gmapy.mappings.mapping_elements import (
    Selector,
    LinearInterpolation,
    Integral,
    IntegralOfProduct,
    FissionAverage,
    Const,
    Replicator
)


class TestMappingElements(unittest.TestCase):

    def eval_expr(self, x0, obj, *inpobjs):
        for inpobj in inpobjs:
            inpobj.assign(x0)
        return obj.evaluate()

    def assert_equal(self, x, y):
        self.assertTrue(np.allclose(x, y))

    def test_addition(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x1 = Selector([0, 1, 2], 6)
        x2 = Selector([1, 2, 3], 6)
        x3 = Selector([2, 3, 4], 6)
        z = x1 + x2 + x3
        self.assert_equal(self.eval_expr(inpvec, z, x1, x2, x3),
                          np.array([6, 9, 12]))

    def test_product(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x1 = Selector([0, 1, 2], 6)
        x2 = Selector([1, 2, 3], 6)
        x3 = Selector([2, 3, 4], 6)
        z = x1 * x2 * x3
        self.assert_equal(
            self.eval_expr(inpvec, z, x1, x2, x3),
            np.array([6, 24, 60])
        )

    def test_ratio(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = x / y
        self.assert_equal(
            self.eval_expr(inpvec, z, x, y),
            np.array([1/4, 2/5, 3/6])
        )

    def test_expression_with_constant(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        c1 = Const([1.]*3)
        c2 = Const([2.7, 3.4, 8.9])
        z = (x+c1)/(y+c2)
        self.assert_equal(
            self.eval_expr(inpvec, z, x, y),
            np.array([2/(4+2.7), 3/(5+3.4), 4/(6+8.9)])
        )

    def test_linear_interpolation(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        z = LinearInterpolation(x, [1, 2, 3], [1.5, 2, 2.5])
        self.assert_equal(
            self.eval_expr(inpvec, z, x),
            np.array([1.5, 2.0, 2.5])
        )

    def test_sum_of_linear_interpolation(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z1 = LinearInterpolation(x, [1, 2, 3], [1.5, 2, 2.5])
        z2 = LinearInterpolation(y, [1, 2, 3], [1, 2, 3])
        z = z1 + z2
        self.assert_equal(
            self.eval_expr(inpvec, z, x, y),
            np.array([5.5, 7.0, 8.5])
        )

    def test_integral(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = x + y
        intz = Integral(z, [1, 2, 3], 'lin-lin', maxord=10)
        self.assert_equal(
            self.eval_expr(inpvec, intz, x, y),
            np.array([14.0])
        )

    def test_integral_with_permutated_input(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        intx = Integral(x, [1, 2, 3], 'lin-lin', maxord=10)
        perm_x = Selector([2, 1, 0], 6)
        perm_intx = Integral(perm_x, [3, 2, 1], 'lin-lin', maxord=10)
        x.assign(inpvec)
        perm_x.assign(inpvec)
        intval = intx.evaluate()
        perm_intval = perm_intx.evaluate()
        self.assertTrue(np.all(intval == perm_intval))

    def test_integral_of_product(self):
        inpvec = np.array([1, 2, 3, 1, 2, 3])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = IntegralOfProduct([x, y], [[1, 2, 3], [1, 2, 3]],
                              ['lin-lin', 'lin-lin'], maxord=10)
        self.assert_equal(
            self.eval_expr(inpvec, z, x, y),
            np.array([26/3])
        )


class TestMappingJacobians(unittest.TestCase):

    def eval_test_jacobian(self, x0, obj, *inpobjs):
        for inpobj in inpobjs:
            inpobj.assign(x0)
        return obj.jacobian().toarray()

    def eval_reference_jacobian(self, x0, obj, *inpobjs):
        def eval(x):
            for inpobj in inpobjs:
                inpobj.assign(x)
            return obj.evaluate()
        return numeric_jacobian(eval, x0)

    def is_jacobian_correct(self, x0, obj, *inpobjs):
        test_jac = self.eval_test_jacobian(x0, obj, *inpobjs)
        ref_jac = self.eval_reference_jacobian(x0, obj, *inpobjs)
        return np.allclose(test_jac, ref_jac)

    def is_subjacobian_correct(self, row_idcs, x0, obj, *inpobjs):
        test_jac = self.eval_test_jacobian(x0, obj, *inpobjs)
        ref_jac = self.eval_reference_jacobian(x0, obj, *inpobjs)
        test_jac = test_jac[:, np.array(row_idcs)]
        ref_jac = ref_jac[:, np.array(row_idcs)]
        return np.allclose(test_jac, ref_jac)

    def test_addition(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x1 = Selector([0, 1, 2], 6)
        x2 = Selector([1, 2, 3], 6)
        x3 = Selector([2, 3, 4], 6)
        z = x1 + x2 + x3
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x1, x2, x3))

    def test_product(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x1 = Selector([0, 1, 2], 6)
        x2 = Selector([1, 2, 3], 6)
        x3 = Selector([2, 3, 4], 6)
        z = x1 * x2 * x3
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x1, x2, x3))

    def test_ratio(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = x / y
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x, y))

    def test_expression_with_constant(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        c1 = Const([1.]*3)
        c2 = Const([2.7, 3.4, 8.9])
        z = (x+c1)/(y+c2)
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x, y))

    def test_linear_interpolation(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        z = LinearInterpolation(x, [1, 2, 3], [1.5, 2, 2.5])
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x))

    def test_sum_of_linear_interpolation(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z1 = LinearInterpolation(x, [1, 2, 3], [1.5, 2, 2.5])
        z2 = LinearInterpolation(y, [1, 2, 3], [1, 2, 3])
        z = z1 + z2
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x, y))

    def test_integral(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = x + y
        intz = Integral(z, [1, 2, 3], 'lin-lin', maxord=10)
        self.assertTrue(self.is_jacobian_correct(inpvec, intz, x, y))

    def test_integral_with_permutated_input(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        intx = Integral(x, [1, 2, 3], 'lin-lin', maxord=10)
        perm_x = Selector([2, 1, 0], 6)
        perm_intx = Integral(perm_x, [3, 2, 1], 'lin-lin', maxord=10)
        x.assign(inpvec)
        perm_x.assign(inpvec)
        jac = intx.jacobian().toarray()
        perm_jac = perm_intx.jacobian().toarray()
        self.assertTrue(np.all(jac == perm_jac))

    def test_integral_of_product(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6])
        x = Selector([0, 1, 2], 6)
        y = Selector([3, 4, 5], 6)
        z = IntegralOfProduct([x, y], [[0, 1, 7], [0, 4, 7]],
                              ['lin-lin', 'lin-lin'], maxord=8)
        self.assertTrue(self.is_jacobian_correct(inpvec, z, x, y))

    def test_involved_expression_with_integral_of_product(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = Selector([0, 1, 2], 9)
        y = Selector([2, 3, 4], 9)
        z = Selector([4, 5, 6], 9)
        x1 = x + y
        x2 = y * z
        z1 = IntegralOfProduct([x1, x2], [[0, 3, 7], [0, 4, 7]],
                               ['lin-lin', 'lin-lin'], maxord=10)
        z2 = Replicator(z1, 3)
        z3 = z2 + y
        self.assertTrue(self.is_jacobian_correct(inpvec, z3, x, y, z))

    def test_fission_average(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        xs = Selector([0, 1, 2, 3], 9)
        fisvals = Selector([4, 5, 6, 7, 8], 9)
        fisavg = FissionAverage([0, 2, 4, 9], xs,
                                [0, 1, 3, 5, 9], fisvals,
                                legacy=False, check_norm=False)
        self.assertTrue(self.is_jacobian_correct(inpvec, fisavg, xs, fisvals))

    def test_legacy_fission_average(self):
        inpvec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        xsidcs = [0, 1, 2, 3]
        fisen = [0, 1, 3, 5, 9]
        xs = Selector(xsidcs, 9)
        fisvals = Selector([4, 5, 6, 7, 8], 9)
        fisavg = FissionAverage([0, 2, 4, 9], xs,
                                fisen, fisvals, check_norm=False,
                                legacy=True, fix_jacobian=True)
        self.assertTrue(self.is_subjacobian_correct(
            xsidcs, inpvec, fisavg, xs, fisvals)
        )


if __name__ == '__main__':
    unittest.main()
