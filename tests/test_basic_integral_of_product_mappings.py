import unittest
import numpy as np
from gmapi.mappings.basic_maps import basic_propagate
from gmapi.mappings.basic_integral_maps import basic_integral_of_product_propagate
from gmapi.mappings.helperfuns import numeric_jacobian, compute_romberg_integral


class TestBasicIntegralOfProductMapping(unittest.TestCase):

    def test_basic_integral_of_product_propagate_for_different_interps(self):
        x1 = np.array([1, 3, 8, 14])
        y1 = np.array([7, 2, 9, 16])
        interp1 = 'lin-lin'
        x2 = np.array([4, 5.5, 6.9])
        y2 = np.array([11, 22, 3, 29])
        interp2 = 'log-log'
        x3 = np.array([2.1, 4.9, 7.8])
        y3 = np.array([5,   5.5, 2.4])
        interp3 = 'lin-log'
        xlist = [x1, x2, x3]
        ylist = [y1, y2, y3]
        def propfun(x):
            r1 = basic_propagate(x1, y1, x, interp1) 
            r2 = basic_propagate(x2, y2, x, interp2)
            r3 = basic_propagate(x3, y3, x, interp3)
            return r1*r2*r3
        min_x = max([min(x1), min(x2), min(x3)])
        max_x = min([max(x1), max(x2), max(x3)])
        ref_x1 = np.array(x1)[np.logical_and(x1 >= min_x, x1 <= max_x)]
        ref_x2 = np.array(x2)[np.logical_and(x2 >= min_x, x2 <= max_x)]
        ref_x3 = np.array(x3)[np.logical_and(x3 >= min_x, x3 <= max_x)]
        ref_x = np.unique(np.concatenate([x1, x2, x3]))
        ref_x = ref_x[np.logical_and(ref_x >= min_x, ref_x <= max_x)]
        interplist = [interp1, interp2, interp3]
        test_res = basic_integral_of_product_propagate(xlist, ylist, interplist, maxord=20, rtol=1e-6)
        ref_res = compute_romberg_integral(ref_x, propfun, maxord=20, rtol=1e-6)
        self.assertTrue(np.all(test_res == ref_res))

    def test_basic_integral_of_product_propagate_with_permutated_input(self):
        np.random.seed(17)
        x1 = np.array([1, 3, 8, 14])
        y1 = np.array([7, 2, 9, 16])
        perm1 = np.random.permutation(len(x1))
        perm_x1 = x1[perm1]
        perm_y1 = y1[perm1]
        interp1 = 'lin-lin'
        x2 = np.array([4, 5.5, 6.9])
        y2 = np.array([11, 22, 3, 29])
        perm2 = np.random.permutation(len(x2))
        perm_x2 = x2[perm2]
        perm_y2 = y2[perm2]
        interp2 = 'log-log'
        x3 = np.array([2.1, 4.9, 7.8])
        y3 = np.array([5,   5.5, 2.4])
        interp3 = 'lin-log'
        perm3 = np.random.permutation(len(x3))
        perm_x3 = x3[perm3]
        perm_y3 = y3[perm3]

        xlist = [x1, x2, x3]
        ylist = [y1, y2, y3]
        interplist = [interp1, interp2, interp3]
        perm_xlist = [x3, x1, x2]
        perm_ylist = [y3, y1, y2]
        perm_interplist = [interp3, interp1, interp2]

        test_res1 = basic_integral_of_product_propagate(xlist, ylist, interplist, maxord=20, rtol=1e-3)
        test_res2 = basic_integral_of_product_propagate(perm_xlist, perm_ylist, perm_interplist,
                                                        maxord=20, rtol=1e-3)
        self.assertTrue(np.all(test_res1 == test_res2))


if __name__ == '__main__':
    unittest.main()

