import unittest
import numpy as np
from gmapi.mappings.basic_maps import romberg_integral_propagate


class TestRombergIntegralMapping(unittest.TestCase):

    def test_integral_of_expx(self):
        x = [1,5]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = romberg_integral_propagate(x, np.exp, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_expx_with_lower_accuracy(self):
        x = [1,5]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = romberg_integral_propagate(x, np.exp, maxord=20, atol=1e-8, rtol=1e-5)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-5))
        self.assertTrue(np.logical_not(np.isclose(ref_intval, test_intval, rtol=1e-6))) 

    def test_integral_of_xsquared(self):
        x = [1,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = romberg_integral_propagate(x, np.square, maxord=20, atol=1e-8, rtol=1e-8)

    def test_integral_of_xsquared_with_lower_accuracy(self):
        x = [1,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = romberg_integral_propagate(x, np.square, maxord=20, atol=1e-8, rtol=1e-5)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-5))
        self.assertTrue(np.logical_not(np.isclose(ref_intval, test_intval, rtol=1e-6))) 

    def test_integral_of_expx_several_segments(self):
        x = [1,3,7,10]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = romberg_integral_propagate(x, np.exp, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_xsquared_several_segments(self):
        x = [1,3,5,7,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = romberg_integral_propagate(x, np.square, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_xsquared_with_permutated_xlist(self):
        np.random.seed(12)
        x = np.array([1,3,5,7,10])
        perm_x = x[np.random.permutation(len(x))]
        test_intval1 = romberg_integral_propagate(x, np.square, maxord=20, atol=1e-4, rtol=1e-4)
        test_intval2 = romberg_integral_propagate(perm_x, np.square, maxord=20, atol=1e-4, rtol=1e-4)
        self.assertTrue(np.all(test_intval1 == test_intval2))


if __name__ == '__main__':
    unittest.main()

