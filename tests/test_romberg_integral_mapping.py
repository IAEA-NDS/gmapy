import unittest
import numpy as np
from gmapy.mappings.basic_maps import (basic_propagate, get_basic_sensmat,
        basic_extract_Sdic_coeffs)
from gmapy.mappings.helperfuns import numeric_jacobian, compute_romberg_integral


class TestRombergIntegralMapping(unittest.TestCase):

    def test_integral_of_expx(self):
        x = [1,5]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = compute_romberg_integral(x, np.exp, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_expx_with_lower_accuracy(self):
        x = [1,5]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = compute_romberg_integral(x, np.exp, maxord=20, atol=1e-8, rtol=1e-5)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-5))
        self.assertTrue(np.logical_not(np.isclose(ref_intval, test_intval, rtol=1e-6))) 

    def test_integral_of_xsquared(self):
        x = [1,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = compute_romberg_integral(x, np.square, maxord=20, atol=1e-8, rtol=1e-8)

    def test_integral_of_xsquared_with_lower_accuracy(self):
        x = [1,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = compute_romberg_integral(x, np.square, maxord=20, atol=1e-8, rtol=1e-5)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-5))
        self.assertTrue(np.logical_not(np.isclose(ref_intval, test_intval, rtol=1e-6))) 

    def test_integral_of_expx_several_segments(self):
        x = [1,3,7,10]
        ref_intval = np.exp(x[-1]) - np.exp(x[0])
        test_intval = compute_romberg_integral(x, np.exp, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_xsquared_several_segments(self):
        x = [1,3,5,7,10]
        ref_intval = (x[-1]**3 - x[0]**3) / 3
        test_intval = compute_romberg_integral(x, np.square, maxord=20, atol=1e-8, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, atol=1e-8, rtol=1e-8))

    def test_integral_of_xsquared_with_permutated_xlist(self):
        np.random.seed(12)
        x = np.array([1,3,5,7,10])
        perm_x = x[np.random.permutation(len(x))]
        test_intval1 = compute_romberg_integral(x, np.square, maxord=20, atol=1e-4, rtol=1e-4)
        test_intval2 = compute_romberg_integral(perm_x, np.square, maxord=20, atol=1e-4, rtol=1e-4)
        self.assertTrue(np.all(test_intval1 == test_intval2))


class TestRombergIntegralJacobian(unittest.TestCase):

    # auxiliary functions
    def generate_myfun(self, xref, yref, interp):
        def myfun(x):
            return basic_propagate(xref, yref, x, interp)
        return myfun

    def generate_mydfun(self, xref, yref, interp):
        def mydfun(x):
            Sdic = get_basic_sensmat(xref, yref, x, interp, ret_mat=False)
            coeffs1, coeffs2 = basic_extract_Sdic_coeffs(Sdic)
            return (coeffs1, coeffs2)
        return mydfun

    def generate_myintfun(self, xref, interp, rtol):
        def myintfun(y):
            def tmpfun(x):
                return basic_propagate(xref, y, x, interp)
            return np.array([compute_romberg_integral(xref, tmpfun,
                                                        maxord=18, rtol=rtol)])
        return myintfun

    # tests starting here
    def test_compare_romberg_integral_jacobian_with_numerical_jacobian(self):
        xref = np.array([1, 5, 7, 10])
        yref = np.square(xref)
        intstrs = ['lin', 'log']
        possible_interp = ['-'.join([a,b]) for a in intstrs for b in intstrs]
        for rtol in [1e-3, 1e-5, 1e-7]:
            for interp in possible_interp:
                myfun = self.generate_myfun(xref, yref, interp)
                mydfun = self.generate_mydfun(xref, yref, interp)
                myintfun = self. generate_myintfun(xref, interp, rtol)
                Sref = numeric_jacobian(myintfun, yref, o=4, h1=0.01)
                Stest = compute_romberg_integral(xref, myfun, dfun=mydfun, maxord=18, rtol=rtol)
                self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_permuted_vs_original_input_for_romberg_integral_jacobian(self):
        xref1 = np.array([1, 5, 7, 10])
        yref1 = np.square(xref1)
        interp1 = np.array(['lin-log', 'log-log', 'log-lin', 'lin-lin'])
        myfun1 = self.generate_myfun(xref1, yref1, interp1)
        mydfun1 = self.generate_mydfun(xref1, yref1, interp1)
        np.random.seed(13)
        perm = np.random.permutation(len(xref1))
        xref2 = xref1[perm]
        yref2 = yref1[perm]
        interp2 = interp1[perm]
        myfun2 = self.generate_myfun(xref2, yref2, interp2)
        mydfun2 = self.generate_mydfun(xref2, yref2, interp2)
        res1 = compute_romberg_integral(xref1, myfun1, dfun=mydfun1, maxord=14)
        res2 = compute_romberg_integral(xref2, myfun2, dfun=mydfun2, maxord=14)
        self.assertTrue(np.all(res1 == res2))


if __name__ == '__main__':
    unittest.main()

