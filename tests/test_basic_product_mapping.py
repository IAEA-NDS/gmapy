import unittest
import numpy as np
from gmapy.mappings.basic_maps import (basic_propagate, basic_product_propagate,
        get_basic_product_sensmats)
from gmapy.mappings.helperfuns import numeric_jacobian


class TestBasicProductPropagate(unittest.TestCase):

    def test_basic_product_propagate_with_two_factors(self):
        x1 = np.array([1, 3, 7, 13])
        y1 = np.array([4, 2, 8, 20])
        interp1 = 'log-log'
        x2 = np.array([2, 4, 6])
        y2 = np.array([10, 5, 9])
        interp2 = 'lin-lin'
        xout = [1, 4, 6]
        vals1 = basic_propagate(x1, y1, xout, interp1)
        # we skip xout1 because outside of mesh in x2
        vals2 = basic_propagate(x2, y2, xout[1:], interp2)
        ref_prod = np.concatenate([[0], vals1[1:]*vals2])
        test_prod = basic_product_propagate([x1, x2], [y1, y2], xout,
                                            [interp1, interp2], zero_outside=True)
        self.assertTrue(np.all(test_prod == ref_prod))

    def test_basic_product_propagate_with_more_factors(self):
        np.random.seed(12)
        xlist = []
        ylist = []
        interplist = []
        for i in range(5):
            cursize = np.random.randint(2, 10, 1)
            xlist.append(np.random.uniform(0, 100, cursize))
            ylist.append(np.random.uniform(0, 10, cursize)) 
            interplist.append(
                    np.random.choice(['lin-lin', 'lin-log', 'log-lin', 'log-log'], cursize)
                    )
        xout = np.random.uniform(0, 100, 30)
        ref_prod = 1.
        for x, y, interp in zip(xlist, ylist, interplist):
            ref_prod *= basic_propagate(x, y, xout, interp, zero_outside=True)
        test_prod = basic_product_propagate(xlist, ylist, xout,
                                            interplist, zero_outside=True)
        self.assertTrue(np.all(test_prod == ref_prod))


class TestBasicProductJacobian(unittest.TestCase):

    def test_basic_product_sensmat_with_two_factors(self):
        x1 = np.array([1, 3, 7, 13])
        y1 = np.array([4, 2, 8, 20])
        interp1 = 'log-log'
        x2 = np.array([2, 4, 6])
        y2 = np.array([10, 5, 9])
        interp2 = 'lin-lin'
        xout = [1, 4, 6]
        xlist = [x1, x2]
        ylist = [y1, y2]
        interplist = [interp1, interp2]
        def propfun1(y):
            curylist = [y, ylist[1]]
            res = basic_product_propagate(xlist, curylist, xout, interplist,
                                          zero_outside=True)   
            return res
        def propfun2(y):
            curylist = [ylist[0], y]
            res = basic_product_propagate(xlist, curylist, xout, interplist,
                                          zero_outside=True)   
            return res

        ref_res1 = numeric_jacobian(propfun1, ylist[0])
        ref_res2 = numeric_jacobian(propfun2, ylist[1]) 
        test_res = get_basic_product_sensmats(xlist, ylist, xout, interplist,
                                              zero_outside=True)
        self.assertTrue(np.all(np.isclose(test_res[0].toarray(), ref_res1)))
        self.assertTrue(np.all(np.isclose(test_res[1].toarray(), ref_res2)))

    def test_basic_product_propagate_with_more_factors(self):
        np.random.seed(12)
        xlist = []
        ylist = []
        interplist = []
        for i in range(5):
            cursize = np.random.randint(2, 10, 1)
            xlist.append(np.random.uniform(0, 100, cursize))
            ylist.append(np.random.uniform(0, 10, cursize)) 
            interplist.append(
                    np.random.choice(['lin-lin', 'lin-log', 'log-lin', 'log-log'], cursize)
                    )
        xout = np.random.uniform(0, 100, 30)
        ref_prod = 1.
        def generate_propfun(i):
            myylist = ylist.copy()
            def propfun(y):
                myylist[i] = y
                prod = 1.
                for x, y, interp in zip(xlist, myylist, interplist):
                    prod *= basic_propagate(x, y, xout, interp, zero_outside=True)
                return prod
            return propfun

        funlist = [generate_propfun(i) for i in range(len(xlist))]
        ref_res_list = [numeric_jacobian(f, y) for f, y in zip(funlist, ylist)]
        test_res = get_basic_product_sensmats(xlist, ylist, xout, interplist,
                                              zero_outside=True)
        for ref_res, test_res in zip(ref_res_list, test_res):
            self.assertTrue(np.all(np.isclose(test_res.toarray(), ref_res)))


if __name__ == '__main__':
    unittest.main()





