import unittest
import numpy as np
from gmapi.mappings.basic_integral_maps import basic_integral_propagate


class TestBasicIntegralMappings(unittest.TestCase):

    # auxiliary functions
    def integrate_linlin(self, x, y):
        return np.sum((y[:-1] + np.diff(y)/2)*np.diff(x))

    def integrate_linlog(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        log_y = np.log(y)
        log_y1 = log_y[:-1]
        log_y2 = log_y[1:]
        C = (log_y2-log_y1)/(x2-x1)
        intvals = y1 * np.exp(-C*x1) * (np.exp(C*x2) - np.exp(C*x1)) / C
        intval = np.sum(intvals)
        return intval

    def integrate_loglog(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x1 = x[:-1]
        x2 = x[1:]
        log_x = np.log(x)
        log_x1 = log_x[:-1]
        log_x2 = log_x[1:]
        y1 = y[:-1]
        log_y = np.log(y)
        log_y1 = log_y[:-1]
        log_y2 = log_y[1:]
        C = (log_y2-log_y1)/(log_x2-log_x1)
        intvals = y1 * np.exp(-C*log_x1) * (x2**(C+1) - x1**(C+1)) / (C+1) 
        intval = np.sum(intvals)
        return intval

    # tests start here
    def test_basic_integral_propagate_for_linlin_interp(self):
        x = [1, 10, 20, 50]
        y = [5, 10,  8, 23]
        test_res = basic_integral_propagate(x, y, interp_type='lin-lin', rtol=1e-8)
        ref_res = self.integrate_linlin(x, y)
        self.assertTrue(np.isclose(test_res, ref_res, rtol=1e-8))

    def test_basic_integral_propagate_for_linlog_interp(self):
        x = [7, 9, 18, 25]
        y = [3, 13, 9, 24]
        ref_intval = self.integrate_linlog(x, y)
        test_intval = basic_integral_propagate(x, y, interp_type='lin-log',
                                               maxord=20, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, rtol=1e-8))

    def test_basic_integral_propagate_for_loglog_interp(self):
        x = [7, 9, 18, 25]
        y = [3, 13, 9, 24]
        ref_intval = self.integrate_loglog(x, y)
        test_intval = basic_integral_propagate(x, y, interp_type='log-log',
                                               maxord=20, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval, rtol=1e-8))

    def test_basic_integral_propagate_for_mixed_interp(self):
        x = [7, 9, 18, 25]
        y = [3, 13, 9, 24]
        interp = ['lin-log', 'log-log', 'lin-lin', 'log-log']
        fundic = {'lin-log': lambda x, y: self.integrate_linlog(x, y),
                  'log-log': lambda x, y: self.integrate_loglog(x, y),
                  'lin-lin': lambda x, y: self.integrate_linlin(x, y)}
        range_list = zip(x[:-1], x[1:], y[:-1], y[1:])
        ref_intval = 0
        for i, cur in enumerate(range_list):
            ref_intval += fundic[interp[i]]([cur[0], cur[1]], [cur[2], cur[3]])
        test_intval = basic_integral_propagate(x, y, interp, maxord=18, rtol=1e-8)
        self.assertTrue(np.isclose(test_intval, ref_intval))

    def test_basic_integral_propagate_with_permuted_input(self):
        x1 = np.array([7, 9, 18, 25])
        y1 = np.array([3, 13, 9, 24])
        perm = np.random.seed(17)
        perm = np.random.permutation(len(x1))
        x2 = x1[perm]
        y2 = y1[perm]
        interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
        for curint in interp_types:
            errmsg = f'failed for interpolation type {curint}'
            res1 = basic_integral_propagate(x1, y1, interp_type=curint, rtol=1e-5)
            res2 = basic_integral_propagate(x2, y2, interp_type=curint, rtol=1e-5)
            self.assertTrue(np.all(res1==res2), errmsg)


if __name__ == '__main__':
    unittest.main()

