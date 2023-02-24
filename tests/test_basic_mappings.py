import unittest
import numpy as np
from scipy.sparse import csr_matrix
from gmapy.mappings.basic_maps import (
    basic_propagate,
    get_basic_sensmat,
)
from gmapy.mappings.helperfuns import numeric_jacobian


class TestBasicMappingsPropagation(unittest.TestCase):

    def test_mapping_failure_for_xout_beyond_mesh(self):
        x = [1, 5, 10]
        y = [11, 13, 19]
        xout1 = [0.99]
        xout2 = [10.01]
        with self.assertRaises(ValueError):
            basic_propagate(x, y, xout1)
        with self.assertRaises(ValueError):
            basic_propagate(x, y, xout2)

    def test_mapping_for_xout_beyond_mesh_with_zero_outside_true(self):
        x = [1, 5, 10]
        y = [11, 13, 19]
        xout1 = [0.99]
        xout2 = [10.01]
        yval1 = basic_propagate(x, y, xout1, zero_outside=True)
        yval2 = basic_propagate(x, y, xout2, zero_outside=True)
        self.assertEqual(yval1, 0.)
        self.assertEqual(yval2, 0.)

    def test_mapping_for_some_xout_beyond_mesh_with_zero_outside_true(self):
        x = [1, 3, 7, 13]
        y = [4, 9, 8, 20]
        interp = 'lin-log'
        xout = [-2, 5, 7, 15]
        red_res = basic_propagate(x, y, xout[1:3], interp)
        ref_res = np.concatenate([[0], red_res, [0]])
        test_res = basic_propagate(x, y, xout, interp, zero_outside=True)
        self.assertTrue(np.all(test_res == ref_res))

    def test_lin_lin_propagate(self):
        x = [1, 5, 10]
        y = [11, 15, 10]
        xout = [2, 3, 7, 9]
        yout = basic_propagate(x, y, xout, interp_type='lin-lin')
        yout_ref = [12, 13, 13, 11]
        self.assertTrue(np.all(yout == yout_ref))

    def test_log_lin_propagate(self):
        x = np.exp([1, 5, 10])
        y = [11, 15, 10]
        xout = np.exp([2, 3, 7, 9])
        yout = basic_propagate(x, y, xout, interp_type='log-lin')
        yout_ref = [12, 13, 13, 11]
        self.assertTrue(np.all(yout == yout_ref))

    def test_lin_log_propagate(self):
        x = [1, 5, 10]
        y = np.exp([11, 15, 10])
        xout = [2, 3, 7, 9]
        yout = basic_propagate(x, y, xout, interp_type='lin-log')
        yout_ref = np.exp([12, 13, 13, 11])
        self.assertTrue(np.all(yout == yout_ref))

    def test_log_log_propagate(self):
        x = np.exp([1, 5, 10])
        y = np.exp([11, 15, 10])
        xout = np.exp([2, 3, 7, 9])
        yout = basic_propagate(x, y, xout, interp_type='log-log')
        yout_ref = np.exp([12, 13, 13, 11])
        self.assertTrue(np.all(yout == yout_ref))

    def test_mixed_interpolation(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 25, 19])
        interp_type = ['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin']
        # output
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        yout = basic_propagate(x, y, xout, interp_type)
        # reference propagates
        yout_linlin = basic_propagate(x, y, xout, interp_type='lin-lin')
        yout_loglin = basic_propagate(x, y, xout, interp_type='log-lin')
        yout_linlog = basic_propagate(x, y, xout, interp_type='lin-log')
        yout_loglog = basic_propagate(x, y, xout, interp_type='log-log')
        linlin_sel = np.array([1,2,3,7])
        loglin_sel = np.array([4,5,6])
        linlog_sel = np.array([0,7,8])
        loglog_sel = np.array([4,5])
        yout_ref = np.empty(len(xout), dtype=float)
        yout_ref[linlin_sel] = yout_linlin[linlin_sel]
        yout_ref[linlog_sel] = yout_linlog[linlog_sel]
        yout_ref[loglin_sel] = yout_loglin[loglin_sel]
        yout_ref[loglog_sel] = yout_loglog[loglog_sel]
        # comparison
        self.assertTrue(np.all(yout == yout_ref))

    def test_permutated_x_and_xout_mesh_for_all_interp_types(self):
        x = np.array([1, 3, 5, 7, 9, 10])
        y = np.array([11, 7, 15, 10, 25, 19])
        xout = np.array([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        possible_interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
        # permutate
        np.random.seed(31)
        for curint in possible_interp_types:
            perm = np.random.permutation(len(x))
            perm_out = np.random.permutation(len(xout))
            perm_x = x[perm]
            perm_y = y[perm]
            perm_xout = xout[perm_out]
            yout_ref = basic_propagate(x, y, xout, curint)[perm_out]
            yout_test = basic_propagate(perm_x, perm_y, perm_xout, curint)
            errmsg = f'Permutation test for interpolation {curint} failed'
            self.assertTrue(np.all(np.isclose(yout_test, yout_ref)), errmsg)

    def test_permutated_x_mesh_for_mixed_interpolation(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 25, 19])
        interp_type = np.array(['lin-log', 'lin-lin', 'log-log',
                                'log-lin', 'log-log', 'lin-lin'])
        # permutate
        np.random.seed(84)
        perm = np.random.permutation(len(x))
        x = x[perm]
        y = y[perm]
        interp_type = interp_type[perm]
        # output
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        yout = basic_propagate(x, y, xout, interp_type)
        # reference propagates
        yout_linlin = basic_propagate(x, y, xout, interp_type='lin-lin')
        yout_loglin = basic_propagate(x, y, xout, interp_type='log-lin')
        yout_linlog = basic_propagate(x, y, xout, interp_type='lin-log')
        yout_loglog = basic_propagate(x, y, xout, interp_type='log-log')
        linlin_sel = np.array([1,2,3,7])
        loglin_sel = np.array([4,5,6])
        linlog_sel = np.array([0,7,8])
        loglog_sel = np.array([4,5])
        yout_ref = np.empty(len(xout), dtype=float)
        yout_ref[linlin_sel] = yout_linlin[linlin_sel]
        yout_ref[linlog_sel] = yout_linlog[linlog_sel]
        yout_ref[loglin_sel] = yout_loglin[loglin_sel]
        yout_ref[loglog_sel] = yout_loglog[loglog_sel]
        # comparison
        self.assertTrue(np.all(yout == yout_ref))

    def test_permuted_propagation_calculation_in_yet_another_way(self):
        x1 = [5, 3, 1, 7]
        y1 = [1, 9, 14, 23]
        perm = np.argsort(x1)
        x2 = np.array(x1, copy=True)[perm]
        y2 = np.array(y1, copy=True)[perm]
        xout = [1.4, 1.8, 3.7, 4.9, 5.2]
        possible_interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
        for curint in possible_interp_types:
            errmsg = f'propagation failed for interpolation type {curint}'
            propvals1 = basic_propagate(x1, y1, xout, curint)
            propvals2 = basic_propagate(x2, y2, xout, curint)
            self.assertTrue(np.all(propvals1==propvals2), errmsg)

    def test_interp_at_mesh_boundaries(self):
        x = [1, 5, 10]
        y = [2, 8, 10]
        xout = [1, 10]
        yout1 = basic_propagate(x, y, xout, 'lin-lin')
        yout2 = basic_propagate(x, y, xout, 'lin-log')
        yout3 = basic_propagate(x, y, xout, 'log-lin')
        yout4 = basic_propagate(x, y, xout, 'log-log')
        yref = [2, 10]
        self.assertTrue(np.all(np.isclose(yout1, yref)))
        self.assertTrue(np.all(np.isclose(yout2, yref)))
        self.assertTrue(np.all(np.isclose(yout3, yref)))
        self.assertTrue(np.all(np.isclose(yout4, yref)))

    def test_interp_with_single_element_in_x(self):
        x = [5]
        y = [10]
        xout = [5]
        yout1 = basic_propagate(x, y, xout, 'lin-lin')
        yout2 = basic_propagate(x, y, xout, 'lin-log')
        yout3 = basic_propagate(x, y, xout, 'log-lin')
        yout4 = basic_propagate(x, y, xout, 'log-log')
        self.assertEqual(yout1, 10)
        self.assertEqual(yout2, 10)
        self.assertEqual(yout3, 10)
        self.assertEqual(yout4, 10)


class TestBasicMappingsJacobian(unittest.TestCase):

    def create_propagate_wrapper(self, x, xout, interp_type):
        def myprop(y):
            return basic_propagate(x, y, xout, interp_type)
        return myprop

    def test_failure_for_xout_beyond_mesh(self):
        x = [1, 5, 10]
        y = [11, 13, 19]
        xout1 = [0.99]
        xout2 = [10.01]
        with self.assertRaises(ValueError):
            get_basic_sensmat(x, y, xout1)
        with self.assertRaises(ValueError):
            get_basic_sensmat(x, y, xout2)
            basic_propagate(x, y, xout2)

    def test_sensmat_for_xout_beyond_mesh_with_zero_outside_true(self):
        x = [1, 5, 10]
        y = [11, 13, 19]
        xout1 = [0.99]
        xout2 = [10.01]
        Smat1 = get_basic_sensmat(x, y, xout1, zero_outside=True)
        Smat2 = get_basic_sensmat(x, y, xout2, zero_outside=True)
        self.assertTrue(np.all(Smat1.toarray() == Smat2.toarray()))

    def test_mapping_for_some_xout_beyond_mesh_with_zero_outside_true(self):
        x = [1, 3, 7, 13]
        y = [4, 9, 8, 20]
        interp = 'lin-log'
        xout = [-2, 5, 7, 15]
        red_Smat = get_basic_sensmat(x, y, xout[1:3], interp)
        test_Smat = get_basic_sensmat(x, y, xout, interp, zero_outside=True)
        self.assertTrue(np.all(test_Smat[1:-1,:].toarray() == red_Smat.toarray()))
        self.assertTrue(np.all(test_Smat[0,:].toarray() == 0.))
        self.assertTrue(np.all(test_Smat[-1,:].toarray() == 0))

    def test_lin_lin_sensitivity(self):
        x = [1, 5, 10]
        y = [11, 15, 10]
        xout = [2, 3, 7, 9]
        myprop = self.create_propagate_wrapper(x, xout, 'lin-lin')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'lin-lin').toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_log_lin_propagate(self):
        x = np.exp([1, 5, 10])
        y = [11, 15, 10]
        xout = np.exp([2, 3, 7, 9])
        myprop = self.create_propagate_wrapper(x, xout, 'log-lin')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'log-lin').toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_lin_log_sensitivity(self):
        x = [1, 5, 10]
        y = np.exp([11, 15, 10])
        xout = [2, 3, 7, 9]
        myprop = self.create_propagate_wrapper(x, xout, 'lin-log')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'lin-log').toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_log_log_sensitivity(self):
        x = np.exp([1, 5, 10])
        y = np.exp([11, 15, 10])
        xout = np.exp([2, 3, 7, 9])
        myprop = self.create_propagate_wrapper(x, xout, 'log-log')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'log-log').toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_mixed_sensitivity(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 3, 7])
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        interp_type = ['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin']
        myprop = self.create_propagate_wrapper(x, xout, interp_type)
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, interp_type).toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_permuted_sensitivity_calculation(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 3, 7])
        interp_type = np.array(['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin'])
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        # do the permutation
        np.random.seed(31)
        perm1 = np.random.permutation(len(x))
        perm2 = np.random.permutation(len(xout))
        x = x[perm1]; y = y[perm1]; interp_type = interp_type[perm1]
        xout = xout[perm2]
        # compare results
        myprop = self.create_propagate_wrapper(x, xout, interp_type)
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, interp_type).toarray()
        maxdiff = np.max(np.abs(Sref-Stest)/np.abs(Sref+1e-8))
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_permuted_sensitivity_calculation_in_another_way(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 3, 7])
        interp_type = np.array(['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin'])
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        # do the permutation
        np.random.seed(31)
        perm = np.random.permutation(len(x))
        x2 = x[perm]; y2 = y[perm]; interp_type2 = interp_type[perm]
        Smat1 = get_basic_sensmat(x, y, xout, interp_type).toarray()
        Smat2 = get_basic_sensmat(x2, y2, xout, interp_type2).toarray()
        res1 = Smat1 @ y
        res2 = Smat2 @ y2
        self.assertTrue(np.all(res1 == res2))

    def test_permuted_sensitivity_calculation_in_yet_another_way(self):
        x1 = [5, 3, 1, 7]
        y1 = [1, 9, 14, 23]
        perm = np.argsort(x1)
        x2 = np.array(x1, copy=True)[perm]
        y2 = np.array(y1, copy=True)[perm]
        xout = [1.4, 1.8, 3.7, 4.9, 5.2]
        possible_interp_types = ['lin-lin', 'lin-log', 'log-lin', 'log-log']
        for curint in possible_interp_types:
            errmsg = f'failed for interpolation type {curint}'
            Smat1 = get_basic_sensmat(x1, y1, xout, curint).toarray()
            Smat2 = get_basic_sensmat(x2, y2, xout, curint).toarray()
            res1 = Smat1 @ y1
            res2 = Smat2 @ y2
            self.assertTrue(np.all(np.isclose(res1, res2)), msg=errmsg)

    def test_sensitivity_calculation_with_a_single_element_in_x(self):
        x = [5]
        y = [10]
        xout = [5]
        possible_interp_types = np.array(['lin-lin', 'lin-log', 'log-lin', 'log-log'])
        for curint in possible_interp_types:
            myprop = self.create_propagate_wrapper(x, xout, curint)
            Sref = numeric_jacobian(myprop, y)
            Stest = get_basic_sensmat(x, y, xout, curint).toarray()
            self.assertTrue(np.all(np.isclose(Sref, Stest)))


if __name__ == '__main__':
    unittest.main()

