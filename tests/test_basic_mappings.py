import unittest
import numpy as np
from scipy.sparse import csr_matrix
from gmapi.mappings.basic_maps import (basic_propagate, get_basic_sensmat,
        basic_multiply_Sdic_rows)
from gmapi.mappings.helperfuns import numeric_jacobian, return_matrix


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

    def test_correct_working_of_basic_multiply_Sdic_rows(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 25, 19])
        interp_type = ['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin']
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        facts = np.arange(len(xout))
        # applying the factor a posteriori
        Smat1 = get_basic_sensmat(x, y, xout, interp_type, ret_mat=True).toarray()
        res1 = np.ravel(Smat1 @ y) * facts
        # incorporating the factor in the sensitivity matrix
        Smat1 *= facts.reshape((len(xout), 1))
        res2 = np.ravel(Smat1 @ y)
        self.assertTrue(np.all(np.isclose(res1, res2)))
        # using my super-duper function to apply the multiplication factors
        # to the sensitivity matrix
        Sdic3 = get_basic_sensmat(x, y, xout, interp_type, ret_mat=False)
        basic_multiply_Sdic_rows(Sdic3, facts)
        Smat3 = return_matrix(idcs1=Sdic3['idcs1'], idcs2=Sdic3['idcs2'],
                vals=Sdic3['x'], dims=(len(xout), len(x)), how='csr').toarray()
        res3 = np.ravel(Smat3 @ y)
        self.assertTrue(np.all(np.isclose(res1, res3)))


class TestBasicMappingsJacobian(unittest.TestCase):

    def create_propagate_wrapper(self, x, xout, interp_type):
        def myprop(y):
            return basic_propagate(x, y, xout, interp_type)
        return myprop

    def test_lin_lin_sensitivity(self):
        x = [1, 5, 10]
        y = [11, 15, 10]
        xout = [2, 3, 7, 9]
        myprop = self.create_propagate_wrapper(x, xout, 'lin-lin')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'lin-lin', ret_mat=True).toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_log_lin_propagate(self):
        x = np.exp([1, 5, 10])
        y = [11, 15, 10]
        xout = np.exp([2, 3, 7, 9])
        myprop = self.create_propagate_wrapper(x, xout, 'log-lin')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'log-lin', ret_mat=True).toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_lin_log_sensitivity(self):
        x = [1, 5, 10]
        y = np.exp([11, 15, 10])
        xout = [2, 3, 7, 9]
        myprop = self.create_propagate_wrapper(x, xout, 'lin-log')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'lin-log', ret_mat=True).toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_log_log_sensitivity(self):
        x = np.exp([1, 5, 10])
        y = np.exp([11, 15, 10])
        xout = np.exp([2, 3, 7, 9])
        myprop = self.create_propagate_wrapper(x, xout, 'log-log')
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, 'log-log', ret_mat=True).toarray()
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_mixed_sensitivity(self):
        x = np.exp([1, 3, 5, 7, 9, 10])
        y = np.exp([11, 7, 15, 10, 3, 7])
        xout = np.exp([2, 3, 4.4, 5, 7, 9, 8, 2.2, 1.4])
        interp_type = ['lin-log', 'lin-lin', 'log-log', 'log-lin', 'log-log', 'lin-lin']
        myprop = self.create_propagate_wrapper(x, xout, interp_type)
        Sref = numeric_jacobian(myprop, y)
        Stest = get_basic_sensmat(x, y, xout, interp_type, ret_mat=True).toarray()
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
        Stest = get_basic_sensmat(x, y, xout, interp_type, ret_mat=True).toarray()
        maxdiff = np.max(np.abs(Sref-Stest)/np.abs(Sref+1e-8))
        self.assertTrue(np.all(np.isclose(Stest, Sref)))

    def test_sensitivity_calculation_with_a_single_element_in_x(self):
        x = [5]
        y = [10]
        xout = [5]
        possible_interp_types = np.array(['lin-lin', 'lin-log', 'log-lin', 'log-log'])
        for curint in possible_interp_types:
            myprop = self.create_propagate_wrapper(x, xout, curint)
            Sref = numeric_jacobian(myprop, y)
            Stest = get_basic_sensmat(x, y, xout, curint, ret_mat=True).toarray()
            self.assertTrue(np.all(np.isclose(Sref, Stest)))


if __name__ == '__main__':
    unittest.main()

