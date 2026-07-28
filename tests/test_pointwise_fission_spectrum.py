import unittest
import pathlib
import numpy as np
import tensorflow as tf
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import create_prior_table
from gmapy.data_management.uncfuns import create_prior_covmat
from gmapy.data_management.specialized_priorblock_apis.pointwise_fission_spectrum import (
    expand_interpolation_spec
)
from gmapy.gma_database_class import GMADatabase
from gmapy.mappings.helperfuns import (
    get_legacy_to_pointwise_fis_factors,
    get_fission_spectrum_interp
)
from gmapy.mappings.basic_integral_maps import (
    basic_integral_propagate,
    basic_integral_of_product_propagate
)
from gmapy.mappings.compound_map import CompoundMap
from gmapy.mappings.tf.compound_map_tf import CompoundMap as CompoundMapTF
from gmapy.mappings.tf.mapping_elements_tf import (
    IntegralLinLin,
    IntegralOfProductLinLin,
    IntegralInterp,
    IntegralOfProductInterp
)


class TestInterpolationSpec(unittest.TestCase):

    def test_expand_string_spec(self):
        res = expand_interpolation_spec('log-log', 4)
        self.assertEqual(res, ['log-log'] * 4)

    def test_expand_per_point_spec(self):
        spec = ['lin-lin', 'log-log', 'log-log']
        res = expand_interpolation_spec(spec, 3)
        self.assertEqual(res, spec)

    def test_expand_region_spec(self):
        spec = [{'last_index': 2, 'law': 'log-log'},
                {'last_index': 4, 'law': 'lin-lin'}]
        res = expand_interpolation_spec(spec, 5)
        # region up to point 2 governs segments with left points 0 and 1;
        # the segment starting at point 2 belongs to the second region
        self.assertEqual(
            res, ['log-log', 'log-log', 'lin-lin', 'lin-lin', 'lin-lin']
        )

    def test_expand_region_spec_last_point_convention(self):
        spec = [{'last_index': 3, 'law': 'log-log'},
                {'last_index': 4, 'law': 'lin-lin'}]
        res = expand_interpolation_spec(spec, 5)
        # law of last point governs no segment and repeats last region law
        self.assertEqual(
            res, ['log-log', 'log-log', 'log-log', 'lin-lin', 'lin-lin']
        )

    def test_invalid_specs(self):
        with self.assertRaises(ValueError):
            expand_interpolation_spec('cubic', 4)
        with self.assertRaises(IndexError):
            expand_interpolation_spec(['lin-lin'] * 3, 4)
        with self.assertRaises(ValueError):
            expand_interpolation_spec(
                [{'last_index': 2, 'law': 'lin-lin'}], 4
            )
        with self.assertRaises(ValueError):
            expand_interpolation_spec(
                [{'last_index': 2, 'law': 'lin-lin'},
                 {'last_index': 2, 'law': 'log-log'}], 3
            )


class TestPointwiseFissionSpectrumDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                  'data_and_sacs.json').resolve().as_posix()
        cls._rawdb = read_gma_database(dbpath)

    @classmethod
    def _get_fisblock(cls):
        for block in cls._rawdb['prior_list']:
            if block['type'] == 'legacy-fission-spectrum':
                return block
        raise IndexError('no fission spectrum block found')

    @classmethod
    def _create_pointwise_prior_list(cls, interp):
        fisblock = cls._get_fisblock()
        ens = np.array(fisblock['ENFIS'])
        vals = np.array(fisblock['FIS'])
        scl = get_legacy_to_pointwise_fis_factors(ens)
        dens = vals * scl
        assert np.all(dens > 0.)
        new_fisblock = {
            'type': 'pointwise-fission-spectrum',
            'energies': ens.tolist(),
            'values': dens.tolist(),
            'interpolation': interp
        }
        prior_list = [
            b if b['type'] != 'legacy-fission-spectrum' else new_fisblock
            for b in cls._rawdb['prior_list']
        ]
        return prior_list

    @classmethod
    def _create_gmadb(cls, prior_list):
        return GMADatabase(
            prior_list=prior_list,
            datablock_list=cls._rawdb['datablock_list'],
            remove_dummy=False
        )

    def test_interp_column_of_prior_table(self):
        priortable = create_prior_table(self._rawdb['prior_list'])
        fis_sel = priortable['NODE'] == 'fis'
        self.assertTrue(
            (priortable.loc[fis_sel, 'INTERP'] == 'legacy-binned').all()
        )
        self.assertTrue(
            (priortable.loc[~fis_sel, 'INTERP'] == 'lin-lin').all()
        )
        self.assertIsNone(
            get_fission_spectrum_interp(priortable[fis_sel])
        )
        pw_prior_list = self._create_pointwise_prior_list('log-log')
        pw_priortable = create_prior_table(pw_prior_list)
        pw_fis_sel = pw_priortable['NODE'] == 'fis'
        self.assertTrue(
            (pw_priortable.loc[pw_fis_sel, 'INTERP'] == 'log-log').all()
        )
        fis_interp = get_fission_spectrum_interp(pw_priortable[pw_fis_sel])
        self.assertTrue(np.all(fis_interp == 'log-log'))

    def test_mixed_interpretation_raises(self):
        priortable = create_prior_table(self._rawdb['prior_list'])
        fistable = priortable[priortable['NODE'] == 'fis'].copy()
        interp = fistable['INTERP'].to_list()
        interp[0] = 'log-log'
        fistable['INTERP'] = interp
        with self.assertRaises(ValueError):
            get_fission_spectrum_interp(fistable)

    def test_zero_uncertainties_equivalent_to_absent(self):
        prior_list1 = self._create_pointwise_prior_list('log-log')
        prior_list2 = [dict(b) for b in prior_list1]
        for b in prior_list2:
            if b['type'] == 'pointwise-fission-spectrum':
                b['uncertainties'] = [0.] * len(b['energies'])
        covmat1 = create_prior_covmat(prior_list1)
        covmat2 = create_prior_covmat(prior_list2)
        self.assertEqual((covmat1 != covmat2).nnz, 0)

    def test_correlated_uncertainties(self):
        prior_list = self._create_pointwise_prior_list('log-log')
        fisblock = None
        for b in prior_list:
            if b['type'] == 'pointwise-fission-spectrum':
                fisblock = b
        n = len(fisblock['energies'])
        uncs = np.linspace(0.01, 0.1, n)
        cormat = np.eye(n)
        cormat[0, 1] = cormat[1, 0] = 0.5
        fisblock['uncertainties'] = uncs.tolist()
        fisblock['correlations'] = cormat.tolist()
        covmat = create_prior_covmat(prior_list)
        priortable = create_prior_table(prior_list)
        fis_idcs = priortable.index[priortable['NODE'] == 'fis'].to_numpy()
        covmat = covmat.toarray()
        fiscov = covmat[np.ix_(fis_idcs, fis_idcs)]
        expected = cormat * np.outer(uncs, uncs)
        self.assertTrue(np.allclose(fiscov, expected))

    def test_pointwise_linlin_equivalent_to_legacy_binned(self):
        # a point-wise density obtained by applying the legacy-to-pointwise
        # conversion factors must reproduce the results obtained with
        # the binned spectrum representation
        gmadb1 = self._create_gmadb(self._rawdb['prior_list'])
        gmadb2 = self._create_gmadb(
            self._create_pointwise_prior_list('lin-lin')
        )
        res = []
        for gmadb in (gmadb1, gmadb2):
            dt = gmadb.get_datatable()
            compmap = CompoundMap(dt, reduce=False)
            preds = compmap.propagate(dt['PRIOR'].to_numpy())
            expsel = dt['NODE'].str.match('exp_').to_numpy()
            is_sacs = dt['REAC'].str.match('MT:(6|10)-').to_numpy() & expsel
            res.append(preds[is_sacs])
        self.assertTrue(np.allclose(res[0], res[1], rtol=1e-8))

    def test_spectrum_scale_invariance(self):
        # predictions of spectrum averaged cross sections and their ratios
        # must be invariant under a rescaling of the fission spectrum
        gmadb = self._create_gmadb(
            self._create_pointwise_prior_list('log-log')
        )
        dt = gmadb.get_datatable()
        compmap = CompoundMap(dt, reduce=False)
        expsel = dt['NODE'].str.match('exp_').to_numpy()
        is_sacs = dt['REAC'].str.match('MT:(6|10)-').to_numpy() & expsel
        fis_sel = (dt['NODE'] == 'fis').to_numpy()
        x1 = dt['PRIOR'].to_numpy()
        x2 = x1.copy()
        x2[fis_sel] *= 2.
        preds1 = compmap.propagate(x1)[is_sacs]
        preds2 = compmap.propagate(x2)[is_sacs]
        self.assertTrue(np.allclose(preds1, preds2, rtol=1e-8))

    def test_legacy_integration_with_pointwise_spectrum_raises(self):
        gmadb = self._create_gmadb(
            self._create_pointwise_prior_list('log-log')
        )
        dt = gmadb.get_datatable()
        with self.assertRaises(ValueError):
            compmap = CompoundMap(dt, reduce=False, legacy_integration=True)
            compmap.propagate(dt['PRIOR'].to_numpy())

    def _test_tf_nontf_equivalence(self, interp):
        gmadb = self._create_gmadb(
            self._create_pointwise_prior_list(interp)
        )
        dt = gmadb.get_datatable()
        compmap = CompoundMap(dt, reduce=True)
        compmap_tf = CompoundMapTF(dt, reduce=True)
        expsel = dt['NODE'].str.match('exp_').to_numpy()
        x = dt.loc[~expsel, 'PRIOR'].to_numpy()
        x_tf = tf.Variable(x, dtype=tf.float64)
        res = compmap.propagate(x)
        res_tf = compmap_tf(x_tf).numpy()
        self.assertTrue(np.allclose(res, res_tf, rtol=1e-4))
        jac = compmap.jacobian(x).toarray()
        grad = np.sum(jac, axis=0)

        def myscalarfun(x):
            return tf.reduce_sum(compmap_tf(x))

        with tf.GradientTape() as tape:
            summed_res_tf = myscalarfun(x_tf)
        grad_tf = tape.gradient(summed_res_tf, x_tf).numpy()
        # the non-tensorflow computation relies on Romberg integration
        # whose convergence criterion targets the total integral, so
        # narrow low-energy segments contributing little to the integral
        # can carry locally large errors that show up in the derivatives
        # with respect to the corresponding spectrum values; hence only
        # loose agreement with the exact segment integrals of the
        # tensorflow computation can be expected here and the tight
        # comparison is done in TestFissionAverageMapJacobian on a
        # small mesh as well as in TestTensorflowIntegralOps
        self.assertTrue(np.allclose(grad, grad_tf, rtol=1e-2, atol=2e-2))

    def test_tf_nontf_equivalence_loglog(self):
        self._test_tf_nontf_equivalence('log-log')

    def test_tf_nontf_equivalence_mixed_regions(self):
        fisblock = self._get_fisblock()
        n = len(fisblock['ENFIS'])
        ens = np.array(fisblock['ENFIS'])
        switch_idx = int(np.searchsorted(ens, 1.0))
        interp = [
            {'last_index': switch_idx, 'law': 'log-log'},
            {'last_index': n - 1, 'law': 'lin-lin'}
        ]
        self._test_tf_nontf_equivalence(interp)


class TestFissionAverageMapJacobian(unittest.TestCase):
    """Compare tf against non-tf fission average map on a small mesh.

    The small meshes permit strict tolerances for the Romberg
    integration underlying the Jacobian computation of the
    non-tensorflow mapping so that a tight comparison with the exact
    segment integrals of the tensorflow implementation is possible.
    """

    def _make_datatable(self):
        import pandas as pd
        xs_ens = np.array([1., 2., 5., 10.])
        xs_vals = np.array([2., 3., 1.5, 0.8])
        fis_ens = np.array([0.5, 1.5, 3., 7., 10.])
        fis_vals = np.array([0.3, 0.4, 0.2, 0.05, 0.01])
        fis_interp = ['log-log', 'lin-lin', 'log-log', 'log-log', 'log-log']
        priortable = pd.DataFrame.from_dict({
            'NODE': ['xsid_1'] * len(xs_ens) + ['fis'] * len(fis_ens),
            'REAC': (['MT:1-R1:1'] * len(xs_ens) +
                     ['MT:9999-R1:9999'] * len(fis_ens)),
            'ENERGY': np.concatenate([xs_ens, fis_ens]),
            'PRIOR': np.concatenate([xs_vals, fis_vals]),
            'INTERP': ['lin-lin'] * len(xs_ens) + fis_interp
        })
        exptable = pd.DataFrame.from_dict({
            'NODE': ['exp_3000'],
            'REAC': ['MT:6-R1:1'],
            'ENERGY': [1.5],
            'PRIOR': [0.],
            'DATA': [1.]
        })
        dt = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        return dt

    def test_fission_average_map_tf_equivalence(self):
        from gmapy.mappings.cross_section_fission_average_map import (
            CrossSectionFissionAverageMap
        )
        from gmapy.mappings.tf.cross_section_fission_average_map_tf import (
            CrossSectionFissionAverageMap as CrossSectionFissionAverageMapTF
        )
        dt = self._make_datatable()
        curmap = CrossSectionFissionAverageMap(
            dt, legacy_integration=False,
            atol=1e-6, rtol=1e-6, maxord=18, reduce=False
        )
        curmap_tf = CrossSectionFissionAverageMapTF(dt, reduce=False)
        x = dt['PRIOR'].to_numpy()
        x_tf = tf.Variable(x, dtype=tf.float64)
        res = curmap.propagate(x)
        res_tf = curmap_tf(x_tf).numpy()
        expsel = dt['NODE'].str.match('exp_').to_numpy()
        self.assertTrue(np.allclose(res[expsel], res_tf[expsel], rtol=1e-5))
        jac = curmap.jacobian(x).toarray()
        jac_tf = tf.sparse.to_dense(curmap_tf.jacobian(x_tf)).numpy()
        self.assertTrue(np.allclose(
            jac[expsel, :], jac_tf[expsel, :], rtol=1e-4, atol=1e-8
        ))


class TestTensorflowIntegralOps(unittest.TestCase):

    def test_integral_interp_all_linlin_matches_integral_linlin(self):
        x = np.array([1., 3., 4., 10.])
        y = np.array([2., 5., 3., 7.])
        yt = tf.constant(y, dtype=tf.float64)
        res1 = IntegralLinLin(x)(yt).numpy()
        res2 = IntegralInterp(x, 'lin-lin')(yt).numpy()
        self.assertTrue(np.allclose(res1, res2, rtol=1e-14))

    def test_integral_interp_loglog_powerlaw(self):
        # exact integral of x^2 between 1 and 8
        x = np.array([1., 2., 8.])
        y = x ** 2
        yt = tf.constant(y, dtype=tf.float64)
        res = IntegralInterp(x, 'log-log')(yt).numpy()
        expected = (8.**3 - 1.) / 3.
        self.assertTrue(np.allclose(res, expected, rtol=1e-12))

    def test_integral_interp_loglog_inverse_powerlaw(self):
        # p = -1 special case: integral of 1/x between 1 and 4
        x = np.array([1., 2., 4.])
        y = 1. / x
        yt = tf.constant(y, dtype=tf.float64)
        res = IntegralInterp(x, 'log-log')(yt).numpy()
        expected = np.log(4.)
        self.assertTrue(np.allclose(res, expected, rtol=1e-12))

    def test_integral_interp_matches_nontf_romberg(self):
        # NOTE: the Romberg integration reaches at best a relative
        # accuracy of about 1e-8 for power-law segments spanning
        # a decade, which limits the tolerance of this comparison
        x = np.array([1e-3, 1e-2, 0.1, 0.5, 1., 5., 20.])
        y = np.array([3e-4, 3e-3, 0.1, 0.9, 1.2, 0.4, 1e-3])
        interp = np.array(['log-log', 'log-log', 'lin-lin', 'log-log',
                           'lin-lin', 'log-log', 'log-log'])
        yt = tf.constant(y, dtype=tf.float64)
        res = IntegralInterp(x, interp)(yt).numpy()
        expected = basic_integral_propagate(
            x, y, interp, atol=1e-7, rtol=1e-7, maxord=18
        )
        self.assertTrue(np.allclose(res, expected, rtol=1e-6))

    def test_integral_of_product_interp_all_linlin_matches_linlin_op(self):
        x1 = np.array([1., 4., 10.])
        y1 = np.array([2., 3., 1.])
        x2 = np.array([0.5, 2., 5., 12.])
        y2 = np.array([1., 4., 2., 3.])
        y1t = tf.constant(y1, dtype=tf.float64)
        y2t = tf.constant(y2, dtype=tf.float64)
        res1 = IntegralOfProductLinLin(x1, x2)(y1t, y2t).numpy()
        res2 = IntegralOfProductInterp(x1, x2, 'lin-lin')(y1t, y2t).numpy()
        self.assertTrue(np.allclose(res1, res2, rtol=1e-14))

    def test_integral_of_product_interp_exact_powerlaw(self):
        # xs(x) = 2 + 3*x and spectrum(x) = x^(-1.5) on identical meshes:
        # exact integral of (2+3x)*x^(-1.5) from 1 to 9 is
        # [-4*x^(-1/2) + 6*x^(1/2)]_1^9
        x1 = np.array([1., 3., 9.])
        y1 = 2. + 3. * x1
        x2 = np.array([1., 3., 9.])
        y2 = x2 ** (-1.5)
        y1t = tf.constant(y1, dtype=tf.float64)
        y2t = tf.constant(y2, dtype=tf.float64)
        res = IntegralOfProductInterp(x1, x2, 'log-log')(y1t, y2t).numpy()
        expected = (-4./3. + 18.) - (-4. + 6.)
        self.assertTrue(np.allclose(res, expected, rtol=1e-12))

    def test_integral_of_product_interp_matches_nontf_romberg(self):
        x1 = np.array([1e-3, 1e-1, 0.5, 1., 20.])
        y1 = np.array([10., 3., 1., 2., 0.5])
        x2 = np.array([1e-3, 1e-2, 0.3, 0.5, 2., 20.])
        y2 = np.array([2e-5, 1e-3, 0.3, 1.1, 0.8, 1e-4])
        interp2 = np.array(['log-log', 'log-log', 'log-log',
                            'lin-lin', 'log-log', 'log-log'])
        y1t = tf.constant(y1, dtype=tf.float64)
        y2t = tf.constant(y2, dtype=tf.float64)
        res = IntegralOfProductInterp(x1, x2, interp2)(y1t, y2t).numpy()
        expected = basic_integral_of_product_propagate(
            [x1, x2], [y1, y2], ['lin-lin', interp2],
            zero_outside=True, atol=1e-7, rtol=1e-7, maxord=18
        )
        self.assertTrue(np.allclose(res, np.squeeze(expected), rtol=1e-6))

    def test_integral_of_product_interp_gradient_finite_differences(self):
        x1 = np.array([1., 3., 9.])
        y1 = np.array([2., 4., 3.])
        x2 = np.array([0.5, 2., 4., 9.])
        y2 = np.array([1., 0.25, 0.5, 0.1])
        interp2 = np.array(['log-log', 'lin-lin', 'log-log', 'log-log'])
        intop = IntegralOfProductInterp(x1, x2, interp2)
        y1t = tf.Variable(y1, dtype=tf.float64)
        y2t = tf.Variable(y2, dtype=tf.float64)
        with tf.GradientTape() as tape:
            res = intop(y1t, y2t)
        grads = tape.gradient(res, [y1t, y2t])
        grads = [tf.convert_to_tensor(g).numpy() for g in grads]
        h = 1e-6
        for k, (arr, grad) in enumerate(zip((y1, y2), grads)):
            for i in range(len(arr)):
                pert_up = arr.copy()
                pert_dn = arr.copy()
                pert_up[i] += h
                pert_dn[i] -= h
                args_up = [pert_up, y2] if k == 0 else [y1, pert_up]
                args_dn = [pert_dn, y2] if k == 0 else [y1, pert_dn]
                res_up = intop(*(tf.constant(a, dtype=tf.float64)
                                 for a in args_up)).numpy()
                res_dn = intop(*(tf.constant(a, dtype=tf.float64)
                                 for a in args_dn)).numpy()
                fd = float(np.squeeze(res_up) - np.squeeze(res_dn)) / (2.*h)
                self.assertTrue(np.isclose(grad[i], fd, rtol=1e-5, atol=1e-9))


if __name__ == '__main__':
    unittest.main()
