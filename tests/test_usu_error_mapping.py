import unittest
import pathlib
import numpy as np
import pandas as pd
from gmapi.data_management.database_IO import read_legacy_gma_database
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.mappings.priortools import attach_shape_prior
from gmapi.mappings.usu_error_map import USUErrorMap
from gmapi.mappings.compound_map import CompoundMap
from gmapi.mappings.helperfuns import numeric_jacobian


class TestUSUErrorMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        db_dic = read_legacy_gma_database(dbpath)
        prior_list = db_dic['prior_list']
        datablock_list = db_dic['datablock_list']

        priortable = create_prior_table(prior_list)
        exptable = create_experiment_table(datablock_list)
        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        datatable = attach_shape_prior(datatable)
        cls._datatable = datatable


    def test_usu_error_propagation(self):
        compmap = CompoundMap()
        usumap = USUErrorMap(compmap, ['FEAT'], NA_values=('NA', np.nan))
        dt = self._datatable
        # we add two USU components
        dt['FEAT'] = 'NA'
        dt.loc[dt.ENERGY < 4, 'FEAT'] = 'RNG1'
        dt.loc[dt.ENERGY > 10, 'FEAT'] = 'RNG2'
        # add the USU errors
        usu_dt = pd.DataFrame.from_dict({
            'NODE': 'usu_errors',
            'PRIOR': [0., 0., 5.],
            'FEAT': ('RNG1', 'RNG2', 'RNG3')
            })
        dt = pd.concat([dt, usu_dt], ignore_index=True)

        usu_sel = dt.NODE.str.match('usu_', na=False)
        exp_sel = dt.NODE.str.match('exp_', na=False)
        rng1_sel = dt.FEAT.str.fullmatch('RNG1', na=False)
        rng2_sel = dt.FEAT.str.fullmatch('RNG2', na=False)
        rng1_usu_sel = (usu_sel & rng1_sel)
        rng2_usu_sel = (usu_sel & rng2_sel)

        refvals1 = dt['PRIOR'].to_numpy()
        refvals2 = refvals1.copy()
        refvals2[rng1_usu_sel] = 0.1
        refvals2[rng2_usu_sel] = -0.2
        propvals = compmap.propagate(dt, dt['PRIOR'].to_numpy())
        propvals1 = usumap.propagate(dt, refvals1)
        propvals2 = usumap.propagate(dt, refvals2)
        # we expect the standard compound mapping and
        # USU mapping propagation to yield identical results
        # if the USU errors are set to zero
        self.assertTrue(np.allclose(propvals, propvals1,
            atol=1e-8, rtol=1e-8, equal_nan=True))
        # we further expect that a USU error of 0.1 (=10%)
        # will cause the USU mapping propagation to produce
        # a 10% larger result than the standard compound mapping,
        # and analogous for -20%
        rng1_exp_sel = (exp_sel & rng1_sel)
        rng2_exp_sel = (exp_sel & rng2_sel)
        self.assertTrue(np.allclose(propvals1[rng1_exp_sel]*1.1,
                                    propvals2[rng1_exp_sel],
                                    atol=1e-8, rtol=1e-8, equal_nan=True))
        self.assertTrue(np.allclose(propvals1[rng2_exp_sel]*0.8,
                                    propvals2[rng2_exp_sel],
                                    atol=1e-8, rtol=1e-8, equal_nan=True))
        # we expect that experiments that are neither associated
        # with RNG1 nor RNG2 are not affected by the USUMap
        other_sel = (exp_sel & ~rng1_exp_sel & ~rng2_exp_sel)
        self.assertTrue(np.allclose(propvals[other_sel],
                                    propvals2[other_sel],
                                    atol=1e-8, rtol=1e-8, equal_nan=True))


    def test_multi_feature_error_propagation(self):
        compmap = CompoundMap()
        usumap = USUErrorMap(compmap, ['FEAT1','FEAT2'], NA_values=('NA', np.nan))
        dt = self._datatable
        # we add two USU components
        usu_dt = pd.DataFrame.from_dict({
            'NODE': 'usu_errors',
            'PRIOR': [0.1, 0., 5., 0.2, -0.1, 0.3],
            'FEAT1': ('RNG1', 'RNG2', 'RNG3', 'NA', 'NA', 'NA'),
            'FEAT2': ('NA', 'NA', 'NA', 'A1', 'A2', 'A3')
            })
        dt = pd.concat([dt, usu_dt], ignore_index=True)

        usu_sel = dt.NODE.str.match('usu_', na=False)
        exp_sel = dt.NODE.str.match('exp_', na=False)
        refvals = dt.PRIOR.to_numpy()
        propvals = compmap.propagate(dt, refvals)
        rng1_sel = dt.ENERGY < 2
        rng3_sel = dt.ENERGY > 7
        a2_sel = dt.ENERGY < 4
        a3_sel = dt.ENERGY > 10
        dt.loc[rng1_sel, 'FEAT1'] = 'RNG1'
        dt.loc[rng3_sel, 'FEAT1'] = 'RNG3'
        dt.loc[a2_sel, 'FEAT2'] = 'A2'
        dt.loc[a3_sel, 'FEAT2'] = 'A3'
        usupropvals = usumap.propagate(dt, refvals)

        rng1_and_a2_sel = (exp_sel & rng1_sel & a2_sel)
        self.assertTrue(np.allclose(usupropvals[rng1_and_a2_sel],
            propvals[rng1_and_a2_sel] * (1 + 0.1 - 0.1)))
        rng3_and_a2_sel = (exp_sel & rng3_sel & a2_sel)
        self.assertTrue(np.allclose(usupropvals[rng3_and_a2_sel],
            propvals[rng3_and_a2_sel] * (1 + 5 - 0.1)))
        rng3_and_a3_sel = (exp_sel & rng3_sel & a3_sel)
        self.assertTrue(np.allclose(usupropvals[rng3_and_a2_sel],
            propvals[rng3_and_a2_sel] * (1 + 5 + 0.3)))


    def test_usu_jacobian(self):
        compmap = CompoundMap()
        usumap = USUErrorMap(compmap, ['FEAT'], NA_values=('NA', np.nan))
        dt = self._datatable
        # we add two USU components
        dt['FEAT'] = 'NA'
        dt.loc[dt.ENERGY < 4, 'FEAT'] = 'RNG1'
        dt.loc[dt.ENERGY > 10, 'FEAT'] = 'RNG2'
        # add the USU errors
        usu_dt = pd.DataFrame.from_dict({
            'NODE': 'usu_errors',
            'PRIOR': [0.1, -0.3, 5.],
            'FEAT': ('RNG1', 'RNG2', 'RNG3')
            })
        dt = pd.concat([dt, usu_dt], ignore_index=True)
        # convert NaN in prior column to 0, because
        # otherwise numeric jacobian and analytic construction
        # are expected to have NaN vs 0. differences
        dt.loc[dt.PRIOR.isna(), 'PRIOR'] = 0.

        usu_sel = dt.NODE.str.match('usu_', na=False)
        exp_sel = dt.NODE.str.match('exp_', na=False)
        rng1_sel = dt.FEAT.str.fullmatch('RNG1', na=False)
        rng2_sel = dt.FEAT.str.fullmatch('RNG2', na=False)
        rng1_usu_sel = (usu_sel & rng1_sel)
        rng2_usu_sel = (usu_sel & rng2_sel)

        def prop_wrap(x):
            refvals = dt.PRIOR.to_numpy()
            refvals[usu_sel] = x
            propvals = usumap.propagate(dt, refvals)
            return propvals[exp_sel]

        orig_refvals = dt.PRIOR.to_numpy()
        S1 = usumap.jacobian(dt, orig_refvals, ret_mat=True)
        S1red = S1[exp_sel,:][:,usu_sel].toarray()
        S2red = numeric_jacobian(prop_wrap, orig_refvals[usu_sel])
        self.assertTrue(np.allclose(S1red, S2red,
                                    atol=1e-8, rtol=1e-8))
        # also check that the compound mapping part coincides
        # with the subblock in the USU map Jacobian
        Scomp = compmap.jacobian(dt, orig_refvals, ret_mat=True)
        not_usu_sel = np.logical_not(usu_sel)
        S1red = S1[:, not_usu_sel].toarray()
        Scred = Scomp[:, not_usu_sel].toarray()
        self.assertTrue(np.all(S1red == Scred))


    def test_permutation_invariance(self):
        compmap = CompoundMap()
        usumap = USUErrorMap(compmap, ['FEAT'], NA_values=('NA', np.nan))
        dt = self._datatable
        # we add two USU components
        dt['FEAT'] = 'NA'
        dt.loc[dt.ENERGY < 4, 'FEAT'] = 'RNG1'
        dt.loc[dt.ENERGY > 10, 'FEAT'] = 'RNG2'
        # add the USU errors
        usu_dt = pd.DataFrame.from_dict({
            'NODE': 'usu_errors',
            'PRIOR': [0.1, -0.3, 5.],
            'FEAT': ('RNG1', 'RNG2', 'RNG3')
            })
        dt = pd.concat([dt, usu_dt], ignore_index=True)

        refvals = dt.PRIOR.to_numpy()
        permdt = dt.reindex(np.random.permutation(dt.index))
        propvals = usumap.propagate(dt, refvals)
        permpropvals = usumap.propagate(permdt, refvals)
        self.assertTrue(np.allclose(propvals, permpropvals,
                                    atol=1e-8, rtol=1e-8, equal_nan=True))
        # same for Jacobian matrix
        S = usumap.jacobian(dt, refvals, ret_mat=True)
        permS = usumap.jacobian(permdt, refvals, ret_mat=True)
        self.assertTrue((S != permS).nnz == 0)



if __name__ == '__main__':
    unittest.main()

