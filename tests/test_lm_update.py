import unittest
import pathlib
import pandas as pd
import numpy as np
from scipy.sparse import block_diag, diags
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table)
from gmapy.data_management.uncfuns import create_experimental_covmat
from gmapy.mappings.priortools import (
    prepare_prior_and_likelihood_quantities,
    attach_shape_prior,
    initialize_shape_prior,
    remove_dummy_datasets,
    create_propagate_source_mask,
    create_propagate_target_mask
)
from gmapy.inference import (
    superseded_lm_update,
    lm_update,
)
from gmapy.data_management.database_IO import read_legacy_gma_database
from gmapy.mappings.compound_map import CompoundMap
from gmapy.data_management.uncfuns import create_relunc_vector
from gmapy.posterior import Posterior


class TestNewLevenbergMarquardtUpdate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        db_dic = read_legacy_gma_database(dbpath)
        prior_list = db_dic['prior_list']
        datablock_list = db_dic['datablock_list']
        remove_dummy_datasets(datablock_list)

        priortable = create_prior_table(prior_list)
        priorcov = diags(np.square(priortable['UNC']), format='csc')

        exptable = create_experiment_table(datablock_list)
        expcov = create_experimental_covmat(datablock_list)

        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        datatable = attach_shape_prior(datatable)

        # the following block to prepare all the quantities
        # to call attach_shape_prior
        expsel = datatable['NODE'].str.match('exp_').to_numpy()
        refvals = datatable['PRIOR']
        reluncs = np.full(len(refvals), np.nan)
        reluncs[expsel] = create_relunc_vector(datablock_list)
        compmap = CompoundMap()
        initialize_shape_prior(datatable, compmap, refvals, reluncs)

        shapecov = diags(np.full(len(datatable)-len(priortable)-len(exptable), np.inf), format='csc')
        totcov = block_diag([priorcov, expcov, shapecov], format='csc')
        cls._dbpath = dbpath
        cls._datatable = datatable
        cls._totcov = totcov

    def test_lm_update_equivalent_lm_update(self):
        datatable = self._datatable
        totcov = self._totcov
        quants = prepare_prior_and_likelihood_quantities(datatable, totcov)
        priorvals = quants['priorvals']
        priorcov = quants['priorcov']
        expvals = quants['expvals']
        expcov = quants['expcov']
        adjidcs = priorcov.diagonal() != 0
        compmap1 = CompoundMap()
        compmap2 = CompoundMap(datatable, reduce=True)
        postdist = Posterior(priorvals, priorcov, compmap2, expvals, expcov)
        r1 = superseded_lm_update(
            compmap1, datatable, totcov, retcov=False, print_status=True, lmb=1
        )
        res1 = priorvals.copy()
        res1[adjidcs] = r1['upd_vals']
        print('\n\n----------------------------------------\n\n')
        r2 = lm_update(postdist, print_status=True, lmb=1.)
        res2 = r2['upd_vals']
        self.assertTrue(np.allclose(res1, res2, atol=1e-12, rtol=1e-12))

    def test_lm_update_convergence_with_ppp(self):
        datatable = self._datatable
        totcov = self._totcov
        quants = prepare_prior_and_likelihood_quantities(datatable, totcov)
        priortable = quants['priortable']
        exptable = quants['exptable']
        priorvals = quants['priorvals']
        priorcov = quants['priorcov']
        expvals = quants['expvals']
        expcov = quants['expcov']
        compmap = CompoundMap(datatable, reduce=True)
        source_mask = create_propagate_source_mask(priortable)
        target_mask = create_propagate_target_mask(exptable, mt6_exp=True)
        postdist = Posterior(
            priorvals, priorcov, compmap, expvals, expcov,
            relative_exp_errors=True, source_mask=source_mask,
            target_mask=target_mask
        )
        res = lm_update(postdist, print_status=True, maxiter=100,
                            must_converge=True, lmb=1., rtol=1e-6, atol=1e-6)
        best = res['upd_vals'].flatten()
        grad = postdist.grad_logpdf(best).flatten()
        self.assertTrue(np.all(np.abs(grad) < 1e-4))


if __name__ == '__main__':
    unittest.main()
