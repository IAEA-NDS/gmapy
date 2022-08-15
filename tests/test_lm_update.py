import unittest
import pathlib
import pandas as pd
import numpy as np
from scipy.sparse import block_diag, diags
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.data_management.uncfuns import create_experimental_covmat
from gmapi.mappings.priortools import attach_shape_prior
from gmapi.inference import gls_update, lm_update
from gmapi.data_management.database_IO import read_legacy_gma_database
from gmapi.mappings.compound_map import CompoundMap



class TestLevenbergMarquardtUpdate(unittest.TestCase):

    def test_gls_lm_equivalence(self):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        db_dic = read_legacy_gma_database(dbpath)
        prior_list = db_dic['prior_list']
        datablock_list = db_dic['datablock_list']

        priortable = create_prior_table(prior_list)
        priorcov = diags(np.square(priortable['UNC']), format='csc') 

        exptable = create_experiment_table(datablock_list)
        expcov = create_experimental_covmat(datablock_list)

        datatable = pd.concat([priortable, exptable], axis=0, ignore_index=True)
        datatable = attach_shape_prior(datatable)
        shapecov = diags(np.full(len(datatable)-len(priortable)-len(exptable), np.inf), format='csc')
        totcov = block_diag([priorcov, expcov, shapecov], format='csc')

        compmap = CompoundMap()

        res1 = gls_update(compmap, datatable, totcov, retcov=False)
        res2 = lm_update(compmap, datatable, totcov, retcov=False) 
        self.assertTrue(np.all(np.isclose(res1['upd_vals'], res2['upd_vals'],
            atol=1e-8, rtol=1e-8)))



if __name__ == '__main__':
    unittest.main()

