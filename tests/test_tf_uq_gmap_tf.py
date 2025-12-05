import unittest

from pathlib import Path
from gmapy.gmap import run_gmap_simplified
from gmapy.data_management.database_IO import read_gma_database
import pandas as pd
import numpy as np
from gmapy.mappings.priortools import (
    remove_dummy_datasets
)
from gmapy.legacy.legacy_gmap import run_gmap as run_gmap_legacy
from gmapy.data_management.tablefuns import (
    create_experiment_table,
)
from gmapy.tf_uq.gmap_tf import evaluate_gma_database as evaluate_gma
from gmapy.legacy.data_extraction_functions import read_gma_result


class TestGmaTensorFlowModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        script_path = Path(__file__).resolve()
        script_dir = script_path.parent
        cls._gma_input_file = (
            script_dir.parent / 'legacy-tests' / 'test_002' / 'input' / 'data.gma'
        )
        cls._gma_reference_result_file = (
            script_dir.parent / 'legacy-tests' / 'test_002' / 'input' / 'gma_ref.res'
        )

    def test_run_gmap_legacy_reproduces_fortran_gmap(self):
        gma_res_raw = read_gma_result(self._gma_reference_result_file)
        gma_res = gma_res_raw[['NODE', 'REAC', 'ENERGY', 'RESULT']].copy()
        gma_res = gma_res.rename(columns={'RESULT': 'GMA_RESULT'})

        gmapy_res2_raw = run_gmap_legacy(
            dbfile=self._gma_input_file, dbtype='legacy', num_iter=3,
            correct_ppp=True, remove_dummy=False, legacy_output=False,
            fix_ppp_bug=False, fix_sacs_jacobian=False, legacy_integration=True,
        )

        df1 = gma_res.copy()
        df2 = gmapy_res2_raw['table']
        df2 = df2[df2.NODE.str.startswith('xsid_')].copy()

        self.assertTrue((df1['NODE'] == df2['NODE']).all())
        self.assertTrue((df1['ENERGY'] == df2['ENERGY']).all())
        self.assertTrue(np.allclose(df1['GMA_RESULT'], df2['POST'], rtol=1e-4))

    def test_gmap_tf_reproduces_run_gmap_simplified(self):
        gmapy_res1_raw = run_gmap_simplified(
            dbfile=self._gma_input_file, dbtype='legacy', num_iter=2,
            correct_ppp=True, remove_dummy=True, reg=1e-4,
        )

        gmadb = read_gma_database(self._gma_input_file)
        remove_dummy_datasets(gmadb['datablock_list'])
        exptable = create_experiment_table(gmadb['datablock_list'])

        gmapy_res2_raw = evaluate_gma(
            gmadb['prior_list'], gmadb['datablock_list'],
            remove_dummy=True, mt6_ppp=False,
            optim_type='iterative-gls',
            optim_opts = {
                'max_iters': 3, 'rel_tol': 1e-40, 'rel_damp_unc': np.sqrt(1e4), 'must_converge': False,
            }
        )

        df1 = gmapy_res1_raw['table'].copy()
        df1 = df1[df1.NODE.str.match('^xsid_|^norm_')].reset_index(drop=True)

        df2 = gmapy_res2_raw['table'].copy()
        df2 = df2[df2.NODE != 'fis'].reset_index(drop=True)

        self.assertTrue((df1.NODE == df2.NODE).all())
        self.assertTrue((df1.ENERGY == df2.ENERGY).all())
        self.assertTrue(np.allclose(df1.POST, df2.POST))
