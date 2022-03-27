import unittest
import pathlib
import numpy as np
import pandas as pd

from gmapi.legacy.database_reading import read_gma_database
from gmapi.legacy.conversion_utils import (sanitize_datablock, sanitize_prior)
from gmapi.data_management.uncfuns import (create_relunc_vector,
        create_experimental_covmat)
from gmapi.data_management.tablefuns import (create_prior_table,
        create_experiment_table)
from gmapi.mappings.priortools import attach_shape_prior
from gmapi.inference import iterative_gls_update
from gmapi.gmap import run_gmap


class TestLegacyDivergence(unittest.TestCase):

    def test_legacy_divergence(self):
        """Compare current to reference result."""
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        reftablepath = (pathlib.Path(__file__).parent / 'testdata' /
                'refoutput-2017-07-26.csv').resolve().as_posix()
        reftable = pd.read_csv(reftablepath, sep=';')

        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=True, legacy_output=False)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        self.assertTrue(np.all(np.isclose(res1, res2, atol=0, rtol=1e-12)))


if __name__ == '__main__':
    unittest.main()

