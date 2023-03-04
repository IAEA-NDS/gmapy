"""
Testing new mappings beyond legacy GMAP.
"""

import unittest
import numpy as np
import pandas as pd

from gmapy.mappings.mapping_elements import (
    InputSelector,
    Distributor
)
from gmapy.mappings.relative_error_map import (
    RelativeErrorMap
)


class TestRelativeErrorMap(unittest.TestCase):

    def test_relative_error_map(self):
        datatable = pd.DataFrame.from_records((
            ('exp_1000', 1, 1.),
            ('exp_1000', 2, 2.),
            ('exp_2000', 1, 3.),
            ('exp_2000', 3, 4.),
            ('model', 1, 5.),
            ('model', 2, 6.),
            ('model', 3, 7.),
            ('model', 4, 8.),
            ('relerr_1000', 2, 9.),
            ('relerr_2000', 1, 10.),
            ('relerr_1000', 1, 11.),
            ('relerr_2000', 3, 12.)
        ), columns=('NODE', 'PTIDX', 'VAL'))

        model_mask = datatable['NODE'].str.match('model')
        model_idcs = datatable[model_mask].index
        model_params = InputSelector(model_idcs, len(datatable))
        exp_mask = datatable['NODE'].str.match('exp_')
        exp_idcs = datatable[exp_mask].index
        model_preds = Distributor(model_params, exp_idcs, len(datatable))

        refvals = datatable['VAL'].to_numpy()
        model_params.assign(refvals)

        relerr = RelativeErrorMap(datatable, model_preds)
        real_result = relerr.propagate(refvals)
        expect_result = np.concatenate([
            [11*5, 9*6, 10*7, 12*8], np.zeros(8)
        ])
        self.assertTrue(np.all(real_result == expect_result))


if __name__ == '__main__':
    unittest.main()
