import unittest
import pathlib
import numpy as np
from pandas.testing import assert_frame_equal
from gmapi.gmap import run_gmap_simplified
from gmapi.legacy.legacy_gmap import run_gmap


class TestGmapSimplified(unittest.TestCase):

    def test_GMAP_and_GMAP_simplified_equivalence(self):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        # the simplified one without the possibility to reproduce
        # the bugs of the legacy version
        res2 = run_gmap_simplified(dbfile=dbpath, dbtype='legacy',
                num_iter=3, correct_ppp=True)
        # the version with more comprehensive input options
        res1 = run_gmap(dbfile=dbpath, dbtype='legacy', legacy_output=False,
                num_iter=3, correct_ppp=True,
                fix_ppp_bug=True, fix_sacs_jacobian=True, legacy_integration=False)
        # compare all the results
        self.assertTrue(np.all(res1['postcov'] == res2['postcov']))
        raised = False
        try:
            assert_frame_equal(res1['table'], res2['table'])
        except:
            raised = True
        self.assertFalse(raised, 'result tables not equal')

        p1 = res1['priorcov']
        p2 = res2['priorcov']
        self.assertTrue(np.all(p1.indices == p2.indices))
        self.assertTrue(np.all(p1.indptr == p2.indptr))
        self.assertTrue(np.allclose(p1.data, p2.data, equal_nan=True))


if __name__ == '__main__':
    unittest.main()

