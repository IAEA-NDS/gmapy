import unittest
import pathlib
from tempfile import TemporaryDirectory
import numpy as np

from gmapi.gmap import run_gmap
from gmapi.legacy.database_reading import read_gma_database
from gmapi.legacy.conversion_utils import (sanitize_datablock,
        desanitize_datablock, compare_legacy_datablock_lists,
        augment_datablocks_with_NTOT, sanitize_prior,
        desanitize_prior, convert_GMA_database_to_JSON)


class TestConversions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._dbpath = (pathlib.Path(__file__).parent / 'testdata' / 'data-2017-08-14.gma').resolve().as_posix()
        cls._dbdic = read_gma_database(cls._dbpath)

    @classmethod
    def tearDownClass(cls):
        del(cls._dbpath)
        del(cls._dbdic)

    def test_result_equivalence_for_converted_database(self):
        dbpath = self._dbpath
        with TemporaryDirectory() as d:
            dirpath = str(d)
            jsonpath = (pathlib.Path(dirpath) / 'gmadata.json').resolve().as_posix()
            convert_GMA_database_to_JSON(dbpath, jsonpath)
            upd_res1 = run_gmap(dbpath, dbtype='legacy',
                    num_iter=1, legacy_output=False)
            upd_res2 = run_gmap(jsonpath, dbtype='json',
                    num_iter=1, legacy_output=False)
            res1 = upd_res1['table']['POST'].to_numpy()
            res2 = upd_res2['table']['POST'].to_numpy()
            relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
            maxrelerr = np.max(relerr)
            self.assertLess(maxrelerr, 1e-8)

    def test_datablock_conversion(self):
        dbdic = self._dbdic
        blocklist = dbdic['datablock_list'] 
        new_blocklist = [sanitize_datablock(b) for b in blocklist]
        old_blocklist = [desanitize_datablock(b) for b in new_blocklist]
        augment_datablocks_with_NTOT(old_blocklist)
        cmpval = compare_legacy_datablock_lists(blocklist, old_blocklist)
        self.assertTrue(cmpval)

    def test_prior_conversion(self):
        dbdic = self._dbdic
        prior = dbdic['APR']
        priorlist = sanitize_prior(prior)
        oldprior = desanitize_prior(priorlist)
        cmpval = compare_legacy_datablock_lists([oldprior], [prior])
        self.assertTrue(cmpval)


if __name__ == '__main__':
    unittest.main()
