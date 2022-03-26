import unittest
import pathlib

from gmapi.legacy.database_reading import read_gma_database
from gmapi.legacy.conversion_utils import (sanitize_datablock,
        desanitize_datablock, compare_legacy_datablock_lists,
        augment_datablocks_with_NTOT, sanitize_prior,
        desanitize_prior)


class TestConversions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._dbpath = (pathlib.Path(__file__).parent / 'testdata' / 'data-2017-08-14.gma').resolve().as_posix()
        cls._dbdic = read_gma_database(cls._dbpath)

    @classmethod
    def tearDownClass(cls):
        del(cls._dbpath)
        del(cls._dbdic)

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
