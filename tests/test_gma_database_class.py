import unittest
import pathlib
import pandas as pd
from gmapi.gma_database_class import GMADatabase
import numpy as np
from copy import deepcopy
from scipy.sparse import csr_matrix



class TestGMADatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        cls._gmadb = GMADatabase(dbpath)

    def test_datapoints_removal(self):
        gmadb = deepcopy(self._gmadb)
        origcovmat = gmadb.get_covmat()
        origtable = gmadb.get_datatable()
        remove_idcs = origtable.index[origtable.NODE.str.fullmatch('exp_1034')]
        gmadb.remove_data(remove_idcs)
        newtable = gmadb.get_datatable()
        newcovmat = gmadb.get_covmat()
        # check for the expected change of dimensions
        self.assertTrue(len(origtable)-len(remove_idcs) == len(newtable))
        self.assertTrue(origcovmat.shape[0]-len(remove_idcs) ==
                newcovmat.shape[0])
        self.assertTrue(origcovmat.shape[1]-len(remove_idcs) ==
                newcovmat.shape[1])
        # manually remove and check if results coincide
        keep_idcs = np.full(len(origtable), True)
        keep_idcs[remove_idcs] = False
        redcovmat = origcovmat[keep_idcs,:][:,keep_idcs]
        self.assertTrue((redcovmat != newcovmat).nnz == 0)
        redtable = origtable.drop(remove_idcs, inplace=False)
        redtable.reset_index(drop=True, inplace=True)
        self.assertTrue(redtable.equals(newtable))

    def test_evaluation_with_temporary_datapoints_removal(self):
        gmadb1 = deepcopy(self._gmadb)
        gmadb2 = deepcopy(self._gmadb)
        tbl = gmadb1.get_datatable()
        remove_idcs = tbl.index[tbl.NODE.str.fullmatch('exp_1034')]
        keep_idcs = np.full(len(tbl), True)
        keep_idcs[remove_idcs] = False
        gmadb1.evaluate(remove_idcs=remove_idcs, maxiter=2)
        gmadb2.remove_data(remove_idcs)
        gmadb2.evaluate(maxiter=2)
        res1 = gmadb1.get_datatable().loc[keep_idcs, 'POST'].to_numpy()
        res2 = gmadb2.get_datatable()['POST'].to_numpy()
        # remove fission spectrum, because it is not propagated at present
        redtbl = gmadb2.get_datatable()
        not_fis_idcs = np.logical_not(redtbl.NODE == 'fis')
        # equal_nan is True because of fission spectrum
        self.assertTrue(np.all(res1[not_fis_idcs] == res2[not_fis_idcs]))

    def test_covmat_passed_by_copy(self):
        gmadb = deepcopy(self._gmadb)
        # check we do not get a reference by get_covmat
        covmat1 = gmadb.get_covmat()
        covmat2 = gmadb.get_covmat()
        self.assertTrue(covmat1 is not covmat2)
        # check we do not pass a reference by set_covmat
        gmadb.set_covmat(covmat1)
        covmat2 = gmadb.get_covmat()
        self.assertTrue(covmat1 is not covmat2)

    def test_datatable_passed_by_copy(self):
        gmadb = deepcopy(self._gmadb)
        # check we do not get a reference by get_datatable
        datatable1 = gmadb.get_datatable()
        datatable2 = gmadb.get_datatable()
        self.assertTrue(datatable1 is not datatable2)
        # check we do not pass a reference by set_datatable
        gmadb.set_datatable(datatable1)
        datatable2 = gmadb.get_datatable()
        self.assertTrue(datatable1 is not datatable2)

    def test_change_of_covmat_dimension_fails(self):
        gmadb = deepcopy(self._gmadb)
        covmat = gmadb.get_covmat()
        origdim = covmat.shape
        newcovmat = covmat[-1:,:]
        self.assertRaises(Exception, gmadb.set_covmat, newcovmat)
        # way of checking for equality of sparse matrices
        self.assertTrue((covmat != gmadb.get_covmat()).nnz == 0)

    def test_change_of_datatable_dimension_fails(self):
        gmadb = deepcopy(self._gmadb)
        oldtable = gmadb.get_datatable()
        newtable = oldtable.drop(5, inplace=False)
        self.assertRaises(Exception, gmadb.set_datatable, newtable)
        table = gmadb.get_datatable()
        self.assertTrue(table.equals(oldtable))

    def test_changing_datatable_has_intended_effects(self):
        gmadb = deepcopy(self._gmadb)
        olddatatable = gmadb.get_datatable()
        oldcovmat = gmadb.get_covmat()
        newdatatable = olddatatable.copy()
        # we only change the uncertainty in the datatable
        newdatatable.loc[7, 'UNC'] = 15
        gmadb.set_datatable(newdatatable)
        # expect the change to be adopted in the datatable
        datatable = gmadb.get_datatable()
        covmat = gmadb.get_covmat()
        self.assertTrue(datatable.equals(newdatatable))
        # but also expect a corresponding change in the covariance matrix
        self.assertTrue(covmat[7,7] == np.square(15))
        redcovmat1 = csr_matrix(np.delete(oldcovmat.toarray(), 7, axis=0))
        redcovmat2 = csr_matrix(np.delete(covmat.toarray(), 7, axis=0))
        self.assertTrue((redcovmat1 != redcovmat2).nnz == 0)
        covmat[7,7] = 0.
        self.assertTrue(np.all(covmat[7,:].toarray() == 0.))
        self.assertTrue(np.all(covmat[:,7].toarray() == 0.))

    def test_changing_covariance_has_intended_effects(self):
        gmadb = deepcopy(self._gmadb)
        table = gmadb.get_datatable()
        covmat = gmadb.get_covmat()
        covmat[2000,2000] = np.square(15)
        gmadb.set_covmat(covmat)
        newtable = gmadb.get_datatable()
        self.assertTrue(newtable.loc[2000, 'UNC'] == 15)
        newtable.loc[2000, 'UNC'] = table.loc[2000, 'UNC']
        self.assertTrue(newtable.equals(table))

    def test_update_datatable_gives_sorted_datatable(self):
        gmadb = deepcopy(self._gmadb)
        oldtable = gmadb.get_datatable()
        newtable = oldtable.sample(frac=1)
        gmadb.set_datatable(newtable)
        newtable = gmadb.get_datatable()
        self.assertTrue(oldtable.equals(newtable))



if __name__ == '__main__':
    unittest.main()

