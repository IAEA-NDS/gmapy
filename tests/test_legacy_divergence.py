import unittest
import pathlib
import os
import numpy as np
import pandas as pd

from gmapy.legacy.legacy_gmap import run_gmap


class TestLegacyDivergence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dbpath = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26.gma').resolve().as_posix()
        reftablepath = (pathlib.Path(__file__).parent / 'testdata' /
                'refoutput-2017-07-26.csv').resolve().as_posix()
        reftable = pd.read_csv(reftablepath, sep=';')

        dbpath_noppp = (pathlib.Path(__file__).parent / 'testdata' /
                'data-2017-07-26-disabled-ppp-option.gma').resolve().as_posix()
        reftablepath_noppp = (pathlib.Path(__file__).parent / 'testdata' /
                'refoutput-2017-07-26-disabled-ppp-option.csv').resolve().as_posix()
        reftable_noppp = pd.read_csv(reftablepath_noppp, sep=';')
        reftable = pd.read_csv(reftablepath, sep=';')
        try:
            os.mkdir('testoutput')
        except FileExistsError:
            pass

        # remove the fis values because they are constant
        reftable = reftable[reftable['NODE'] != 'fis']
        reftable.reset_index(drop=True, inplace=True)

        reftable_noppp = reftable_noppp[reftable_noppp['NODE'] != 'fis']
        reftable_noppp.reset_index(drop=True, inplace=True)

        cls._dbpath = dbpath
        cls._reftable = reftable
        cls._dbpath_noppp = dbpath_noppp
        cls._reftable_noppp = reftable_noppp

    @classmethod
    def tearDownClass(cls):
        del(cls._dbpath)
        del(cls._reftable)
        del(cls._dbpath_noppp)
        del(cls._reftable_noppp)

    def print_comparison_info(self, msg, reftable, res2, relerr, fprefix=None):
        print(msg)
        probvals = [0.75, 0.9, 0.95, 0.99, 0.995, 0.999]
        qvals = [np.quantile(relerr, p) for p in probvals]
        pq = zip(probvals, qvals)
        cmptable = reftable.copy()
        cmptable['MYPOST'] = res2
        cmptable['RELERR'] = relerr
        relerr2 = cmptable['RELERR'] / cmptable['RELPOSTUNC']
        cmptable['RELERR2'] = relerr2
        perm = np.argsort(relerr2)[::-1]
        if fprefix is not None:
            fname = fprefix + '.xlsx'
            fpath = os.path.join('testoutput', fname)
            fname_sorted = fprefix + '_sorted.xlsx'
            fpath_sorted = os.path.join('testoutput', fname_sorted)
            cmptable.to_excel(fpath)
            sorted_cmptable = cmptable.sort_values(by='RELERR2', ignore_index=True, ascending=False)
            sorted_cmptable.to_excel(fpath_sorted)
        print(cmptable.iloc[perm[:50]])
        extramsg = '\n'.join([f'p-val: {x[0]} - q-val: {x[1]}' for x in pq])
        print(extramsg)

    def test_ppp_fix_irrelevant_if_ppp_option_disabled(self):
        dbpath = self._dbpath
        upd_res1 = run_gmap(dbpath, num_iter=3, fix_ppp_bug=False,
                  correct_ppp=False, legacy_output=False)
        upd_res2 = run_gmap(dbpath, num_iter=3, fix_ppp_bug=True,
                  correct_ppp=False, legacy_output=False)
        res1 = upd_res1['table']['POST'].to_numpy()
        res2 = upd_res2['table']['POST'].to_numpy()
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        self.assertLess(maxrelerr, 1e-8)

    def test_equivalence_if_ppp_option_disabled(self):
        dbpath = self._dbpath_noppp
        reftable = self._reftable_noppp
        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=False, legacy_output=False,
                fix_sacs_jacobian=False)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        idcs = upd_res['idcs']
        res2 = res2[idcs]
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        self.assertLess(maxrelerr, 1e-8)

    def test_equivalence_if_ppp_bug_not_fixed(self):
        dbpath = self._dbpath
        reftable = self._reftable
        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=True, legacy_output=False,
                fix_ppp_bug=False, fix_sacs_jacobian=False)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        idcs = upd_res['idcs']
        res2 = res2[idcs]
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        self.assertLess(maxrelerr, 1e-8)

    def test_deviation_if_ppp_bug_fixed(self):
        dbpath = self._dbpath
        reftable = self._reftable
        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=True, legacy_output=False,
                fix_ppp_bug=True, fix_sacs_jacobian=False)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        idcs = upd_res['idcs']
        res2 = res2[idcs]
        # just select the cross sections and
        # skip the normalization constants
        sel = reftable['NODE'].str.match('xsid')
        res1 = res1[sel]
        res2 = res2[sel]
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        titlemsg = '### with ppp correction and ppp bug fixed  ###'
        self.print_comparison_info(titlemsg, reftable[sel], res2, relerr,
                fprefix = 'testoutput-comparison-results-ppp-bug-fixed')
        # We expect deviations due to fixing the PPP bug
        # to lead to differences of more than 0.5% but
        # not more than 0.7%
        self.assertGreater(maxrelerr, 5e-3)
        self.assertLess(maxrelerr, 7e-3)

    def test_deviation_if_no_ppp_correction(self):
        dbpath = self._dbpath
        reftable = self._reftable
        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=False, legacy_output=False,
                fix_ppp_bug=True, fix_sacs_jacobian=False)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        idcs = upd_res['idcs']
        res2 = res2[idcs]
        # just select the cross sections and
        # skip the normalization constants
        sel = reftable['NODE'].str.match('xsid')
        res1 = res1[sel]
        res2 = res2[sel]
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        titlemsg = '### no ppp correction ###'
        self.print_comparison_info(titlemsg, reftable[sel], res2, relerr,
                fprefix = 'testoutput-comparison-results-ppp-vs-noppp-correction')
        # We expect deviations to a calculaton with and
        # without PPP treatment to differ by more than
        # 3% but nowhere more than 5%
        self.assertGreater(maxrelerr, 0.03)
        self.assertLess(maxrelerr, 0.05)

    def test_deviation_if_correct_sacs_jacobian(self):
        dbpath = self._dbpath
        reftable = self._reftable
        upd_res = run_gmap(dbpath, num_iter=3,
                correct_ppp=True, legacy_output=False,
                fix_ppp_bug=False, fix_sacs_jacobian=True)
        res1 = reftable['POST'].to_numpy()
        res2 = upd_res['table']['POST'].to_numpy()
        idcs = upd_res['idcs']
        res2 = res2[idcs]
        relerr = np.abs(res1-res2) / np.maximum(1e-8, res2)
        maxrelerr = np.max(relerr)
        #self.print_comparison_info(titlemsg, reftable[sel], res2, relerr)
        self.assertGreater(maxrelerr, 1e-5)
        self.assertLess(maxrelerr, 1e-4)


if __name__ == '__main__':
    unittest.main()

