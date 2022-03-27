### Legacy tests

These legacy tests were used to ensure the functional
equivalence of the Python and the Fortran
version of the GMAP code. Each test contains a
`run_test.sh` file that runs the test and returns
exit code 0 if the test passess, otherwise a different
value.

Only `test_002` was systematically run for each
commit to check the functional equivalence. The
other tests were added later and only
occassionally used. Hence the other tests may
do not pass for  all commits made in the past.
In particular, the Fortran code was updated a
few times due to bugs found therein, and 
only the script of `test_002` updated accordingly
with the commid id of the updated Fortran code.
If a test fails, the test script can be compared
to the one of `test_002` to see whether this is
the case.

However, with the exception of `test_001`,
the Python code passed all legacy tests
at version 0.11
(commit id `c43c30f82953fb6e52dc365138622800fa5c88a7`)
The test `test_001` failed at some point because
the Python code aborts for user correlation matrices
with diagonal elements greater one, whereas the
Fortran code proceeds in this scenario and forces 
the diagonal elements to be one.

After version 0.11, the Python code will be
extended/improved in a way that will break the
perfect numerical equivalence of results. The legacy
tests are therefore expected to fail from this
moment onwards.

Quality assurance is from now on established by
relying on the tests in the `tests/` directory
which make use of the Python `unittest` framework.
Especially the test `test_legacy_divergence.py`
is planned to keep track of the magnitude of
differences in the numerical results between
the legacy result as produced by `test_002`
and the result of the new Python code.

