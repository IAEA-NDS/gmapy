### Important note

Please be aware that the test script `run_fort.sh`
contains a commit id for the corresponding Fortran
version. If it happens that the Fortran version is
updated then this commit id must also be updated.
Check the Fortran commit id in test_002 to see the
correct value.

Furthermore, in the past there was a bug in the
standards file: A EDBL statement was missing.
This has been corrected at some point in test_002.
If test_003 fails, make sure that the EDBL statement
is there by comparing with the data.gma file in
test_002. This is especially important when doing
debugging by bisection.
