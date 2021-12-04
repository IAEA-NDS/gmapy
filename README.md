## About GMAP

GMAP developed by Wolfgang P. Poenitz is a code to obtain evaluations
of cross sections and their uncertainties based on the combined data
from various experiments. The code employs the Bayesian version
of the Generalized Least Squares method and is named after the
mathematicians Gauss, Markov, and Aitken, who all contributed
to the statistical theory around the linear least squares method.
The Fortran version and some more information on GMAP can be found
in [this repository](https://github.com/iaea-nds/GMAP-Fortran).

## Python version of GMAP 

As Python is a popular general-purpose language nowadays with many
modules to interface with web applications and databases, the Fortran
version of GMAP was translated to Python in order to facilitate future
developments and extensions.

### Launching the Python version

The following steps are necessary to run the Python version.
We recommend to use a package manager, such as `Anaconda`, and
to work in a new environment (e.g., `conda create -n gmap` and
then `conda activate gmap`).

1. Install the `fortranformat` and the `numpy` package.
```
  pip install numpy
  pip install fortranformat
```

2. Compile the `LINPACK` module to be accessible by Python
```
 cd source
 f2py -c linpack_slim.pyf linpack_slim.f
```
3. Point the `PYTHONPATH` to the directory of `GMAP.py`
```
 PYTHONPATH="<path-to-GMAP.py>"
```
4. Create a directory, create an input file inside named `data.gma`
(e.g., take it from `tests/test_001/input`), change into that directory
and launch GMAP:
```
python "<path-to-GMAP.py>/GMAP.py"
```

### Comments on conversion

- This Python version of GMAP was initially a line-by-line translation of the
original Fortran code including goto-statements enabled by the
[goto-statement] module.
Afterwards it has been extensively refactored to
improve readability. All goto-statements have been removed and replaced
by other control structures, such as if-blocks and loop control statements
and consequently the dependence on the [goto-statement] module has been removed.

- The [fortranformat] module is used to read and write files using the
same Fortran format descriptors as present in the original Fortran code.

- The Fortran version relies on `LINPACK` routines for Cholesky decomposition and
inversion whereas the `scipy` module in Python employs `LAPACK`---the successor
of `LINPACK`. At the moment, the Python version does not use `scipy` 
but instead uses the `LINPACK` functions in file `linpack_slim.f`. The reason
being that some values in the input file (`data.gma`) are only given with four
digits after the comma and in combination with different algorithmic
implementations could lead to small numerical differences in the result.

- The Fortran version processed incomplete datablocks (i.e., end of block indicator EDBL
immediately followed by DATA keyword). Since commit 88fc22c the Python version does
not reproduce the results of the Fortran version anymore in the case of an incomplete
datablock.

- If it is unknown to the Fortran GMAP version how to link specific measurement
points to the prior, it proceeds but ignores those measurement points.
The Python version does not accept unusable datapoints anymore since commit 07da8f1.

[goto-statement]: https://pypi.org/project/goto-statement/
[fortranformat]: https://pypi.org/project/fortranformat/ 

### Ensuring functional equivalence

To ensure the functional equivalence of the Python and Fortran version,
a test case was introduced that compares the output of the Python and Fortran
version. As both versions work with double precision, implement
the same code logic and rely on the same Fortran functions for the
Cholesky decomposition and inversion, they produce exactly the same output.
The test case is written in bash and has only been tested with Linux.

To run the test case, change into the `tests/test_002` directory and run
```
./run_fort.sh
```
The return code accessible by typing `$?` is zero if successful.
If differences exist, they will be written out to the console.
Please note that if you re-run the test case after a modification of the Python
source code, you should first delete the `result` directory in `tests/test_002`.

