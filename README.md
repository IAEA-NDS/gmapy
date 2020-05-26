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

The following steps are necessary to run the Python version:

1. Compile the `LINPACK` module to be accessible by Python
```
 cd source
 f2py -c linpack_slim.pyf linpack_slim.f
```
2. Point the `PYTHONPATH` to the directory of `GMAP.py`
```
 PYTHONPATH="<path-to-GMAP.py>/source"
```
3. Create a directory, create an input file inside named `data.gma`
(e.g., take it from `tests/test_001/input`), change into that directory
and launch GMAP:
```
python "<path-to-GMAP.py>/GMAP.py"
```

### Comments on conversion

- Presently, the Python version in this repository is a line-by-line translation
of the original Fortran code.
The `goto` statement was introduced using the
[goto-statement] module and reading and writing information to files and
console using Fortran format descriptors was achieved by relying on the
[fortranformat] module.

- A particularity in the translated version is the
artificial incrementing of counter variables after loops---the reason for that
being that a loop in Fortran up to `N` will leave the counter variable at `N+1`
after the termination of the loop whereas it is `N` in Python.

- The Fortran version relies on `LINPACK` routines for Cholesky decomposition and
inversion whereas the `scipy` module in Python employs `LAPACK`---the successor
of `LINPACK`. At the moment, the Python version does not use `scipy` 
but instead uses the `LINPACK` functions in file `linpack_slim.f`. The reason
being that some values in the input file (`data.gma`) are only given with four
digits after the comma and in combination with different algorithmic
implementations could lead to small numerical differences in the result.

[goto-statement]: https://pypi.org/project/goto-statement/
[fortranformat]: https://pypi.org/project/fortranformat/ 

### Ensuring functional equivalence

To ensure the functional equivalence of the Python and Fortran version,
a test case was introduced that compares the output of the Python and Fortran
version. As both versions work with double precision, implement
the same code logic and rely on the same Fortran functions for the
Cholesky decomposition and inversion, they produce exactly the same output.

### Additional `debug` branch

During the creation of this Python version based on the Fortran version,
statements have been introducedd to write additional information to file
`debug.out` for debugging and ensuring the equivalence of the codes.
The Python code with these additional statements is stored in the
`debug` branch and those extra code blocks are enclosed by
`if VERIFY ... #endif` to facilitate enabling and disabling them.
