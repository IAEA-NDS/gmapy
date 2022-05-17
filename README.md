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
2. Point the `PYTHONPATH` to the directory of `GMAP.py`
```
 PYTHONPATH="<path-to-GMAP.py>"
```
3. Create a directory, create an input file inside named `data.gma`
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
test cases were introduced that compare the output of the Python and Fortran
version. They can be found in the `legacy-tests` directory along with a
README file with more information.

Because the Python version will go beyond the capabilities of the Fortran
version, testing by comparing to the results of the Fortran version
will then not be possible anymore. Therefore, tests were
introduced relying on the `unittest` package to validate the Python
version without invoking the Fortran version, which can be found in the
`tests` directory.

