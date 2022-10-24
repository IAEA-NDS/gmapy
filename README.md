## About gmapy

`gmapy` is a Python package to produce nuclear data
evaluations of cross sections including uncertainties
based on the combined data from various experiments.

This package can be regarded as a modernized version
of the [GMAP code] developed by Wolfgang P. Poenitz,
which was employed for the evaluation of the
[neutron data standards][std2017-paper].
A number of [test cases][legacy-tests]
have been created to prove that the latest neutron
standards evaluation can be reproduced with gmapy.
To learn more about improvements of gmapy over GMAP,
please see the respective section below.

[GMAP code]: https://github.com/iaea-nds/GMAP-Fortran
[legacy-tests]: https://github.com/iaea-nds/gmapy/legacy-tests
[std2017-paper]: https://www.sciencedirect.com/science/article/pii/S0090375218300218

### Installation

#### Prerequisites

gmapy depends on the [SuiteSparse] library.
With appropriate build tools, it can probably be installed
on Windows but this has not been tested.
On Linux it can be installed by
```
sudo apt install libsuitesparse-dev
```
and on MacOs via
```
brew install suite-sparse
```

It is also important that the source and library files
of SuiteSparse can be found by the compiler and linker.
If the installation instructions for gmapy in the next
section fail with the error message that `cholmod.h`
cannot be found, execute the following instructions
with appropriate paths:
```
export SUITESPARSE_INCLUDE_DIR=/opt/local/include
export SUITESPARSE_LIBRARY_DIR=/opt/local/lib
```
The given paths are examples of commonly used locations
and may need to be adjusted on your system.

[SuiteSparse]: https://github.com/DrTimothyAldenDavis/SuiteSparse

#### Installation of gmapy

We recommend to create a virtual environment with
Python version 3.9. With conda you can create a new
environment by
```
conda env create -y -n gmapy python=3.9 pip
```
Then activate the environment (`conda activate gmapy`)
and install the gmapy package:
```
pip install git+https://github.com/iaea-nds/gmapy.git
```

### Basic use of gmapy

The capabilities of the gmapy package are continuously evolving
and documentation is still missing to a large extent.
Regarding the GMA database format, you can consult
[this document][gma-format-cheatsheet].

[gma-format-cheatsheet]: https://github.com/IAEA-NDS/gmapy/blob/master/DOCUMENTATION.md


### Beyond GMAP: New features in gmapy

Here are the capabilities of gmapy at present that
go beyond those of the Fortran GMAP code:

- Exact maximum likelihood optimization based on
  a custom version of the Levenberg-Marquardt algorithm
  with convergence checking.
- Adaptive numerical integration by Romberg's method
  with convergence checking for a precise computation
  of spectrum averaged cross sections.
- Indidvidual energy meshes for different reactions
  are possible to account for reaction-specific structures.
- Experimental data need not be aligned perfectly
  to the the computational mesh thanks to the use of
  linear interpolation.
- A new data type `ratios of spectrum averaged cross sections`
  is available.
- Data can be read in the database format used by GMAP
  as well as in a new JSON format.
- Unknown uncertainties can be introduced and estimated
  during the estimation of the cross sections and their
  uncertainties via maximum likelihood estimation.


## Legal note

This code is distributed under the MIT license, see the
accompanying license file for more information.

Copyright (c) International Atomic Energy Agency (IAEA)
