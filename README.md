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

This package is still evolving and not well documented yet.
To get a first impression of the usage of gmapy, you can
inspect the Jupyter notebook `examples/example-000-basic-usage.ipynb`.
You can directly open this notebook in the browser by clicking
[here][mybinder-example-000].

[mybinder-example-000]: https://mybinder.org/v2/gh/iaea-nds/gmapy/dev?labpath=examples%2Fexample-000-basic-usage.ipynb
[GMAP code]: https://github.com/iaea-nds/GMAP-Fortran
[legacy-tests]: https://github.com/iaea-nds/gmapy/legacy-tests
[std2017-paper]: https://www.sciencedirect.com/science/article/pii/S0090375218300218

### Installation

gmapy depends on the [SuiteSparse] library.
Please consults the [scikit-sparse] GitHub repo for installation instructions.
If you are using the conda package manager, you can install `gmapy` in this way:

Download or clone this repository on your computer. Let's assume the repo is
stored in the directory `gmapy`. Then you can execute:

```
conda create -n gmapy-env python=3.10
conda activate gmapy-env
conda install -c conda-forge scikit-sparse==0.4.14
pip install ./gmapy
```

[SuiteSparse]: https://github.com/DrTimothyAldenDavis/SuiteSparse
[scikit-sparse]: https://github.com/scikit-sparse/scikit-sparse


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
