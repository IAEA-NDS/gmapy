# Documentation

Here useful snippets of information are provided
regarding the use of the GMAP program. This document
is work in progress and will be converted into another
form at some point once it includes enough information.

## Dataset specification

A dataset specification is given as a dictionary.
The fields in the dictionary can be divided into
three groups:

1. Experimental meta-information
2. Measured quantity
3. Uncertainty quantification

Regarding (1) experimental meta-information, we have the following fields:

- **type** (string): must be "legacy-experiment-dataset" at present
- **NS** (integer): a unique 4-digit identfication number of the dataset
- **YEAR** (integer): the year when the measurement has been performed 
- **TAG** (integer): an integer to indicate some classification (not used by GMAP) 
- **CLABL** (string): names of the authors
- **BREF** (string): reference to the publication 

Regarding (2) measured quantity, we have the fields:

- **MT** (integer): An integer indicating the measured quantity,
                    see the [section](#measurement-types) on measurement types.
- **NT** (list of integer): a list with the identification numbers of the
                    corresponding cross sections in the prior. For instance,
                    for a ratio of cross sections (MT=2), two ids are expected.
- **E** (list of floats): the incident energies of the measurements
- **CSS** (list of floats): the measured quantities, e.g., cross sections

The group of fields regarding (3) uncertainty quantification can be split
up into specifications that make reference to the points *(a) within the dataset*
and *(b) to other datasets*.  

For (3a), uncertainty quantification within the dataset, the following fields
are relevant: **EPAF**, **CO**, **ENFF**, **NNCOX**. The number of datapoints
in the dataset is denoted by *N*.

- **CO** (N x 11 array with floats): each datapoint is associated with 11 available 
                                     slots to store relative uncertainty components in
                                     percent, e.g., a value of *2.5* in a slot
                                     would mean 2.5% uncertainty. Importantly,
                                     the *first two slots are ignored*, i.e., not
                                     used to calculate uncertainties of or
                                     correlations between datapoints.
- **ENFF** (list with 10 floats): 10 slots to store relative uncertainty components
                                  that apply to all datapoints in the dataset.
                                  For instance, a value of *2.5* in a slot would
                                  mean 2.5% uncertainty. Importantly, even though the
                                  same uncertainty is added to each datapoint,
                                  no correlations are introduced between datapoints.
                                  In other words, the underlying errors are assumed
                                  to have the same uncertainty but are assumed
                                  to be independent from each other. The uncertainties
                                  in **ENFF** are ignored for *shape measurements*.
- **NENFF** (list with 10 integers): This field stores a list with integers that
                                  can serve as tags to label the uncertainty
                                  components. Apart from being read into a data
                                  structure, *it is not used by GMAP*.
- **NNCOX** (integer): If not zero, all uncertainties in **CO** will be multiplied
                       by a factor of *1/10*, hence largely decreasing the uncertainties
                       effectively used in the GMAP fit.
- **EPAF** (3 x 11 array with floats): Used to determine the energy-dependent 
                                       covariance of distinct measurements in the dataset.
                                       Each column contains three numbers for each of the
                                       11 slots in **CO**. These numbers are the values
                                       of parameters in a formula that determines the decline of the
                                       covariance as a function of energy difference for each slot.
                                       The formula is explained below.
- **NETG** (list with 11 integers): If a value is 0 or 9, the uncertainty in the respective
                                    component in **CO** is ignored, i.e., both uncertainties
                                    and correlations.

The values in the arrays **CO** and **EPAF** are used to determine the
covariance between pairs of datapoints for each uncertainty component in **CO**.
For the uncertainty component in slot *i* and and two measurements with index 
*j* and *k* in the dataset and associated incident energies *E[j]* and *E[k]*
with *E[j] < E[k]*, the covariance is calculated as follows:
```
FKS = EPAF[0,i] + EPAF[1,i]
XYY = EPAF[1,i] - (E[k] - E[j]) / (EPAF[2,i]*E[j])
XYY_cut = max(XYY, 0)
FKT = EPAF[0, i] + XYY
COV[i,j,k] = CO[j,i]*CO[k,i]*FKS*FKT
```
The result `COV[i,j,k]` represents the relative covariance between the j-th
and k-th datapoint for the i-th uncertainty component.
The total relative covariance for datapoints is obtained by summing over all
partial covariances above, i.e, `COV[j,k] = sum_i COV[i,j,k]`.
The total relative covariance matrix is afterwards converted to a correlation
matrix using the uncertainties as specified by **CO** and **ENFF**.
The details can be found in the code
[here](https://github.com/IAEA-NDS/GMAP-Python/blob/dev/gmapi/data_management/uncfuns.py#L57)
and
[here](https://github.com/IAEA-NDS/GMAP-Python/blob/1a96fbf45cb4713ebd2270c60e302388a757a9a0/gmapi/data_management/uncfuns.py#L9).

Finally, regarding (3b) correlations between datasets, the fields presented below
are relevant (*M* is the number of datasets correlated to the current one.

- **NCSST** (a list of size M with integers): A list with the dataset identification numbers
                                    (**NS** above) that are correlated with this
                                    dataset.
- **NEC** (2 x 10 x M array of integers): Determines which uncertainty component
          of this dataset is correlated with which component of the other datasets.
          There are ten slots to establish mappings between the uncertainty
          components of the datasets. For instance, `NEC[0,0,4] = 7` 
          and `NEC[1,0,4] = 2` signals that the uncertainty component in slot 7
          in **ENFF** of the current dataset is correlated with the uncertainty component
          in slot 2 in **ENFF** of the other dataset with identification number given in
          `NCSST[4]`. If a specification like this would have been already made,
          hence slot 0 already occupied, we would have to move the
          specification in the example to slot 1, i.e., `NEC[0,1,4] = 7` and
          `NEC[1,1,4] = 2`. An integer larger 10 indicates that the covariances
          between the two components involved are energy dependent. For instance,
          `NEC[0,0,4] = 17` and `NEC[1,0,4] = 12` would indicate energy-dependent
          covariances between the uncertainty component in slot 7 in **CO** of the
          current dataset and the uncertainty component in slot 2 in **CO** of the
          other dataset with identification number `NCSST[4]`. The formulas
          are provided below.
- **FCFC** (10 x M array of floats): A mutliplication factor to damp the
          covariances between datasets. For instance, the specification
          `FCFC[5,4]=0.5` indicates that the covariance between the current
          dataset and the dataset with identification number `NCSST[4]`
          for the uncertainty component `NEC[0,5,4]` of the current dataset
          and the component `NEC[1,5,4]` of the other dataset should be
          halved.

Now we elaborate on the formulas to determine the covariance between
the uncertainty components of two distinct datasets in the same datablock.
As an example, let us assume that we deal with the following specification
`NEC[0,1,4]=7` and `NEC[1,1,4]=3`, which indicates correlation between
the uncertainty component **ENFF[7]** of the current dataset and the
uncertainty component **ENFF[3]** of the other dataset with id `NCSST[4]`.
The formula to calculate the covariance between these two uncertainty
components is given by:
```
COV = FCFC[1,4] * ENFF[7] * ENFF[3]
```

For the other case, where `NEC[.,.,.] > 10` please consult the formulas
in the Python [source code](https://github.com/IAEA-NDS/GMAP-Python/blob/1a96fbf45cb4713ebd2270c60e302388a757a9a0/gmapi/data_management/uncfuns.py#L246).


## Measurement types

The **MT** number (integer) defines the measurement quantity.
Sometimes we abbreviate cross section by *xs* in the following list.
Possible choices of the MT number are:

- **MT 1**: cross section
- **MT 2**: shape of cross section
- **MT 3**: ratio of cross sections
- **MT 4**: shape of ratio of cross sections
- **MT 5**: sum of cross sections
- **MT 6**: spectrum averaged cross section
- **MT 7**: xs1 / (xs2 + xs3) 
- **MT 8**: shape of sum of cross sections
- **MT 9**: shape of xs1 / (xs2+xs3)

The addition *shape of ...* means that the normalization factor
of the measured quantity is assumed to be unknown and needs
to be estimated during fitting. Hence, there is the following
correspondence between normalized and unnormalized quantities:

- **MT 2** - **MT 1**
- **MT 4** - **MT 3**
- **MT 8** - **MT 5**
- **MT 9** - **MT 7**

