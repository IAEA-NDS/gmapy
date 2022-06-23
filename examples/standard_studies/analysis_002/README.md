### Comparison of different evaluation scenarios 

Each Python file starting with `gmap_fit_*`
reads a specific version of the GMA database
with experimental data, potentially modifies
data on the fly and performs a Generalized
Least Squares (GLS) fit.

The R script `std_data_analysis.R` calls the
Python scripts and produces plots of the data
and the various fits.

*Note:* Ensure that the scripts are executed
with this directory as the working directory.
If you run `std_data_analysis.R`, consider
that you have to adapt the line `use_condaenv("reticulate")`.
The `reticulate` environment is supposed to contain
the required packages to run the Python scripts
in this directory. If all required packages are
already available in your system-wide Python installation,
you can remove this line.
