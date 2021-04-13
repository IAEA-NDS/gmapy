#!/bin/sh

# DESCRIPTION
#   compare if the gfortran and Python version
#   yield the same result

basedir=`pwd`
GMAP_fortran_url="https://github.com/IAEA-NDS/GMAP-Fortran.git"
GMAP_fortran_commit_id="6659931b87e4c2d3be8bdd861630346367597690" 
GMAP_python_dir="$basedir/../../source"
GMAP_python_exe="$GMAP_python_dir/GMAP.py"

if [ ! -d "GMAP-Fortran" ]; then
    git clone $GMAP_fortran_url
    retcode=$?
    if [ $retcode -ne "0" ]; then
        print "Downloading Fortran version of GMAP from $GMAP_fortran_url failed"
        exit $retcode
    fi
    if [ ! -d "GMAP-Fortran "]; then
        print "The directory expected to contain the Fortran source code does not exist"
        exit $?
    fi
    cd GMAP-Fortran \
    && git checkout $GMAP_fortran_commit_id \
    && rm -rf .git \
    && cd source \
    && gfortran -o GMAP GMAP.FOR \
    && mv GMAP .. \
    && chmod +x GMAP
    cd "$basedir"
fi

if [ ! -d result/fortran ]; then
    mkdir -p result/fortran
    cp input/data.gma result/fortran/
    cd result/fortran/
    ../../GMAP-Fortran/GMAP
    # Fortran print -0.0 whereas Python 0.0
    # thus remove the odd minus sign from Fortran output
    sed -i -e 's/-\(00*\.0* \)/ \1/g' gma.res
    sed -i -e 's/-\(00*\.0* \)/ \1/g' plot.dta
    cd "$basedir"
fi

# generate Python result
mkdir -p result/python
cp input/data.gma result/python/
cd result/python/
PYTHONPATH="$GMAP_python_dir"
python $GMAP_python_exe

# compare the results
cd "$basedir"
diff result/fortran/gma.res result/python/gma.res \
  && diff result/fortran/plot.dta result/python/plot.dta

retcode=$?
exit $retcode
