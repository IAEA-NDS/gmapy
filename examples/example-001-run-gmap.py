############################################################
#
# Author:       Georg Schnabel
# Email:        g.schnabel@iaea.org
# Date:         2022/03/24
# Institution:  IAEA
#
# This example script allows to launch the GMAP
# program from the command line. It can read the
# experimental data either from a JSON file or
# a file of the legacy structure used by the
# Fortran GMAP code. This script is a simple
# command line interface to the gmapy package.
#
# Usage:
#     python example-001-run-gmap.py --dbfile <DBFILE>
#
#     <DBFILE>: GMAP Database file in legacy format
#
#     Instead of --dbfile you can use the --jsondb
#     argument to load data from a JSON file.
############################################################

import argparse
from gmapy.legacy.legacy_gmap import run_gmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform generalized least squares analysis')
    parser.add_argument('--dbfile', help='name of the GMA database file', required=False, default='data.gma')
    parser.add_argument('--jsondb', help='name of the json database file', required=False, default='')
    args = parser.parse_args()
    dbtype = 'json' if args.jsondb != '' else 'legacy'
    dbfile = args.jsondb if dbtype == 'json' else args.dbfile
    run_gmap(dbfile=dbfile, dbtype=dbtype)