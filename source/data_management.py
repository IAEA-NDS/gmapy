from generic_utils import Bunch
import numpy as np

#
#   Data block / data set   
#
#      NP      NUMBER OF DATA POINTS
#      E       ENERGIES OF EXPERIMENTAL DATA SET
#      CSS     MEASUREMENT VALUES OF EXPERIMENTAL DATA SET
#      DCS     TOTAL UNCERTAINTIES OF EXPERIMENTAL VALUES
#      CO      ENERGY DEPENDENT UNCERTAINTY COMPONENTS
#      ECOR    CORRELATION MATRIX OF EXPERIMENTS IN DATA BLOCK
#      ENFF    NORMALIZATION UNCERTAINTIES COMPONENTS
#      EPAF    UNCERTAINTY COMPONENT PARAMETERS
#      FCFC    CROSS CORRELATION FACTORS
#
#      KAS        indexes of experimental cross sections
#      KA         indexes
#      IDEN       data set info (see below)
#      NENF       tags of norm. uncertainty components
#      NETG       tags of energy dependent uncertainty components
#      NCSST      data set Nr.s for cross correlations
#      NEC        error component pairs for cross correlations
#
#      NT         id of cross sections involved in measured quantity
#
#
#      IDEN(K,I)  K DATA SET SEQUENCE
#
#                 I=1 NO OF DATA POINTS IN SET
#                 I=2 INDEX OF FIRST VALUE IN ECOR
#                 I=3 YEAR
#                 I=4 DATA SET TAG
#                 I=5 NO OF OTHER SETS WITH WHICH CORRELATIONS ARE GIVEN
#                 I=6 DATA SET NR
#                 I=7 MT-TYPE OF DATA
#                 I=8 1 ABSOLUTE, 2 SHAPE

def init_datablock():

    data = Bunch({
        'num_datasets': 0,
        'num_datapoints': 0,

        'NP': 0,
        'E': np.zeros(250+1, dtype=float),
        'CSS': np.zeros(250+1, dtype=float),
        'DCS': np.zeros(250+1, dtype=float),
        'ECOR': np.zeros((250+1, 250+1), dtype=float),
        'CO': np.zeros((12+1, 250+1), dtype=float),
        'ENFF': np.zeros((30+1, 10+1), dtype=float),
        'EPAF': np.zeros((3+1, 11+1, 30+1), dtype=float),
        'FCFC': np.zeros((10+1, 10+1), dtype=float),
        'AAA': np.zeros((250+1, 250+1), dtype=float),

        'KAS': np.zeros((250+1, 5+1), dtype=int),
        'KA': np.zeros((1200+1, 250+1), dtype=int),
        'IDEN': np.zeros((30+1, 8+1), dtype=int),
        'NENF': np.zeros((40+1, 10+1), dtype=int),
        'NETG': np.zeros((11+1,40+1), dtype=int),
        'NCSST': np.zeros(10+1, dtype=int),
        'NEC': np.zeros((2+1,10+1,10+1), dtype=int),

        # TODO: convert following to arrays with index ID
        'NT': np.zeros(5+1, dtype=int)
        })

    return data

