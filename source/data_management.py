from generic_utils import Bunch
import numpy as np

def init_labels():

    #
    #  CONTROLS/LABELS
    #
    LABL = Bunch({
        'AKON':  [None for i in range(12)],
        'CLAB': np.empty((35+1,2+1), dtype=object),
        'CLABL': np.empty(4+1, dtype=object),
        'TYPE': np.zeros(10+1, dtype=object),
        'BREF': np.zeros(4+1, dtype=object)

        })
    #
    #      CONTROL
    #      MODE    MODE OF EXP. CORRELATION MATRIX CONSTRUCTION
    #      I/OC    I/O CONTROL
    #      APRI    INPUT OF APRIORI CROSS SECTIONS
    #      FIS*    FISSION FLUX FACTORS
    #      BLCK    START OF DATA BLOCK
    #      DATA    INPUT OF DATA
    #      END*    END OF EXP  DATA, START OF EVALUATION
    #      EDBL    END OF EXP. DATA SET BLOCK
    #      ELIM    DATA SETS TO BE ELIMINATED
    #

    LABL.AKON[1] = 'APRI'
    LABL.AKON[2] = 'DATA'
    LABL.AKON[3] = 'END*'
    LABL.AKON[4] = 'BLCK'
    LABL.AKON[5] = 'I/OC'
    LABL.AKON[7] = 'EDBL'
    LABL.AKON[8] = 'FIS*'
    LABL.AKON[9] = 'MODE'
    LABL.AKON[10] = 'STOP'
    LABL.AKON[11] = 'ELIM'
    #
    #      MEASUREMENT TYPE IDENTIFICATION
    #
    LABL.TYPE[1] = 'CROSS SECTION   '
    LABL.TYPE[2] = 'CS-SHAPE        '
    LABL.TYPE[3] = 'RATIO           '
    LABL.TYPE[4] = 'RATIO SHAPE     '
    LABL.TYPE[5] = 'TOTAL  CS       '
    LABL.TYPE[6] = 'FISSION AVERAGE '
    LABL.TYPE[7] = 'ABS. S1/(S2+S3) '
    LABL.TYPE[8] = 'SHAPE OF SUM    '
    LABL.TYPE[9] = 'SHP. S1/(S2+S3) '

    return LABL


#   Parameters/apriori
#
#      EN    ENERGY GRID
#      CS    APRIORI CROSS SECTIONS
#      MCS   INDEXES
#      NR    number of parameters (cross sections)
#      NC    number of cross section types

def init_prior():

    APR = Bunch({
        'EN': np.zeros(1200+1, dtype=float),
        'CS': np.zeros(1200+1, dtype=float),
        'MCS': np.zeros((35+1,3+1), dtype=int),
        'NR': 0,
        'NC': 0
        })
    return APR


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
#      MTTP       array of shape-flag; element is 2 if dataset contains shape data else 1
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
        'NT': np.zeros(5+1, dtype=int),

        'MTTP': np.zeros(30+1, dtype=int)
        })

    return data


#
#       AA      COEFFICIENT MATRIX
#       AM      MEASUREMENT VECTOR
# VP    ECOR    inverse of correlation matrix of measurement vector AM
# VP            or relative weight matrix
# VP    BM      vector accumulating (in data block cycle) sum of 
# VP            AA(transpose)*ECOR*AM
# VP    B       matrix accumulating (in data block cycle) sum of 
# VP            AA(transpose)*ECOR*AA
#       DE      ADJUSTMENT VECTOR
# VP            equals B(inverse)*BM
#

def init_gauss():

    gauss = Bunch({
            'AA': np.zeros((1200+1, 250+1), dtype=float),
            'AM': np.zeros(250+1, dtype=float),
            'DE': np.zeros(1200+1, dtype=float),
            'BM': np.zeros(1200+1, dtype=float),
            'B': np.zeros(720600+1, dtype=float),
            'EGR': np.zeros(199+1, dtype=float),
            'EEGR': np.zeros(400+1, dtype=float),
            'RELTRG': np.zeros((200+1, 200+1), dtype=float),
            'RELCOV': np.zeros((200+1, 200+1), dtype=float)
    })

    return gauss

