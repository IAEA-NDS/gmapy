from generic_utils import Bunch
import numpy as np
from collections import OrderedDict


SIZE_LIMITS = Bunch({
    'MAX_NUM_DATAPOINTS_PER_DATABLOCK': 250,
    'MAX_NUM_DATASETS_PER_DATABLOCK': 30,
    'MAX_NUM_CORRELATED_DATASETS_PER_DATASET': 10,
    'MAX_NUM_UNKNOWNS': 1200
})


def init_labels():

    #
    #  CONTROLS/LABELS
    #
    LABL = Bunch({
        'AKON':  [None for i in range(12)],
        'TYPE': np.zeros(10+1, dtype=object)
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
#      NSHP  number of shape data sets
#      NSETN shape data set numbers

def init_prior():

    APR = Bunch({
        'CLAB': np.empty((35+1,2+1), dtype=object),
        'EN': np.zeros(1200+1, dtype=float),
        'CS': np.zeros(1200+1, dtype=float),
        'MCS': np.zeros((35+1,3+1), dtype=int),
        'NR': 0,
        'NC': 0,
        'NSHP': 0,
        'NSETN': np.zeros(200+1, dtype=int)
        })
    return APR


#
#   Data block / data set   
#
#      NP      NUMBER OF DATA POINTS
#      E       ENERGIES OF EXPERIMENTAL DATA SET
#      CSS     MEASUREMENT VALUES OF EXPERIMENTAL DATA SET
#      DCS     TOTAL UNCERTAINTIES OF EXPERIMENTAL VALUES
#      effDCS  Total uncertainties of experimental values as used in fit, i.e., after PPP correction
#      CO      ENERGY DEPENDENT UNCERTAINTY COMPONENTS
#      userECOR Correlation matrix for datablock as provided by user
#      ECOR    Constructed correlation matrix for datablock based on uncertainty components
#      effECOR the ECOR as used in the fit, i.e., after a possible adjustment
#                                                 to ensure it being positive-definite
#      invECOR inverse CORRELATION MATRIX OF EXPERIMENTS IN DATA BLOCK
#      num_inv_tries  number of attempts to invert ECOR by modifying it
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
#      NT         id of cross sections involved in measured quantity for each dataset
#      NCT        number of cross section types involved for each dataset
#
#      NCOX       array of flags: if element not zero, correlation matrix is given for respective dataset
#      NNCOX      array of flags: if element not zero, divide uncertainties by 10 for respective dataset
#      MTTP       array of shape-flag; element is 2 if dataset contains shape data else 1
#
#      AA         COEFFICIENT MATRIX
#      AM         MEASUREMENT VECTOR
#
#      MODC       mode of uncertainty specification: 1 - Ecor matrix provided
#                                                    2 - uncorrelated uncertainties
#                                                    3 - all correleated errors given
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
#
#
#      num_datasets          number of datasets in datablock
#      num_datapoints        number of datapoints in datablocks over all datasets
#      num_datapoints_used   number of datapoints really used in Bayesian inference
#
#      excluded_datasets     identification numbers of excluded datasets
#      missing_datasets      id numbers of cross-correlated datasets that are missing
#      invalid_datapoints    dictionary with keys being the dataset number and the
#                            associated values lists of datapoints that could not be
#                            mapped to the prior mesh
#
#      Following variables are added to help in the separation of the
#      printing from the calculations in fill_AA_AM_COV
#
#      EAVR       TODO
#      SFIS       TODO
#      FL         TODO
#
#      Following variables is calculated in get_matrix_products
#      and saved in the datablock structure for later printing
#
#      SIGL       TODO
#      NTOT       number of datapoints to be used in the fit up to
#                 and including this datablock
#
#      There is a problematic variable L in the Fortran code which depending
#      the place of execution contains a different quantity. For reproducibility,
#      we store the value of L in the data structure for later printing
#
#      problematic_L             a dictionary with keys given by dataset ID and L as the value
#      problematic_L_Ecor        a dictionary with keys given by dataset ID and L as the value
#      problematic_L_dimexcess   a dictionary with keys given by dataset ID and L as the value
#

def init_datablock():

    MAXDP = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
    MAXDS = SIZE_LIMITS.MAX_NUM_DATASETS_PER_DATABLOCK
    MAXCOR = SIZE_LIMITS.MAX_NUM_CORRELATED_DATASETS_PER_DATASET

    data = Bunch({

        'CLABL': np.empty((MAXDS, 4+1), dtype=object),
        'BREF': np.zeros((MAXDS, 4+1), dtype=object),

        'num_datasets': 0,
        'num_datapoints': 0,
        'num_datapoints_used': 0,

        'excluded_datasets': set(),
        'missing_datasets': OrderedDict(),
        'invalid_datapoints': OrderedDict(),

        'E': np.zeros(MAXDP+1, dtype=float),
        'CSS': np.zeros(MAXDP+1, dtype=float),
        'DCS': np.zeros(MAXDP+1, dtype=float),
        'effDCS': np.zeros(MAXDP+1, dtype=float),
        'userECOR': np.zeros((MAXDP+1, MAXDP+1), dtype=float),
        'ECOR': np.zeros((MAXDP+1, MAXDP+1), dtype=float),
        'effECOR': np.zeros((MAXDP+1, MAXDP+1), dtype=float),
        'invECOR': np.zeros((MAXDP+1, MAXDP+1), dtype=float),
        'num_inv_tries': 0,
        'CO': np.zeros((12+1, MAXDP+1), dtype=float),
        'ENFF': np.zeros((MAXDS+1, 10+1), dtype=float),
        'EPAF': np.zeros((3+1, 11+1, MAXDS+1), dtype=float),
        'FCFC': np.zeros((MAXDS+1, 10+1, MAXCOR+1), dtype=float),
        'AAA': np.zeros((MAXDP+1, MAXDP+1), dtype=float),

        'KAS': np.zeros((MAXDP+1, 5+1), dtype=int),
        'KA': np.zeros((1200+1, MAXDP+1), dtype=int),
        'IDEN': np.zeros((MAXDS+1, 8+1), dtype=int),
        'NENF': np.zeros((MAXDS+1, 10+1), dtype=int),
        'NETG': np.zeros((11+1,MAXDS+1), dtype=int),
        'NCSST': np.zeros((MAXDS+1, MAXCOR+1), dtype=int),
        'NEC': np.zeros((MAXDS+1, 2+1,10+1,MAXCOR+1), dtype=int),

        'NT': np.zeros((MAXDS+1, 5+1), dtype=int),
        'NCT': np.zeros(MAXDS+1, dtype=int),

        'NCOX': np.zeros(MAXDS+1, dtype=int),
        'NNCOX': np.zeros(MAXDS+1, dtype=int),
        'MTTP': np.zeros(MAXDS+1, dtype=int),

        'AA': np.zeros((1200+1, MAXDP+1), dtype=float),
        'AM': np.zeros(MAXDP+1, dtype=float),

        'EAVR': np.zeros(MAXDP+1, dtype=float),
        'SFIS': np.zeros(MAXDP+1, dtype=float),
        'FL': np.zeros(MAXDP+1, dtype=float),

        'SIGL': 0.,
        'NTOT': 0,

        'MODC': 0,
        'MOD2': 0,
        'AMO3': 0.,

        'problematic_L': {},
        'problematic_L_Ecor': {},
        'problematic_L_dimexcess': {}
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
#       NTOT    number of datapoints used in fitting
#       SIGMA2  TODO
#

def init_gauss():

    MAXDP = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
    MAXDS = SIZE_LIMITS.MAX_NUM_DATASETS_PER_DATABLOCK
    MAXCOR = SIZE_LIMITS.MAX_NUM_CORRELATED_DATASETS_PER_DATASET

    gauss = Bunch({
            'AA': np.zeros((1200+1, MAXDP+1), dtype=float),
            'AM': np.zeros(MAXDP+1, dtype=float),
            'DE': np.zeros(1200+1, dtype=float),
            'BM': np.zeros(1200+1, dtype=float),
            'B': np.zeros(720600+1, dtype=float),
            'EGR': np.zeros(199+1, dtype=float),
            'EEGR': np.zeros(400+1, dtype=float),
            'RELTRG': np.zeros((200+1, 200+1), dtype=float),
            'RELCOV': np.zeros((200+1, 200+1), dtype=float),
            'NTOT': 0,
            'SIGMA2': 0.
    })

    return gauss

