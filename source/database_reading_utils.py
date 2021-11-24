from generic_utils import unflatten, Bunch
from fortran_utils import fort_range, fort_read, fort_write
from data_management import init_datablock, init_gauss, SIZE_LIMITS
from gmap_snippets import (should_downweight, get_AX, get_prior_range,
                           get_dataset_range, get_dataset_id_from_idx)

import numpy as np

import linpack_slim
from linpack_utils import (pack_symmetric_matrix, unpack_symmetric_matrix,
                           unpack_utriang_matrix)


# constants for size limits
#      LDA   MAX NO IN DATA BLOCK
#      LDB NO OF UNKNOWNS
LDA = SIZE_LIMITS.MAX_NUM_DATAPOINTS_PER_DATABLOCK
LDB = SIZE_LIMITS.MAX_NUM_UNKNOWNS
LDF = 200



def read_prior(MC1, MC2, APR, file_IO3):
    #
    #      INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
    #
    #      MC1=NE      NO OF ENERGIES/PARAMETERS (total)
    #      MC2=NC      NO OF CROSS SECTION TYPES
    #      NR          NO OF PARAMETERS (cross sections)
    #
    NE = MC1
    NC = MC2
    APR.NC = NC
    NE1 = NE+1
    NR = 0
    for K in fort_range(1,NC):  # .lbl33
        format120 = r"(2A8)"
        APR.CLAB[K, 1:3] = fort_read(file_IO3, format120)

        APR.MCS[K,1] = 0
        for L in fort_range(1,NE1):  # .lbl34
            format103 = "(2E10.4)"
            EX1, CSX1 = fort_read(file_IO3, format103)
            if EX1 == 0:
                break
            APR.MCS[K,1] += 1
            NR += 1
            APR.NR = NR
            APR.EN[NR] = EX1
            APR.CS[NR] = CSX1

    #
    #  MCS(k,l) CONTAINES IN l= 1 TOTAL NUMBER EACH CS
    #                           2 START BOUNDARY
    #                           3 END OF BOUNDARY
    #
    APR.MCS[1, 2] = 1
    APR.MCS[1, 3] = APR.MCS[1,1]
    for K in fort_range(2,NC):
        APR.MCS[K, 2] = APR.MCS[K-1, 2] + APR.MCS[K-1, 1]
        APR.MCS[K, 3] = APR.MCS[K-1, 3] + APR.MCS[K, 1]



def read_datablock(MODC, MOD2, AMO3,
        IELIM, NELIM, LABL, file_IO3):

    format100 = "(A4,1X,8I5)"
    ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8 = fort_read(file_IO3,  format100)
 
    # LABL.AKON[4] == 'BLCK'
    if not ACON == LABL.AKON[4]:
        raise ValueError('Expecting BLCK at beginning of datablock')

    data = prepare_for_datablock_input(MODC, MOD2, AMO3)

    while True:

        ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8 = fort_read(file_IO3,  format100)

        # LABL.AKON[2] == 'DATA'
        if ACON == LABL.AKON[2]:
            deal_with_dataset(MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
                    data, IELIM, NELIM, file_IO3)

        # LABL.AKON[7] == 'EDBL'
        elif ACON == LABL.AKON[7]:
            break

        else:
            raise ValueError('Inadmissible keyword ' + str(ACON) + ' while reading datablock')

    #
    #    Data BLOCK complete
    #

    return data



def prepare_for_datablock_input(MODC, MOD2, AMO3):
    #
    #      BLOCK INPUT
    #
    #      N     TOTAL NO OF DATA POINTS IN BLOCK
    #      ID    TOTAL NO OF DATA SETS IN BLOCK
    #

    data = init_datablock()
    data.MODC = MODC
    data.MOD2 = MOD2
    data.AMO3 = AMO3

    return data



def deal_with_dataset(MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
        data, IELIM, NELIM, file_IO3):

    read_dataset_input(
            MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
            data,
            file_IO3
    )

    exclflag = should_exclude_dataset(data, IELIM, NELIM)

    return



def read_dataset_input(MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8,
        data,
        file_IO3):
    #
    #      DATA SET INPUT
    #
    #      MC1 NS      DATA SET NO
    #      MC2 MT      TYPE OF MEASUREMENT
    #      MC3 NCOX    CORRELATION MATRIX GIVEN IF .NE. 0
    #      MC4 NCT     NO OF CROSS SECTIONS INVOLVED
    #      MC5 NT(1)   CROSS SECTION IDENTIFICATION
    #      MC6 NT(2)   SAME
    #      MC7 NT(3)   SAME
    #      MC8 NNCOX   DIVIDE UNCERTAINTIES BY 10.
    #
    # label .lbl2
    ID = data.num_datasets
    NS = MC1
    MT = MC2
    NCOX = MC3
    NCT = MC4
    NT = np.zeros(data.NT.shape[1], dtype=int)
    NT[1] = MC5
    NT[2] = MC6
    NT[3] = MC7
    NNCOX = MC8
    NENF = data.NENF
    NETG = data.NETG
    NCSST = data.NCSST
    NEC = data.NEC

    format123 = "(16I5)"
    if NCT > 3:
        NT[4:(NCT+1)] = fort_read(file_IO3, format123)

    ID = ID+1
    data.NT[ID,:] = NT
    data.NCT[ID] = NCT
    data.NCOX[ID] = NCOX
    data.NNCOX[ID] = NNCOX
    IDEN = data.IDEN

    IDEN[ID,6] = NS
    IDEN[ID,7] = MT
    #
    #       identify absolute or shape data
    #
    MTTP = 1
    IDEN[ID, 8] = 1
    if (MT == 2 or MT == 4 or MT == 8 or MT == 9):
        MTTP = 2
        IDEN[ID, 8] = 2
    data.MTTP[ID] = MTTP

    # VP      if(modrep .ne. 0) go to 140
    format131 = "(3I5,4A8,4A8)"
    IDEN[ID,3:6], data.CLABL[ID, 1:5], data.BREF[ID, 1:5] = \
            unflatten(fort_read(file_IO3, format131), [[3],[4],[4]])

    # label .lbl183
    NCCS = IDEN[ID, 5]
    #
    #       READ(3,    ) NORMALIZATION UNCERTAINTIES
    #
    if MTTP != 2:
        format201 = "(10F5.1,10I3)"
        data.ENFF[ID, 1:11], NENF[ID, 1:11] = unflatten(fort_read(file_IO3, format201), [[10],[10]])

    #
    #       READ(3,    ) ENERGY DEPENDENT UNCERTAINTY PARAMETERS
    #
    format202 = "(3F5.2,I3)"
    for K in fort_range(1,11):
        data.EPAF[1:4, K, ID], NETG[K, ID] = unflatten(fort_read(file_IO3, format202), [[3], 1])
    #
    #       READ(3,    ) CORRELATIONS INFORMATION
    #
    if NCCS != 0:
        format841 = "(10F5.1)"
        format205 = "(I5,20I3)"
        for K in fort_range(1,NCCS):  # .lbl204
            NCSST[ID, K], tmp = unflatten(fort_read(file_IO3, format205), [1,[20]])
            NEC[ID, 1:3, 1:11, K] = np.reshape(tmp, (2,10), order='F')
            #NCSST[K], NEC[0:2, 0:10, K] = unflatten(fort_read(file_IO3, format205), [1,[20]])
            data.FCFC[ID, 1:11, K] = fort_read(file_IO3, format841)

    # read the energies, cross sections and uncertainty components
    NADD = data.num_datapoints + 1
    NALT = NADD
    for KS in fort_range(1,LDA):
        format109 = "(2E10.4,12F5.1)"
        data.E[NADD], data.CSS[NADD], data.CO[1:13, NADD] = \
            unflatten(fort_read(file_IO3, format109), [2,[12]])
        if data.E[NADD] == 0:
            break
        NADD += 1

    if data.E[NADD] != 0:
        print('ERROR: too many datapoints regarding current LDA setting')
        exit()

    data.num_datapoints = NADD - 1
    IDEN[ID, 1] = NADD - NALT
    data.num_datasets = ID

    # read correlation matrix if given
    if NCOX != 0:
        #
        #      INPUT OF ECOR
        #
        format161 = "(10F8.5)"
        for KS in fort_range(1,NCOX):  # .lbl61
            num_el_read = 0
            num_el_desired = KS
            res = []
            while num_el_read < num_el_desired:
                tmp = fort_read(file_IO3, format161)
                tmp = [x for x in tmp if x is not None]
                res += tmp
                num_el_read += len(tmp)

            data.userECOR[KS,1:(KS+1)] = res
            data.userECOR[1:(KS+1),KS] = data.userECOR[KS,1:(KS+1)]

    #  uncertainty transformations
    XNORU = 0.
    if data.MTTP[ID] != 2:
        #
        #       CALCULATE TOTAL NORMALIZATION UNCERTAINTY
        #
        for L in fort_range(1,10):  # .lbl208
            XNORU = XNORU + (data.ENFF[ID,L])**2

    NADD_MAX = data.num_datapoints
    NADD_MIN = data.num_datapoints - IDEN[ID,1]  + 1
    for NADD in fort_range(NADD_MIN, NADD_MAX):
        #
        #      this is the Axton special (uncertainties have been multiplied by 10
        #         in order to preserve precision beyond 0.1%)
        #
        if NNCOX != 0:
            for LZ in fort_range(1,11):  # .lbl57
                data.CO[LZ, NADD] = data.CO[LZ, NADD] / 10.

        #
        #      test option:  as set with mode control
        #
        #      changing weights of data based on year or data set tag
        #
        if should_downweight(ID, data):
                for I in fort_range(3,11):
                    data.CO[I, NADD] = data.AMO3*data.CO[I, NADD]
        #
        #      CALCULATE TOTAL UNCERTAINTY  DCS
        #
        RELU = 0.
        for L in fort_range(3,11):  # .lbl207
            RELU += data.CO[L, NADD]**2

        data.DCS[NADD] = np.sqrt(XNORU + RELU)
        data.effDCS[NADD] = np.sqrt(XNORU + RELU)

    return



def should_exclude_dataset(data, IELIM, NELIM):

    should_exclude = False
    ID = data.num_datasets
    IDEN = data.IDEN
    NS = IDEN[ID, 6]
    MTTP = IDEN[ID, 8]
    NADD = data.num_datapoints + 1
    NALT = NADD - IDEN[ID, 1]

    NP = NADD - NALT
    if IELIM > 0:
        #      data set excluded ?
        if NS in NELIM[1:(IELIM+1)]:
            data.excluded_datasets.add(NS)
            should_exclude = True
    #
    #      NO VALID DATA POINTS OR SET REQUESTED TO BE EXCLUDED
    #
    if NP == 1 and MTTP == 2:
        data.excluded_datasets.add(NS)
        should_exclude = True

    if NP == 0:
        data.excluded_datasets.add(NS)
        should_exclude = True

    if should_exclude:
        ID = ID - 1
        NADD = NALT

    data.num_datasets = ID
    data.num_datapoints = NADD - 1
    return (should_exclude)



