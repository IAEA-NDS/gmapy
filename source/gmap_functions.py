from generic_utils import unflatten, Bunch
from fortran_utils import fort_range, fort_read, fort_write
from data_management import init_datablock, SIZE_LIMITS
from gmap_snippets import (should_downweight, get_AX, get_prior_range,
                           get_dataset_range, get_dataset_id_from_idx)
from output_management import (write_dataset_info, write_prior_info,
                               write_datablock_header, write_KAS_check,
                               write_overflow_message, write_dataset_exclusion_info,
                               write_missing_dataset_info, write_invalid_datapoints_info,
                               write_dataset_table, write_fission_average)

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

#
#      test option:  forced stop for testing purpose
#
def force_stop(file_IO4):
    format107 = "( '  REQUESTED STOP ' )"
    fort_write(file_IO4, format107)
    exit()


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



def accounting(ID, data, APR):
    #
    #      ACCOUNTING
    #
    #      N,NADD      NO OF TOTAL DATA POINTS SO FAR IN BLOCK
    #      ID          NO OF EXPERIMENTAL DATA SETS
    #      NP          NO OF DATA POINTS IN THIS SET
    #
    IDEN = data.IDEN
    NS = IDEN[ID, 6]
    MT = IDEN[ID, 7]
    KAS = data.KAS
    NT = data.NT[ID,:]
    NCT = data.NCT[ID]

    NADD_MIN, NADD_MAX = get_dataset_range(ID, data)
    for NADD in fort_range(NADD_MIN, NADD_MAX):  # .lbl21

        #
        #      SORT EXP ENERGIES  TO FIND CORRESPONDING INDEX OF EVALUATION EN
        #
        #      KAS(I,L)   GIVES INDEX OF EVALUATION ENERGY FOR I.TH EXP POINT
        #                 AND L.TH CROSS SECTION
        #
        if MT != 6:
            #
            #      NCT is the number of cross sections involved
            #
            for L in fort_range(1,NCT):  # .lbl48
                JE = APR.MCS[NT[L], 2]
                JI = APR.MCS[NT[L], 3]

                found = False
                for K in fort_range(JE, JI):  # .lbl12
                    E1 = .999*APR.EN[K]
                    E2 = 1.001*APR.EN[K]
                    if data.E[NADD] > E1 and data.E[NADD] < E2:
                        found = True
                        break

                if not found:
                    print('ERROR: experimental energy does not match energy mesh')
                    exit()

                KAS[NADD, L] = K
                #
                #      Exception for dummy data sets
                #
                if NS >= 900 and NS <= 909:
                    data.CSS[NADD] = APR.CS[K]
                # .lbl48

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



def construct_Ecor(ID, data):
    #
    #      CONSTRUCT ECOR
    #
    #         MODE  1   INPUT OF ECOR
    #               2   UNCORRELATED
    #               3   ALL CORRELATED ERRORS GIVEN
    #
    IDEN = data.IDEN
    NETG = data.NETG
    NCCS = IDEN[ID, 5]
    MTTP = IDEN[ID, 8]
    NS = IDEN[ID, 6]
    NALT, NADD1 = get_dataset_range(ID, data)
    NP = IDEN[ID, 1]
    NCSST = data.NCSST
    NEC = data.NEC
    NCOX = data.NCOX[ID]
    MODC = data.MODC

    data.missing_datasets[NS] = []

    if NCOX != 0:
        MODAL = MODC
        MODC = 1

    if MODC == 1:
        # related to INPUT OF ECOR
        L = NCOX+1   # to match the value of L after READ (label 61 in Fortran code)
        MODC = MODAL


    elif MODC == 2:
        #
        #       UNCORRELATED OR SINGLE VALUE
        #
        L = 13  # this values is due to line 252
                # caused by read statement starting with
                # E[NADD], ... in accounting
        for KLK in fort_range(NALT, NADD1):  # .lbl74
            data.ECOR[KLK,KLK] = 1.


    elif MODC >= 3 and MODC <= 6:
        #
        #       CONSTRUCT ECOR FROM UNCERTAINTY COMPONENTS
        #
        data.ECOR[NALT, NALT] = 1.
        if NP == 1:
            L = 13  # this value is due to line 252
                    # caused by read statement starting with
                    # N[NADD], ... in accounting
        else:
            NALT1 = NALT + 1
            for KS in fort_range(NALT1, NADD1):  # .lbl62
                C1 = data.DCS[KS]
                KS1 = KS - 1

                for KT in fort_range(NALT, KS1):  # .lbl162
                    Q1 = 0.
                    C2 = data.DCS[KT]
                    for L in fort_range(3,11):  # .lbl215
                        if NETG[L, ID] == 9:
                            continue
                        if NETG[L, ID] == 0:
                            continue

                        FKS = data.EPAF[1,L,ID] + data.EPAF[2,L,ID]
                        XYY = data.EPAF[2,L,ID] - (data.E[KS]-data.E[KT])/(data.EPAF[3,L,ID]*data.E[KS])
                        if XYY < 0.:
                            XYY = 0.
                        FKT = data.EPAF[1, L, ID] + XYY
                        Q1=Q1+data.CO[L,KS]*data.CO[L,KT]*FKS*FKT

                    L = L + 1  # to match L value of fortran after loop

                    #
                    #       CALCULATE TOTAL NORMALIZATION UNCERTAINTY
                    #
                    XNORU = 0.
                    if data.MTTP[ID] != 2:
                        for K in fort_range(1,10):  # .lbl208
                            XNORU = XNORU + (data.ENFF[ID,K])**2

                    CERR = (Q1 + XNORU) / (C1*C2)

                    if CERR > .99:
                        CERR = .99
                    # limit accuracy of comparison to reflect
                    # Fortran behavior
                    data.ECOR[KS,KT] = CERR
                    data.ECOR[KT,KS] = data.ECOR[KS,KT]

                data.ECOR[KS, KS] = 1.

        #
        #   ADD CROSS CORRELATIONS OF EXPERIMENTAL DATA BLOCK
        #
        if ID != 1 and NCCS != 0:

            ID1 = ID - 1
            for I in fort_range(1,NCCS):  # .lbl271

                NSET = NCSST[ID, I]
                found = False
                for II in fort_range(1,ID1):  # .lbl272
                    if IDEN[II,6] == NSET:
                        found = True
                        break

                if not found:
                    #
                    #   CORRELATED DATA SET NOT FOUND AHEAD OF PRESENT DATA
                    #   SET WITHIN DATA BLOCK
                    #
                    data.missing_datasets[NS].append(NSET)

                else:
                    NCPP = IDEN[II, 1]
                    #
                    #      cross correlation
                    #
                    MTT = IDEN[II, 8]
                    if not (MTT == 2 and NP == 1) and \
                       not (MTTP == 2 and NCPP == 1):

                        NCST = IDEN[II, 2]
                        NCED = NCPP + NCST - 1
                        for K in fort_range(NALT, NADD1):  # .lbl278
                            C1 = data.DCS[K]
                            for KK in fort_range(NCST, NCED):  # .lbl279
                                # TODO: The need to use effDCS to have consistency with the
                                #       Fortran version is probably a bug. Either original
                                #       or PPP corrected uncertainties should be used everywhere.
                                #       Here we use a mix of uncorrected (C1) and corrected (C2)
                                #       uncertainties.
                                C2 = data.effDCS[KK]
                                Q1 = 0.
                                for KKK in fort_range(1,10):  # .lbl281
                                    NC1 = NEC[ID, 1, KKK, I]
                                    NC2 = NEC[ID, 2, KKK, I]
                                    if NC1 > 21 or NC2 > 21:
                                        continue

                                    if NC1 == 0 or NC2 == 0:
                                        break

                                    AMUFA = data.FCFC[ID, KKK, I]
                                    if NC1 > 10:
                                        NC1 = NC1 - 10
                                        if NETG[NC1, ID] == 9:
                                            FKT = 1.
                                        else:
                                            FKT = data.EPAF[1, NC1, ID] + data.EPAF[2, NC1, ID]

                                        C11 = FKT*data.CO[NC1, K]
                                    else:
                                        C11 = data.ENFF[ID, NC1]

                                    if NC2 > 10:
                                        NC2 = NC2 - 10

                                        if NETG[NC2, II] == 9:
                                            FKS = 1.
                                        else:
                                            XYY = data.EPAF[2,NC2,II] - np.abs(data.E[K]-data.E[KK])/ (data.EPAF[3,NC2,II]*data.E[KK])
                                            if XYY < 0.:
                                                XYY = 0.
                                            FKS = data.EPAF[1, NC2, II] + XYY

                                        C22 = FKS * data.CO[NC2, KK]

                                    else:
                                        C22 = data.ENFF[II, NC2]

                                    Q1 = Q1 + AMUFA*C11*C22

                                data.ECOR[K,KK] = Q1/(C1*C2)
                                data.ECOR[KK,K] = data.ECOR[K,KK]

    data.problematic_L_Ecor[ID] = L
    return



def apply_PPP_correction(ID, data, APR):

    start_idx, end_idx = get_dataset_range(ID, data)
    for idx in fort_range(start_idx, end_idx):
        AX = get_AX(ID, idx, data, APR)
        AZ = AX / data.CSS[idx]
        data.effDCS[idx] = AZ*data.DCS[idx]



def init_shape_prior(ID, data, APR):
    #
    #      DETERMINE APRIORI NORMALIZATION FOR SHAPE MEASUREMENTS
    #

    MT = data.IDEN[ID, 7]
    MTTP = data.IDEN[ID,8]
    if MT == 6 or MTTP != 2:
        raise ValueError('init_shape_prior can only be used for shape data and MT != 6')

    start_idx, end_idx = get_dataset_range(ID, data)

    AP = 0.
    WWT = 0.
    for K in fort_range(start_idx, end_idx):  # .lbl29
        CSSK = data.CSS[K]
        DCSK = data.DCS[K]
        WXX = 1./(DCSK*DCSK)
        WWT = WWT + WXX

        AX = get_AX(ID, K, data, APR)
        AZ = AX / CSSK

        AP=AP+AZ*WXX

    NS = data.IDEN[ID,6]
    L = APR.NR + np.where(APR.NSETN == NS)[0][0]
    AP=AP/WWT
    AP = 1.0 / AP
    APR.CS[L] = AP

    return



def is_usable_datapoint(idx, data):

    ID = get_dataset_id_from_idx(idx, data)
    J = data.KAS[idx,1]
    I = data.KAS[idx,2]
    I8 = data.KAS[idx,3]
    MT = data.IDEN[ID, 7]

    if (J == 0 and MT != 6) or \
       (I == 0 and MT in (3,4,5,7,8,9)) or \
       (I8 == 0 and MT in (7,9)):
           return False
    else:
        return True



def count_usable_datapoints(data, end_idx=None):
    end_idx = data.num_datapoints if end_idx is None else end_idx

    N = 0
    if data.num_datapoints > 0:
        for idx in fort_range(1, end_idx):
            if is_usable_datapoint(idx, data):
                N += 1
    return N



def fill_AA_AM_COV(data, fisdata, APR, gauss):
    #
    #      FILL AA,AM,AND COV
    #
    IDEN = data.IDEN
    KAS = data.KAS
    KA = data.KA
    N = 0

    gauss.AM.fill(0.)
    gauss.AA.fill(0.)

    for KS in fort_range(1, data.num_datapoints):  # .lbl18

        ID = get_dataset_id_from_idx(KS, data)

        MT =  IDEN[ID, 7]
        NT = data.NT[ID,:]
        NCT = data.NCT[ID]
        NS = IDEN[ID,6]
        MTTP = data.IDEN[ID, 8]

        if MTTP == 2:
            NSHP_IDX = np.where(APR.NSETN == NS)[0]
            if len(NSHP_IDX) > 1:
                raise IndexError('Dataset ' + str(NS) + ' appears multiple times')
            NSHP_IDX = NSHP_IDX[0]

        data.invalid_datapoints[NS] = []

        DQQQ = data.effDCS[KS]*data.CSS[KS]*0.01

        if not is_usable_datapoint(KS, data):
            data.invalid_datapoints[NS].append(KS)
            continue

        N = N + 1

        if MT == 6:
            #
            #      FISSION AVERAGE
            #
            K = 0
            if NT[1] == 9:
                K = 1

            JA, JE = get_prior_range(NT[1], APR)
            NW = 2 if NT[1]==9 else 1
            FL=0.
            SFL=0.

            for LI in fort_range(JA+1, JE-1):  # .lbl53
                NW=NW+1
                FL=FL+fisdata.FIS[NW]
                EL1=(APR.EN[LI]+APR.EN[LI-1])*0.5
                EL2=(APR.EN[LI]+APR.EN[LI+1])*0.5
                DE1=(APR.EN[LI]-EL1)*0.5
                DE2=(EL2-APR.EN[LI])*0.5
                SS1=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI-1]))
                SS2=.5*(APR.CS[LI]+0.5*(APR.CS[LI]+APR.CS[LI+1]))
                CSSLI=(SS1*DE1+SS2*DE2)/(DE1+DE2)
                SFL=SFL+CSSLI*fisdata.FIS[NW]

            FL=FL+fisdata.FIS[1]+fisdata.FIS[NW+1]
            SFL=SFL+fisdata.FIS[1]*APR.CS[JA]+fisdata.FIS[NW+1]*APR.CS[JE]
            SFIS=SFL/FL

            EAVR = 0.

            data.EAVR[KS] = EAVR
            data.SFIS[KS] = SFIS
            data.FL[KS] = FL

            CX=SFIS
            for J in fort_range(JA, JE):  # .lbl39
                K=K+1
                KA[J,1]=KA[J,1]+1
                KR=KA[J,1]
                KA[J,KR+1]=N
                if J == JA or J == JE:
                    CSSJ = APR.CS[J]
                else:
                    EL1=(APR.EN[J]+APR.EN[J-1])*0.5
                    EL2=(APR.EN[J]+APR.EN[J+1])*0.5
                    DE1=(APR.EN[J]-EL1)*0.5
                    DE2=(EL2-APR.EN[J])*0.5
                    SS1=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J-1]))
                    SS2=.5*(APR.CS[J]+0.5*(APR.CS[J]+APR.CS[J+1]))
                    CSSJ=(SS1*DE1+SS2*DE2)/(DE1+DE2)

                gauss.AA[J,KR]=CSSJ*fisdata.FIS[K]/DQQQ

            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 5:
            #
            #      TOTAL CROSS SECTION
            #
            CX = 0.
            for I in fort_range(1,NCT):  # .lbl49
                II = KAS[KS,I]
                CX = CX+APR.CS[II]


            for I in fort_range(1,NCT):  # .lbl60
                J  = KAS[KS,I]
                KA[J,1] = KA[J,1]+1
                KR = KA[J,1]
                KA[J,KR+1] = N
                gauss.AA[J,KR] = APR.CS[J]/DQQQ

            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 8:
            #
            #   SHAPE OF SUM
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = 0.
            for I in fort_range(1,NCT):  # .lbl253
                II=KAS[KS,I]
                CX=CX+APR.CS[II]*AP

            APDQ=AP/DQQQ
            for I in fort_range(1,NCT):  # .lbl254
                J=KAS[KS,I]
                KA[J,1]=KA[J,1]+1
                KR=KA[J,1]
                KA[J,KR+1]=N
                gauss.AA[J,KR]=APR.CS[J]*APDQ

            KA[L,1]=KA[L,1]+1
            KR=KA[L,1]
            KA[L,KR+1]=N
            gauss.AA[L,KR]=CX/DQQQ
            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue


        J = KAS[KS,1]
        I = KAS[KS,2]
        I8 = KAS[KS,3]

        KA[J,1] = KA[J,1] + 1
        KR = KA[J,1]
        KA[J,KR+1] = N

        if MT == 1:
            #
            #      CROSS SECTION
            #
            CX = APR.CS[J]
            gauss.AA[J,KR] = CX / DQQQ
            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 2:
            #
            #      CROSS SECTION SHAPE    L is shape data norm. const. index
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = APR.CS[J]*AP
            CXX = CX/DQQQ
            gauss.AA[J,KR] = CXX
            KA[L,1] = KA[L,1]+1
            KR = KA[L,1]
            KA[L,KR+1] = N
            gauss.AA[L,KR] =  CXX
            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 3:
            #
            #      RATIO
            #
            CX = APR.CS[J]/APR.CS[I]
            CCX = CX/DQQQ
            gauss.AA[J,KR] = CCX
            KA[I,1] = KA[I,1]+1
            KR = KA[I,1]
            KA[I,KR+1] = N
            gauss.AA[I,KR] = -CCX
            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 4:
            #
            #      RATIO SHAPE
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CX = APR.CS[J]*AP/APR.CS[I]
            CXX = CX/DQQQ
            gauss.AA[J,KR] = CXX
            KA[I,1] = KA[I,1]+1
            KR = KA[I,1]
            KA[I,KR+1] = N
            gauss.AA[I,KR] = -CXX
            KA[L,1] = KA[L,1]+1
            KR = KA[L,1]
            KA[L,KR+1] = N
            gauss.AA[L,KR] =  CXX
            gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
            continue

        elif MT == 7:
            #
            #   ABSOLUTE RATIO S1/(S2+S3)
            #
            CX=APR.CS[J]/(APR.CS[I]+APR.CS[I8])
            if I == J:
                CBX=CX*CX*APR.CS[I8]/(APR.CS[J]*DQQQ)
                gauss.AA[J,KR]=CBX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                gauss.AA[I8,KR]=-CBX
                gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue
            else:
                CBX=CX/DQQQ
                gauss.AA[J,KR]=CBX
                KA[I,1]=KA[I,1]+1
                KR=KA[I,1]
                KA[I,KR+1]=N
                CBX2=CBX*CBX*DQQQ/APR.CS[J]
                CCX=CBX2*APR.CS[I]
                gauss.AA[I,KR]=-CCX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                CCX=CBX2*APR.CS[I8]
                gauss.AA[I8,KR]=-CCX
                gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue

        elif MT == 9:
            #
            #   SHAPE OF RATIO S1/(S2+S3)
            #
            L = APR.NR + NSHP_IDX
            AP = APR.CS[L]
            CII8=APR.CS[I]+APR.CS[I8]
            CX=AP*APR.CS[J]/CII8
            CBX=CX/DQQQ
            if I == J:
                CCX=CBX*APR.CS[I8]/CII8
                gauss.AA[J,KR]=CCX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                gauss.AA[I8,KR]=-CCX
                KA[L,1]=KA[L,1]+1
                KR=KA[L,1]
                KA[L,KR+1]=N
                gauss.AA[L,KR]=CBX
                gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue
            else:
                gauss.AA[J,KR]=CBX
                KA[I,1]=KA[I,1]+1
                KR=KA[I,1]
                KA[I,KR+1]=N
                CDX=CBX*APR.CS[I]/CII8
                gauss.AA[I,KR]=-CDX
                KA[I8,1]=KA[I8,1]+1
                KR=KA[I8,1]
                KA[I8,KR+1]=N
                CDX=CBX*APR.CS[I8]/CII8
                gauss.AA[I8,KR]=-CDX
                KA[L,1]=KA[L,1]+1
                KR=KA[L,1]
                KA[L,KR+1]=N
                gauss.AA[L,KR]=CBX
                gauss.AM[N]=(data.CSS[KS]-CX)/DQQQ
                continue

    return



def complete_symmetric_Ecor(data):
    #
    #      FILL IN SYMMETRIC TERM
    #
    MODC = data.MODC
    N = data.num_datapoints_used
    for K in fort_range(1,N-1):  # .lbl25
        for L in fort_range(K+1, N):  # .lbl25
            if MODC == 2:
                data.ECOR[L, K] = 0.
            data.ECOR[K, L] = data.ECOR[L, K]
            # label .lbl25



def invert_Ecor(data):
    #
    #      INVERT ECOR
    #
    MODC = data.MODC
    N = count_usable_datapoints(data)

    data.effECOR = data.ECOR.copy()

    if MODC == 2 or N == 1:
        data.invECOR = data.ECOR.copy()
        return True

    IREP = 0
    while True:
        # cholesky decomposition
        #CALL DPOFA(ECOR,LDA,N,INFO)
        # INFO = np.array(0)
        # choleskymat = np.array(data.effECOR[1:(N+1),1:(N+1)], dtype='float64', order='F')
        # linpack_slim.dpofa(a=choleskymat, info=INFO)

        # ALTERNATIVE USING NUMPY FUNCTION cholesky
        INFO = 0
        try:
            choleskymat = np.linalg.cholesky(data.effECOR[1:(N+1), 1:(N+1)]).T
        except np.linalg.LinAlgError:
            INFO = 1

        if INFO == 0:
            break
        else:
            #
            #      ATTEMPT TO MAKE CORR. MATRIX POSITIVE DEFINITE
            #
            data.num_inv_tries += 1
            IREP=IREP+1
            N1=N-1
            for K in fort_range(1,N1):  # .lbl2211
                K1=K+1
                for L in fort_range(K1, N):  # .lbl2211
                    if MODC == 2:
                        data.effECOR[L,K] = 0.
                    data.effECOR[K,L] = data.effECOR[L,K]

            for K in fort_range(1,N):  # .lbl2212
                data.effECOR[K,K] = 1.

            CXZ=0.10
            for K in fort_range(1,N):  # .lbl37
                for L in fort_range(1,N):
                    data.effECOR[K,L]=data.effECOR[K,L]/(1.+CXZ)
                    if K == L:
                        data.effECOR[K,L] = 1.

            if IREP >= 15:
                return False

    JOB=1
    # CALL DPODI(ECOR,LDA,N,DET,JOB)
    tmp = np.array(choleskymat, dtype='float64', order='F')
    tmp_det = np.array([0., 0.], dtype='float64', order='F') 
    linpack_slim.dpodi(tmp, det=tmp_det, job=JOB)
    data.invECOR[1:(N+1),1:(N+1)] = tmp

    # ALTERNATIVE USING NUMPY inv function
    # tmp = inv(data.ECOR[1:(N+1),1:(N+1)])
    # data.ECOR[1:(N+1),1:(N+1)] = np.matmul(tmp.T, tmp)

    for K in fort_range(2,N):  # .lbl17
        L1=K-1
        for L in fort_range(1,L1):
            data.invECOR[K,L] = data.invECOR[L,K]
        L = L + 1  # to match L value of fortran after loop

    return True



def get_matrix_products(gauss, data, APR):
    #
    #      GET MATRIX PRODUCTS
    #
    NRS=APR.NR + APR.NSHP
    KA = data.KA
    N = data.num_datapoints_used
    gauss.NTOT += data.num_datapoints_used
    NTOT = gauss.NTOT
    SIGMA2 = gauss.SIGMA2

    for I in fort_range(1,NRS):  # .lbl90
        NI=KA[I,1]
        if NI == 0:
            continue

        for J in fort_range(I, NRS):  # .lbl83
            NJ=KA[J,1]
            if NJ == 0:
                continue
            IJ=J*(J-1)//2+I

            for MI in fort_range(1,NI):  # .lbl85
                MIX=KA[I,MI+1]
                for MJ in fort_range(1,NJ):  # .lbl85
                    MJX=KA[J,MJ+1]
                    gauss.B[IJ]=gauss.B[IJ]+gauss.AA[I,MI]*gauss.AA[J,MJ]*data.invECOR[MIX,MJX]

    for I in fort_range(1,NRS):  # .lbl91
        NI=KA[I,1]
        if NI == 0:
            continue

        for MI in fort_range(1,NI):  # .lbl86
            MIX=KA[I,MI+1]
            for MJ in fort_range(1,N):  #.lbl86
                gauss.BM[I]=gauss.BM[I]+gauss.AA[I,MI]*data.invECOR[MIX,MJ]*gauss.AM[MJ]

    for I in fort_range(1,N):  # .lbl26
        SUX=0.
        for J in fort_range(1,N):  # .lbl52
            SUX=SUX+data.invECOR[I,J]*gauss.AM[J]
        
        SIGMA2=SIGMA2+gauss.AM[I]*SUX

    data.NTOT = NTOT
    data.SIGL=SIGMA2/NTOT
    if N > LDA:
        exit()
    if NRS > LDB:
        exit()

    gauss.SIGMA2 = SIGMA2



def get_result(gauss, APR):
    #
    #      GETTING THE RESULT
    #
    NRS = APR.NR + APR.NSHP
    NUMEL = NRS*(NRS+1)//2
    tmp = np.array(gauss.B[1:(NUMEL+1)], dtype='float64', order='F')
    tmp = unpack_utriang_matrix(tmp)
    try:
        tmp = np.linalg.cholesky(tmp.T).T 
    except np.linalg.LinAlgError:
        format105 = "(/' EXP BLOCK CORREL. MATRIX NOT PD',20X,'***** WARNING *')" 
        format106 = "( '  SOLUTION  CORREL. MATRIX NOT PD ' )"
        fort_write(None, format106)
        exit()
    tmp = np.linalg.inv(tmp)
    tmp = np.matmul(tmp, tmp.T)
    gauss.DE[1:(NRS+1)]=gauss.DE[1:(NRS+1)]+ np.matmul(tmp, gauss.BM[1:(NRS+1)])

    gauss.B[1:(NUMEL+1)] = pack_symmetric_matrix(tmp)


def output_result(gauss, fisdata, APR, MODAP,
        file_IO4, file_IO5):
    #
    #      output of the result
    #
    NR = APR.NR
    NC = APR.NC
    NSHP = APR.NSHP
    NRS = NR + NSHP
    NFIS = fisdata.NFIS
    NSETN = APR.NSETN

    for L in fort_range(1,NC):  # .lbl14
        format117 = "(1H1,'   RESULT',5X,2A8//)" 
        fort_write(file_IO4, format117, [APR.CLAB[L,1:3]])
        fort_write(file_IO5, format117, [APR.CLAB[L,1:3]])
        format112 = "( '   E/MEV         CS/B            DCS/B       DCS/%" + \
                    "     DIF/%    CS*SQRT(E)'/)"
        fort_write(file_IO4, format112, [])
        JA=APR.MCS[L,2]
        JI=APR.MCS[L,3]
        FLX=0.

        for K in fort_range(JA, JI):  # .lbl77
            KBK=K*(K-1)//2+K
            DDX=APR.CS[K]*np.sqrt(gauss.B[KBK])
            CXX=APR.CS[K]*(1.+gauss.DE[K])
            CXXD=100.*(CXX-APR.CS[K])/CXX

            found = False
            for KK in fort_range(1,NFIS):  # .lbl705
                if fisdata.ENFIS[KK] > .999*APR.EN[K] and fisdata.ENFIS[KK] < 1.001*APR.EN[K]:
                    found = True
                    break

            if found:
                if K == JA or K == JI:
                    CSSK=CXX
                else:
                    EL1=(APR.EN[K]+APR.EN[K-1])*0.5
                    EL2=(APR.EN[K]+APR.EN[K+1])*0.5
                    DE1=(APR.EN[K]-EL1)*0.5
                    DE2=(EL2-APR.EN[K])*0.5
                    SS1=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K-1])*APR.CS[K-1]))
                    SS2=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K+1])*APR.CS[K+1]))
                    CSSK=(SS1*DE1+SS2*DE2)/(DE1+DE2)

                FLX=FLX+fisdata.FIS[KK]*CSSK

            FQW=DDX*100./CXX
            SECS=np.sqrt(APR.EN[K])*CXX
            format153 = "(1X,E10.4,2F15.8,2X,F6.1,3X,F7.2,3X,F10.5)" 
            fort_write(file_IO4, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            fort_write(file_IO5, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            if not (MODAP == 0) and \
               not (MODAP == 2 and K <= APR.MCS[5,3]):
                APR.CS[K]=CXX

        # VP: 13 lines below are added by VP, 26 July, 2004
        format588 = "(6(1X,E10.5))"
        fort_write(file_IO4, format588, [
            (APR.EN[JA]*500000.),
            (APR.EN[JA:JI]+APR.EN[(JA+1):(JI+1)])*500000.,
            (-APR.EN[JI-1]+3*APR.EN[JI])*500000.
        ])

        tmp = np.vstack([APR.EN[JA:(JI+1)]*1000000., APR.CS[JA:(JI+1)]])
        tmp = tmp.T.flatten()
        fort_write(file_IO4, format588, tmp)
        for K in fort_range(JA+1, JI-1):
            DSMOOA = (APR.CS[K+1] * (APR.EN[K] - APR.EN[K-1]) \
                    +APR.CS[K-1] * (APR.EN[K+1] - APR.EN[K]) \
                    -APR.CS[K] * (APR.EN[K+1] - APR.EN[K-1])) \
                    /2./(APR.EN[K+1] - APR.EN[K-1])
            DSMOOR = DSMOOA / APR.CS[K]*100.
            SSMOO = APR.CS[K] + DSMOOA
            fort_write(file_IO4, format153, [APR.EN[K], APR.CS[K], SSMOO, DSMOOR])
        # VP above is writing CS in B-6 format and smoothing with CS conserving

        format158 = "(1H*//,'  FISSION AVERAGE ' ,F8.4//)" 
        fort_write(file_IO4, format158, [FLX])

    #
    #   OUTPUT OF NORM. FACTORS FOR SHAPE DATA
    #
    format114 = "(1H*///, '  NORMALIZATION  OF SHAPE DATA '///)"
    fort_write(file_IO4, format114, [])
    NR1=NR+1
    LLX=0
    if NSHP != 0:
        for K in fort_range(NR1, NRS):  # .lbl82
            LLX=LLX+1
            KK=K*(K-1)//2+K
            ZCS=APR.CS[K]
            DDX=APR.CS[K]*np.sqrt(gauss.B[KK])
            CXX=APR.CS[K]*(1.+gauss.DE[K])
            DDXD=DDX*100./CXX
            format115 = "(2I6,4F10.4)"
            fort_write(file_IO4, format115, [K,NSETN[LLX],CXX,DDX,DDXD,ZCS])
            APR.CS[K]=CXX

    return APR



def output_result_correlation_matrix(gauss, data, APR, IPP,
        file_IO4):
    #
    #   OUTPUT OF CORRELATION MATRIX OF THE RESULT
    #
    NC = APR.NC
    JA = APR.MCS[NC, 2]

    if IPP[6] != 0:
        format151 = "(1X,24F7.4)"
        for K in fort_range(1,NC):  # .lbl78
            J1=APR.MCS[K,2]
            J2=APR.MCS[K,3]

            # CVP 3 lines below are added by VP, 26 July, 2004
            NROW=J2-J1+2
            for III in fort_range(1, NROW):
                gauss.EGR[III] = 1.0*III
            # CVP

            for L in fort_range(1,K):  # .lbl80
                format122 = "(1H1, '  CORRELATION MATRIX OF THE RESULT   ',2A8,2A8///)"
                fort_write(file_IO4, format122, [APR.CLAB[K,1:3], APR.CLAB[L,1:3]])
                J3=APR.MCS[L,2]
                J4=APR.MCS[L,3]

                # CVP 3 lines below are added by VP, 26 July 2004
                NCOL = J4-J3+2
                for III in fort_range(1, NROW+NCOL):
                    gauss.EEGR[III] = 1.0*III
                # CVP

                if K == L:
                    for I in fort_range(J1, J2):  # .lbl55
                        II=I*(I-1)//2+I
                        for J in fort_range(J1,I):  # .lbl27
                            IJ=I*(I-1)//2+J
                            JJ=J*(J-1)//2+J
                            gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                            # CVP lines below are added by VP, 26 July, 2004
                            gauss.RELTRG[I-J1+1,J-J1+1] = gauss.B[IJ]
                            data.AAA[I-J1+1, J-J1+1] = gauss.BM[J]
                            data.AAA[J-J1+1, I-J1+1] = gauss.BM[J]
                            # CVP end

                        fort_write(file_IO4, format151, [gauss.BM[J1:(I+1)]])

                    format389 = '(2x,f7.3,1x,200(E10.4,1x))'
                    IMAX = J2-J1+1
                    for I in fort_range(1,IMAX):
                        fort_write(file_IO4, format389,
                                [APR.EN[JA+I-1], data.AAA[I,1:(J2-J1+2)]])

                    # CVP   Lines below are added by VP, 26 July, 2004
                    format388 = '(6E11.4)'
                    tmp = [[gauss.RELTRG[III,JJJ]
                            for III in fort_range(JJJ,NROW-1)]
                            for JJJ in fort_range(1, NROW-1)]
                    fort_write(file_IO4, format388,
                            [gauss.EGR[1:(NROW+1)], tmp])
                    # CVP

                else:

                    for I in fort_range(J1, J2):  # .lbl88
                        II=I*(I-1)//2+I
                        for J in fort_range(J3, J4):  # .lbl16
                            IJ=I*(I-1)//2+J
                            JJ=J*(J-1)//2+J
                            gauss.BM[J]=gauss.B[IJ]/np.sqrt(gauss.B[II]*gauss.B[JJ])
                            # CVP three lines below are inserted by VP
                            gauss.RELCOV[I-J1+1, J-J3+1] = gauss.B[IJ]
                            data.AAA[I-J1+1, J-J3+1] = gauss.BM[J]
                            data.AAA[J-J3+1, I-J1+1] = gauss.BM[J]
                            # CVP

                        fort_write(file_IO4, format151, [gauss.BM[J3:(J4+1)]]) 

                    # CVP   Lines below are added by VP, 26 July, 2004
                    format388 = '(6E11.4)'
                    fort_write(file_IO4, format388,
                            [gauss.EEGR[1:(NROW+NCOL+1)],
                                gauss.RELCOV[1:NROW, 1:NCOL].flatten()])
                    fort_write(file_IO4, format388,
                            [gauss.EEGR[1:(NROW+NCOL+1)],
                             gauss.RELCOV[1:NROW, 1:NCOL].flatten(order='F')])
                    # CVP   print below is inserted by VP Aug2013
                    IMAX = J2-J1+1
                    format389 = '(2x,f7.3,1x,200(E10.4,1x))'
                    for I in fort_range(1, IMAX):
                        fort_write(file_IO4, format389,
                                [APR.EN[JA+I-1],
                                 data.AAA[I,1:(J4-J3+2)]])




def input_fission_spectrum(MC1, file_IO3, file_IO4):
    #
    #      INPUT OF FISSION SPECTRUM
    #
    #

    #   Fission Data block / data set
    #
    #      ENFIS   ENERGIES OF FISSION SPECTRUM
    #      FIS     FISSION SPECTRUM*BINWIDTH
    #
    data = Bunch({
        'FIS': np.zeros(250+1, dtype=float),
        'ENFIS': np.zeros(250+1, dtype=float),
        'NFIS': 0
        })

    if MC1 == 0:
        #
        #       MAXWELLIAN SPECTRUM
        #
        EAVR=MC2/1000.
        NFIS=APR.MCS[MC3,1]
        JA=APR.MCS[MC3, 2]
        JE=APR.MCS[MC3, 3]
        LL=0
        for K in fort_range(JA, JE):  # .lbl693
            LL=LL+1

        data.ENFIS[LL] = APR.EN[K]
        NFIS1=NFIS-1
        FISUM=0.
        for K in fort_range(1, NFIS1):  # .lbl695
            E1=(data.ENFIS[K-1]+data.ENFIS[K])/2.
            E2=(data.ENFIS[K+1]+data.ENFIS[K])/2.
            DE12=E2-E1
            F1=np.sqrt(E1)*np.exp(-1.5*E1/EAVR)
            F2=np.sqrt(E2)*np.exp(-1.5*E2/EAVR)
            E12=(E1+E2)/2.
            F3=np.sqrt(E12)*np.exp(-1.5*E12/EAVR)
            data.FIS[K]=((F1+F2)*.5+F3)*.5
            FISUM=FISUM+DE12*data.FIS[K]

        data.FIS[1]=np.sqrt(APR.EN[JA])*np.exp(-1.5*APR.EN[JA]/EAVR)
        data.FIS[NFIS]=np.sqrt(APR.EN[JE])*np.exp(-1.5*APR.EN[JE]/EAVR)
        DE12=(data.ENFIS[2]-data.ENFIS[1])/2.
        DE13=data.ENFIS[1]+DE12
        FISUM=FISUM+data.FIS[1]*DE13
        DE14=(data.ENFIS(NFIS)-data.ENFIS[NFIS1])/2.
        FISUM=FISUM+data.FIS[NFIS]*2.*DE14

    else:
        format119 = "(2E13.5)"
        for K in fort_range(1,LDF):  # .lbl690
            data.ENFIS[K], data.FIS[K] = fort_read(file_IO3, format119)
            if data.ENFIS[K] == 0:
                break

        NFIS = K - 1

    format800 = "(/' FISSION SPECTRUM * BIN WIDTH'/)"
    fort_write(file_IO4, format800, [])
    if MC1 == 0:
        for K in fort_range(2, NFIS1):  # .lbl696
            E1=(data.ENFIS[K-1]+data.ENFIS[K])/2.
            E2=(data.ENFIS[K+1]+data.ENFIS[K])/2.
            DE12=E2-E1

        data.FIS[K]=data.FIS[K]*DE12/FISUM
        data.FIS[NFIS]=data.FIS[NFIS]*DE14/FISUM
        data.FIS[1]=data.FIS[1]*DE13/FISUM

    format157 = "(2F10.6)"
    for KQ in fort_range(1,NFIS):  # .lbl694
        fort_write(file_IO4, format157, [data.ENFIS[KQ], data.FIS[KQ]])

    data.NFIS = NFIS
    return data



def link_prior_and_datablocks(APR, datablock_list, MODREP):
    APR.NSHP = 0
    for data in datablock_list:
        if data.num_datasets == 0:
            continue

        for ID in fort_range(1, data.num_datasets):
            NS = data.IDEN[ID,6]
            if data.IDEN[ID, 7] != 6:
                MTTP = data.IDEN[ID, 8]
                if MTTP == 2:
                    APR.NSHP += 1
                    APR.NSETN[APR.NSHP] = NS
                    data.problematic_L_dimexcess[ID] = APR.NR + APR.NSHP
                    if APR.NR + APR.NSHP > SIZE_LIMITS.MAX_NUM_UNKNOWNS:
                        raise IndexError('Too many shape datasets')

        for ID in fort_range(1, data.num_datasets):
            accounting(ID, data, APR)
            data.num_datapoints_used = count_usable_datapoints(data)

        for ID in fort_range(1, data.num_datasets):
            MT = data.IDEN[ID,7]
            MTTP = data.IDEN[ID,8]
            if MT != 6 and MTTP == 2 and MODREP == 0:
                init_shape_prior(ID, data, APR)

