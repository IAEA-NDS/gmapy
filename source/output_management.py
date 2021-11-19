from fortran_utils import fort_range, fort_write
from gmap_snippets import (should_downweight, get_AX, get_dataset_range,
        get_num_shapedatasets)
from data_management import SIZE_LIMITS
import numpy as np


def write_overflow_message(ID, data, APR, file_IO4):
    IDEN = data.IDEN
    MTTP = IDEN[ID, 8]
    NSHP = APR.NSHP
    NR = APR.NR
    if data.IDEN[ID, 7] != 6:
        if MTTP == 2:
            NSHP = NSHP + 1
            L = NR + NSHP
            if L > SIZE_LIMITS.MAX_NUM_UNKNOWNS:
                format701 = "( '   OVERFLOW OF UNKNOWN-VECTOR SPACE WITH SET  ',I3)"
                fort_write(file_IO4, format701, [NS])



def write_KAS_check(ID, data, IPP, file_IO4):
    if data.IDEN[ID, 7] != 6:
        #
        #   output of KAS for checking
        #
        if IPP[7] != 0:
            format702 = "(20I5)"
            NCT = data.NCT[ID]
            dataset_start_index, dataset_end_index = get_dataset_range(ID, data)
            for K in fort_range(1,NCT):
                fort_write(file_IO4, format702,
                        [data.KAS[dataset_start_index:(dataset_end_index+1)], K])



def write_datablock_header(file_IO4):
    format108 = "(/' DATABLOCK************************DATABLOCK**************" + \
                "******************************************DATABLOCK '/)"
    fort_write(file_IO4, format108, [])



def write_dataset_info(ID, data, APR, LABL, file_IO4):

    IDEN = data.IDEN
    NCT = data.NCT[ID]
    NT = data.NT[ID,:]
    NS = IDEN[ID,6]
    MT = IDEN[ID,7]

    format142 = "(//, ' ***********DATASET**************************** '/)"
    fort_write(file_IO4, format142, [])
    NU = NCT
    if NCT > 4:
        NU = 4
    NCT2 = NCT - NU
    NU1 = NU + 1
    format139 = "(2X,8HDATA SET,I5,2X,A16,4(2X,2A8))"
    tmp = [[APR.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(1,NU)]
    fort_write(file_IO4, format139, [NS, LABL.TYPE[MT],tmp])
    if NCT2 > 0:
        format149 = "(2X,6(2X,2A8))"
        tmp = [[APR.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(NU1,NCT2)]
        fort_write(file_IO4, format149, tmp)

    #
    #       NAME ID AND REFERENCE I/O
    #
    format132 = "(/' YEAR',I5,' TAG',I3,' AUTHOR:  ',4A8,4A8/)"
    fort_write(None, format132, [IDEN[ID, 3:5], data.CLABL[ID,1:5], data.BREF[ID,1:5]])
    # VP      if(modrep .ne. 0) go to 183
    fort_write(file_IO4, format132, [IDEN[ID, 3:5], data.CLABL[ID,1:5], data.BREF[ID,1:5]])

    if not should_downweight(ID, data) and (data.MOD2 > 1 and data.MOD2 < 10):
        format339 = "('  WEIGHTING OPTION NOT IMPLEMENTED, DATA SET  ',I5/)"
        fort_write(file_IO4, format339, NS)



def write_prior_info(APR, IPP, file_IO4):
    # from here onwards only output
    NC = APR.NC
    NR = APR.NR
    # label .lbl30
    format134 = r"(//2X,36HCROSS SECTIONS OF PRESENT EVALUATION//)"
    fort_write(file_IO4, format134, [])
    format135 = "(10X,I3,5X,2A8)"
    for K in fort_range(1,NC):
        fort_write(file_IO4, format135, [K, APR.CLAB[K, 1:3]])
    # label .lbl22
    if IPP[1] != 0:
        format136 = "(1H1//,2X,35HENERGIES AND APRIORI CROSS SECTIONS//)" 
        fort_write(file_IO4, format136, [])
        format137 = "(/ '     INDEX     E/MEV   ',7X,2A8 /)"
        for  K in fort_range(1,NC):  # .lbl24
            fort_write(file_IO4, format137, APR.CLAB[K,1:3])
            JC1 = APR.MCS[K, 2]
            JC2 = APR.MCS[K, 3]
            LQ = 0
            format138 = "(2X,2I4,3X,E10.4,3X,F15.8)"
            for L in fort_range(JC1, JC2):
                LQ += 1
                fort_write(file_IO4, format138, [LQ, L, APR.EN[L], APR.CS[L]])

    format113 = "(/,' TOTAL NO OF PARAMETERS ',I4/)"
    fort_write(file_IO4, format113, [NR])

    #
    #      for checking
    #
    if IPP[7] != 0:
        format4390 = "(' No of Parameters per Cross Section '/)"
        fort_write(file_IO4, format4390, [])
        format154 = "(3X,3HAPR.MCS,10I5)"
        fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 1]])
        format4391 = "(/' Start Address '/)"
        fort_write(file_IO4, format4391, [])
        fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 2]])
        format4392 = "(/' End Address '/)"
        fort_write(file_IO4, format4392, [])
        fort_write(file_IO4, format154, [APR.MCS[1:(NC+1), 3]])



def output_Ecor_matrix(data, file_IO4):
    #
    #      output of correlation matrix of data block
    #
    N = data.num_datapoints_used

    format101 = "(1H*//,'   CORRELATION MATRIX OF DATA BLOCK'/)"
    fort_write(file_IO4, format101, [])
    format151 = "(1X,24F7.4)"
    for K in fort_range(1,N):
        fort_write(file_IO4, format151, [data.ECOR[K,1:(K+1)]])



def write_dataset_exclusion_info(NS, data, file_IO4):
    if NS in data.excluded_datasets:
        # label .lbl517
        format168 = "(' SET ',I5,' W/O VALID POINTS OR ELIMINATED'/)"
        fort_write(file_IO4, format168, [NS])


def write_missing_dataset_info(ID, data, file_IO4):
    NS = data.IDEN[ID,6]
    if NS in data.missing_datasets:
        for NSET in data.missing_datasets[NS]:
            format274 = "('CORRELATED DATA SET  ',I5,' NOT FOUND FOR SET ',I5)"
            fort_write(file_IO4, format274, [NSET, NS])


def write_invalid_datapoints_info(NS, data, file_IO4):
    if NS in data.invalid_datapoints:
        for KS in data.invalid_datapoints[NS]:
            format704 = "( '  DATA POINT BUT NOT AN AP FOR SET ',I5,' NO ',I4)"
            fort_write(file_IO4, format704, [NS, KS])


def write_dataset_table(ID, data, APR, MPPP, IPP, file_IO4):

    if data.IDEN[ID, 7] == 6:
        return

    IDEN = data.IDEN
    NS = IDEN[ID, 6]

    NADD = 1
    for xID in fort_range(1, ID):
        NADD += IDEN[xID, 1]

    NALT = NADD - IDEN[ID, 1]
    NADD1 = NADD - 1

    if ID in data.problematic_L_dimexcess:
        problematic_L = data.problematic_L_dimexcess[ID]
    else:
        problematic_L = data.problematic_L_Ecor[ID]

    L = problematic_L

    #VP   PRIOR/EXP column is added
    format5173 = "(/'  ENERGY/MEV   VALUE    ABS. UNCERT. " + \
                 " PRIOR/EXP UNCERT./%    DIFF./%" + \
                 "  VAL.*SQRT(E)'/)"
    fort_write(file_IO4, format5173, [])

    AP = 0.
    WWT = 0.
    for K in fort_range(NALT, NADD1):
        CSSK = data.CSS[K]
        DCSK = data.DCS[K]

        AX = get_AX(ID, K, data, APR)
        AZ = AX / CSSK

        WXX = 1./(DCSK*DCSK)
        WWT = WWT + WXX

        #VPEND 
        #
        #      DATA OUTPUT
        #
        if IPP[2] != 0:
            SECS = np.sqrt(data.E[K])*CSSK
            FDQ = DCSK * CSSK/100.
            DIFF = (CSSK-AX)*100./AX
            #VP   AZ print out was added
            format133 = "(2X,E10.4,2X,E10.4,2X,E10.4,3X,F6.4,3X,F6.2," + \
                        " 3X,F10.2,3X,F10.4)"
            fort_write(file_IO4, format133, [data.E[K], CSSK, FDQ, AZ, DCSK, DIFF, SECS])
            #VP   Print out for Ratio of pior/exp value is added

        AP=AP+AZ*WXX

    AP=AP/WWT

    # VP      if(modrep .ne. 0) go to 2627
    format111 = "(/' APRIORI NORM ',I4,F10.4,I5,2X,4A8)"
    fort_write(file_IO4, format111, [L, AP, NS, data.CLABL[ID,1:5]])



def write_fission_average(ID, data, file_IO4):
    MT = data.IDEN[ID, 7]
    dataset_start_index, dataset_end_index = get_dataset_range(ID, data)
    if MT == 6:
        for KS in fort_range(dataset_start_index, dataset_end_index):
            format156 = "( 'AP FISSION AVERAGE ',3F10.4,'  EXP. VAL. ',2F10.4)"
            fort_write(file_IO4, format156, [data.EAVR[KS], data.SFIS[KS], data.FL[KS],
                data.CSS[KS], data.effDCS[KS]])



def write_added_points_info(APR, data, MODREP, file_IO4):
    N = data.num_datapoints_used
    NTOT = data.NTOT
    NSHP = APR.NSHP
    NR = APR.NR
    NRS=NR+NSHP
    format476 = "(/' ADDED ',I5,' TO GIVE ',I5,' TOTAL',2I5,F10.2/)"
    fort_write(None, format476, [N, NTOT, NSHP, NRS, data.SIGL])
    if MODREP == 0:
        fort_write(file_IO4, format476, [N, NTOT, NSHP, NRS, data.SIGL])



def write_inv_attempt_info(data, IPP, file_IO4):
    for k in range(data.num_inv_tries):
        format105 = "(/' EXP BLOCK CORREL. MATRIX NOT PD',20X,'***** WARNING *')"
        fort_write(file_IO4, format105, [])
    #
    #      output of inverted correlation matrix of data block
    #
    if IPP[5] != 0:
        format151 = "(1X,24F7.4)"
        for K in fort_range(1,N):
            fort_write(file_IO4, format151, [data.invECOR[K,1:(K+1)]])



def write_datablock_info(APR, data, MODREP, MPPP, IPP, LABL, file_IO4):
    if MODREP == 0:
        write_datablock_header(file_IO4)

    for ID in fort_range(1, data.num_datasets):
        write_dataset_info(ID, data, APR, LABL, file_IO4)
        write_missing_dataset_info(ID, data, file_IO4)
        write_KAS_check(ID, data, IPP, file_IO4)
        write_overflow_message(ID, data, APR, file_IO4)
        write_dataset_table(ID, data, APR, MPPP, IPP, file_IO4)
        write_fission_average(ID, data, file_IO4)

    for NS in data.excluded_datasets:
        write_dataset_exclusion_info(NS, data, file_IO4)
    for NS in data.invalid_datapoints:
        write_invalid_datapoints_info(NS, data, file_IO4)

    format2830 = "(80X,4HN = ,I5)"
    fort_write(file_IO4, format2830, [data.num_datapoints_used])

    N = data.num_datapoints_used
    MODC = data.MODC

    if not (IPP[3] == 0 or N == 1 or MODC == 2):
        output_Ecor_matrix(data, file_IO4)

    write_inv_attempt_info(data, IPP, file_IO4)

    write_added_points_info(APR, data, MODREP, file_IO4)



def write_result_info(APR, gauss, IPP, file_IO4):

    LDB = SIZE_LIMITS.MAX_NUM_UNKNOWNS

    NRS = APR.NR + APR.NSHP
    NTOT = gauss.NTOT
    SIGMA2 = gauss.SIGMA2

    format6919 = "(' start getting the result ')"
    fort_write(None, format6919, [])
    SIGMAA=SIGMA2/float(NTOT-NRS)
    format9679 = "(/' UNCERTENTY SCALING   ',E12.4/)"
    fort_write(file_IO4, format9679, [SIGMAA])
    NRST=NRS*(NRS+1)/2
    if IPP[8] ==  0:
        force_stop(file_IO4)
    if IPP[4] != 0:
        format116 = "(1H*//,'  MATRIX PRODUCT'//)"
        fort_write(file_IO4, format116, [])
        format152 = "(2X,10E10.4)"
        fort_write(file_IO4, format152, gauss.B[1:(NRST+1)])

    format2840 = "(80X,9HLDB,NRS= ,2I6,6H  NTOT,I8)"
    fort_write(file_IO4, format2840, [LDB, NRS, NTOT])

    format7103 = "(2E16.8)"
    format6918 = "(' start on matrix inversion ')"
    fort_write(None, format6918, [])

    format9171 = "(' INVERT SOLUTION MATRIX')"
    fort_write(file_IO4, format9171, [])
    fort_write(None, format9171, [])

    format6917 = "(' completed inversion of matrix')"
    fort_write(None, format6917, [])



def output_result(gauss, fisdata, APR, MODAP,
        file_IO4, file_IO5, new_APR_CS = None):
    #
    #      output of the result
    #
    NR = APR.NR
    NC = APR.NC
    NSHP = APR.NSHP
    NRS = NR + NSHP
    NFIS = fisdata.NFIS
    NSETN = APR.NSETN

    if new_APR_CS is None:
        new_APR_CS = APR.CS.copy()
    else:
        new_APR_CS = new_APR_CS.copy()

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
            DDX=new_APR_CS[K]*np.sqrt(gauss.B[KBK])
            CXX=new_APR_CS[K]*(1.+gauss.DE[K])
            CXXD=100.*(CXX-new_APR_CS[K])/CXX

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
                    SS1=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K-1])*new_APR_CS[K-1]))
                    SS2=.5*(CXX+0.5*(CXX+(1.+gauss.DE[K+1])*new_APR_CS[K+1]))
                    CSSK=(SS1*DE1+SS2*DE2)/(DE1+DE2)

                FLX=FLX+fisdata.FIS[KK]*CSSK

            FQW=DDX*100./CXX
            SECS=np.sqrt(APR.EN[K])*CXX
            format153 = "(1X,E10.4,2F15.8,2X,F6.1,3X,F7.2,3X,F10.5)" 
            fort_write(file_IO4, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            fort_write(file_IO5, format153, [APR.EN[K],CXX,DDX,FQW,CXXD,SECS])
            if not (MODAP == 0):
                new_APR_CS[K]=CXX

        # VP: 13 lines below are added by VP, 26 July, 2004
        format588 = "(6(1X,E10.5))"
        fort_write(file_IO4, format588, [
            (APR.EN[JA]*500000.),
            (APR.EN[JA:JI]+APR.EN[(JA+1):(JI+1)])*500000.,
            (-APR.EN[JI-1]+3*APR.EN[JI])*500000.
        ])

        tmp = np.vstack([APR.EN[JA:(JI+1)]*1000000., new_APR_CS[JA:(JI+1)]])
        tmp = tmp.T.flatten()
        fort_write(file_IO4, format588, tmp)
        for K in fort_range(JA+1, JI-1):
            DSMOOA = (new_APR_CS[K+1] * (APR.EN[K] - APR.EN[K-1]) \
                    +new_APR_CS[K-1] * (APR.EN[K+1] - APR.EN[K]) \
                    -new_APR_CS[K] * (APR.EN[K+1] - APR.EN[K-1])) \
                    /2./(APR.EN[K+1] - APR.EN[K-1])
            DSMOOR = DSMOOA / new_APR_CS[K]*100.
            SSMOO = new_APR_CS[K] + DSMOOA
            fort_write(file_IO4, format153, [APR.EN[K], new_APR_CS[K], SSMOO, DSMOOR])
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
            ZCS=new_APR_CS[K]
            DDX=new_APR_CS[K]*np.sqrt(gauss.B[KK])
            CXX=new_APR_CS[K]*(1.+gauss.DE[K])
            DDXD=DDX*100./CXX
            format115 = "(2I6,4F10.4)"
            fort_write(file_IO4, format115, [K,NSETN[LLX],CXX,DDX,DDXD,ZCS])
            new_APR_CS[K]=CXX

    return



def write_iteration_info(APR, datablock_list, fisdata, gauss, MODREP, MODAP, MPPP, IPP, LABL, file_IO4, file_IO5):
    curNSHP = 0
    totNSHP = APR.NSHP
    for data in datablock_list:
        curNSHP += get_num_shapedatasets(data)
        APR.NSHP = curNSHP
        write_datablock_info(APR, data, MODREP, MPPP, IPP, LABL, file_IO4)
        APR.NSHP = totNSHP

    write_result_info(APR, gauss, IPP, file_IO4)

    output_result(gauss, fisdata, APR, MODAP,
                  file_IO4, file_IO5)

