from fortran_utils import fort_range, fort_write
from gmap_snippets import should_downweight
from data_management import SIZE_LIMITS


def write_overflow_message(data, APR, file_IO4):
    ID = data.num_datasets
    IDEN = data.IDEN
    MTTP = IDEN[ID, 8]
    NSHP = APR.NSHP
    NR = APR.NR
    if data.IDEN[data.num_datasets, 7] != 6:
        if MTTP == 2:
            NSHP = NSHP + 1
            L = NR + NSHP
            if L > SIZE_LIMITS.MAX_NUM_UNKNOWNS:
                format701 = "( '   OVERFLOW OF UNKNOWN-VECTOR SPACE WITH SET  ',I3)"
                fort_write(file_IO4, format701, [NS])



def write_KAS_check(data, IPP, file_IO4):
    if data.IDEN[data.num_datasets, 7] != 6:
        #
        #   output of KAS for checking
        #
        if IPP[7] != 0:
            format702 = "(20I5)"
            NCT = data.NCT[data.num_datasets]
            for K in fort_range(1,NCT):
                NADD = data.num_datapoints + 1
                NALT = NADD - ID[IDEN, 1]
                fort_write(file_IO4, format702,
                        [data.KAS[NALT:NADD], K])



def write_datablock_header(file_IO4):
    format108 = "(/' DATABLOCK************************DATABLOCK**************" + \
                "******************************************DATABLOCK '/)"
    fort_write(file_IO4, format108, [])



def write_dataset_info(ID, data, LABL, file_IO4):

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
    tmp = [[LABL.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(1,NU)]
    fort_write(file_IO4, format139, [NS, LABL.TYPE[MT],tmp])
    if NCT2 > 0:
        format149 = "(2X,6(2X,2A8))"
        tmp = [[LABL.CLAB[NT[K],L] for L in fort_range(1,2)] for K in fort_range(NU1,NCT2)]
        fort_write(file_IO4, format149, tmp)

    #
    #       NAME ID AND REFERENCE I/O
    #
    format132 = "(/' YEAR',I5,' TAG',I3,' AUTHOR:  ',4A8,4A8/)"
    fort_write(None, format132, [IDEN[ID, 3:5], LABL.CLABL[1:5], LABL.BREF[1:5]])
    # VP      if(modrep .ne. 0) go to 183
    fort_write(file_IO4, format132, [IDEN[ID, 3:5], LABL.CLABL[1:5], LABL.BREF[1:5]])

    if not should_downweight(ID, data) and (data.MOD2 > 1 and data.MOD2 < 10):
        format339 = "('  WEIGHTING OPTION NOT IMPLEMENTED, DATA SET  ',I5/)"
        fort_write(file_IO4, format339, NS)



def write_prior_info(APR, IPP, LABL, file_IO4):
    # from here onwards only output
    NC = APR.NC
    NR = APR.NR
    # label .lbl30
    format134 = r"(//2X,36HCROSS SECTIONS OF PRESENT EVALUATION//)"
    fort_write(file_IO4, format134, [])
    format135 = "(10X,I3,5X,2A8)"
    for K in fort_range(1,NC):
        fort_write(file_IO4, format135, [K, LABL.CLAB[K, 1:3]])
    # label .lbl22
    if IPP[1] != 0:
        format136 = "(1H1//,2X,35HENERGIES AND APRIORI CROSS SECTIONS//)" 
        fort_write(file_IO4, format136, [])
        format137 = "(/ '     INDEX     E/MEV   ',7X,2A8 /)"
        for  K in fort_range(1,NC):  # .lbl24
            fort_write(file_IO4, format137, LABL.CLAB[K,1:3])
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


