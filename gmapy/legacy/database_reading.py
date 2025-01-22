# helpful for fortran to python conversion
import numpy as np

from .fortran_utils import fort_range, fort_read, fort_write
from .data_management import init_gauss, init_prior, init_labels
from .gmap_snippets import TextfileReader
from .database_reading_utils import (read_prior, read_datablock,
        read_fission_spectrum)



def read_gma_database(dbfile, format_dic={}):

    # IMPLICIT definitions in original version
    # IMPLICIT REAL*8 (A-H,O-Z)

    # NOTE: This is part of a hack to reproduce a
    #       bug in the Fortran version of GMAP 
    #       with values in ENFF leaking into
    #       the next datablock
    data = None

    #
    #      IPP        i/o choices
    #      NRED       data set Nr.s for downweighting
    #      NELIM      data set Nr.s to exclude from evaluation
    #
    #   INTEGER*2 KAS(250,5),NT(5),IDEN(30,8),NSETN(200),IPP(8),
    #  1 NENF(40,10),NETG(11,40),NCSST(10),NEC(2,10,10),NRED(160)
    #  2 ,KA(1200,250),NELIM(40)
    IPP = np.zeros(8+1, dtype=int)
    NRED = np.zeros(160+1, dtype=int)
    NELIM = np.zeros(40+1, dtype=int)

    basedir = '.'
    file_IO3 = TextfileReader(dbfile)

    APR = init_prior()
    gauss = init_gauss()
    LABL = init_labels()

    # fisdata is optional
    fisdata = None

    datablock_list = []

    #
    #      INITIALIZE PARAMETERS
    #
    #
    #      DATA DE/1200*0.D0/,BM/1200*0.D0/,B/720600*0.D0/,NRED/160*0/
    #
    MOD2 = 0
    MODC = 3
    AMO3 = 0.0
    IELIM = 0

    MPPP = 1
    MODAP = 3
    IPP = [None, 1, 1, 1, 0, 0, 1, 0, 1]

    #
    #      CONTROL
    #
    #      AKON     CONTROL WORD  A4
    #      MC1-8    PARAMETERS   8I5
    #

    while True:

        #
        #      Control parameter input
        #
        format100 = "(A4,1X,8I5)"
        line_tuple = fort_read(file_IO3,  format100)
        if len(line_tuple) == 0:
            ACON = LABL.AKON[3] # if EOF, use END indicator
        else:
            ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8 = line_tuple

        # LABL.AKON[9] == 'MODE'
        if ACON == LABL.AKON[9]:
            #
            #   MODE DEFINITION
            #
            MODC = MC1
            MOD2 = MC2
            #VPBEG MPPP=1 allows to use anti-PPP option, when errors of exp data 
            #VP   are taken as % uncertainties from true (posterior) evaluation 
            MPPP = MC5
            #VPEND
            AMO3=MC3/10.
            MODAP=MC4
            if MC2 == 10:
                #
                #      test option:  input of data set numbers which are to be downweighted
                #
                K1=1
                K2=16
                format677 = "(16I5)"
                for K in fort_range(1,10):  # .lbl678
                    fort_write(file_IO3, format677, [NRED[K1:(K2+1)]])
                    if NRED[K2] == 0:
                        break
                    K1=K1+16
                    K2=K2+16

                for K in fort_range(K1,K2):  # .lbl680
                    if NRED[K] == 0:
                        break

                NELI=K-1


        # LABL.AKON[5] == 'I/OC'
        elif ACON == LABL.AKON[5]:
            #
            #      I/O CONTROL
            #
            IPP = [None for i in range(9)]
            IPP[1] = MC1
            IPP[2] = MC2
            IPP[3] = MC3
            IPP[4] = MC4
            IPP[5] = MC5
            IPP[6] = MC6
            IPP[7] = MC7
            IPP[8] = MC8


        # LABL.AKON[1] == 'APRI'
        elif ACON == LABL.AKON[1]:
            # INPUT OF CROSS SECTIONS TO BE EVALUATED,ENERGY GRID AND APRIORI CS
            read_prior(MC1, MC2, APR, file_IO3)


        # LABL.AKON[8] == 'FIS*'
        elif ACON == LABL.AKON[8]:
            fisdata = read_fission_spectrum(MC1, file_IO3)


        # LABL.AKON[4] == 'BLCK'
        elif ACON == LABL.AKON[4]:

            file_IO3.seek(file_IO3.get_line_nr()-1)

            data = read_datablock(MODC, MOD2, AMO3,
                           IELIM, NELIM, LABL, file_IO3, format_dic=format_dic)

            datablock_list.append(data)


        # LABL.AKON[3] == 'END*'
        elif ACON == LABL.AKON[3]:

            NTOT = 0
            for datablock in datablock_list:
                NTOT += datablock.num_datapoints
                datablock.NTOT = NTOT

            # add fisdata to APR because it has to be
            # seen as a priori knowledge
            APR.fisdata = fisdata

            return({
                'APR': APR,
                'datablock_list': datablock_list,
                'MPPP': MPPP,
                'IPP': IPP,
                'MODAP': MODAP
                })


        # LABL.AKON[11] == 'ELIM'
        elif ACON == LABL.AKON[11]:
            # input:  data set numbers which are to be excluded from the evaluation
            IELIM = MC1
            format171 = r"(16I5)"
            NELIM = fort_read(file_IO3, format171)


        # LABL.AKON[10] == 'STOP'
        elif ACON == LABL.AKON[10]:
            #
            #      test option:  forced stop for testing purpose
            #
            format107 = "( '  REQUESTED STOP ' )"
            fort_write(file_IO4, format107)
            exit()


        elif ACON == LABL.AKON[6]:
            format104 = "(A4,2X,'  CONTROL CODE UNKNOWN')"
            fort_write(file_IO4, format104, [ACON])
            exit()

