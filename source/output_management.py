from fortran_utils import fort_range, fort_write
from gmap_snippets import should_downweight

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

