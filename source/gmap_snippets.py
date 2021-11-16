import numpy as np
from fortran_utils import fort_range


class TextfileReader:

    def __init__(self, filename):
        self.name = filename
        self.fcont = open(filename, 'r').read().splitlines()
        self.linecnt = 0


    def readline(self):
        self.linecnt += 1
        if self.linecnt > len(self.fcont):
            raise IndexError('Attempting to read beyond EOF')
        return self.fcont[self.linecnt-1]


    def seek(self, linenr):
        self.linecnt = linenr


    def get_line_nr(self):
        return self.linecnt



def should_downweight(ID, data):
    MOD2 = data.MOD2
    AMO3 = data.AMO3
    IDEN = data.IDEN
    # 1st if line: downweighting based on year of measurement
    # 2nd if line: downweighting of specified data sets
    # 3rd+ if line: downweighting data sets with tags .NE. 1
    if (MOD2 > 1000 and IDEN[ID, 3] < MOD2) or \
       (MOD2 == 10 and IDEN[ID, 6] in NRED[1:(NELI+1)]) or \
       (MOD2 == 1 and IDEN[ID, 4] != 1) or \
       (MOD2 < 0 and IDEN[ID, 4] != 1):
        return True
    else:
        return False



def get_AX(ID, K, data, APR):
    IDEN = data.IDEN
    MT = IDEN[ID, 7]
    NCT = data.NCT[ID]
    KAS = data.KAS
    KX = KAS[K, 1]
    KY=KAS[K,2]
    KZ=KAS[K, 3]
    AX = APR.CS[KX]
    if MT == 4 or MT == 3:
        AX = AX / APR.CS[KY]
    if MT == 8 or MT == 5:
        AX = AX + APR.CS[KY]
    if MT == 5 and NCT == 3:
        AX = AX + APR.CS[KZ]
    if MT == 8 and NCT == 3:
        AX = AX + APR.CS[KZ]
    if MT == 9 or MT == 7:
        AX = AX/(APR.CS[KY]+APR.CS[KZ])

    return AX



def get_prior_range(xsid, APR):
    JA = APR.MCS[xsid, 2]
    JE = APR.MCS[xsid, 3]
    return (JA, JE)



def get_dataset_range(ID, data):
    start_index = 1
    if ID > 1:
        for k in fort_range(1, ID-1):
            num_points = data.IDEN[k, 1]
            start_index += num_points

    num_points = data.IDEN[ID, 1]
    end_index = start_index + num_points - 1
    return (start_index, end_index)



def get_dataset_id_from_idx(idx, data):
    if data.num_datasets == 0:
        raise IndexError('no dataset in datablock present')

    start_index = 1
    found_idx = -1
    for curID in fort_range(1, data.num_datasets):
        num_points = data.IDEN[curID, 1]
        start_index += num_points
        if start_index > idx:
            found_idx = curID
            break

    if found_idx == -1:
        raise IndexError('provided index outside range of dataset indices')

    return found_idx



def get_num_shapedatasets(data, end_idx=None):
    if end_idx is None:
        end_idx = data.num_datasets

    num = 0
    for ID in fort_range(1, end_idx):
        if data.IDEN[ID,8] == 2:
            num += 1
    return num

