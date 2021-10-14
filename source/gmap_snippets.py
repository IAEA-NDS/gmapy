import numpy as np


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

