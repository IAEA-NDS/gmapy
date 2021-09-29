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

