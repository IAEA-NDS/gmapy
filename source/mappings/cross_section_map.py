import numpy as np
from .basic_maps import get_sensmat_exact



class CrossSectionMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:1-R1:')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable):
        pass


    def jacobian(self, priortable, exptable):
        if not np.all(self.is_responsible(exptable)):
            raise TypeError('This handler can only map cross sections (MT=1)') 
        priormask = priortable['REAC'].str.match('MT:1-R1:')
        reacs = exptable['REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == curreac]
            exptable_red = exptable[exptable['REAC'] == curreac]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            vals1 = priortable_red['PRIOR']
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index
            # calculate the sensitivity matrix
            Sdic = get_sensmat_exact(ens1, ens2, idcs1red, idcs2red)
            return Sdic

