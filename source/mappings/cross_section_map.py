import numpy as np
from .basic_maps import get_sensmat_exact
from .helperfuns import return_matrix



class CrossSectionMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:1-R1:')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable):
        vals = priortable['PRIOR']
        S = self.jacobian(priortable, exptable, ret_mat=True)
        return np.array(S @ vals)


    def jacobian(self, priortable, exptable, ret_mat=False):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = priortable['REAC'].str.match('MT:1-R1:')
        priortable = priortable[priormask]
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
            idcs1 = concat([idcs1, Sdic['idcs1']])
            idcs2 = concat([idcs2, Sdic['idcs2']])
            coeff = concat([coeff, Sdic['x']])

        return return_matrix(idcs1, idcs2, coeff,
                  dims = (len(exptable.index), len(priortable.index)),
                  how = 'csr' if ret_mat else 'dic')

