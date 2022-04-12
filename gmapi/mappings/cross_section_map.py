import numpy as np
from .basic_maps import basic_propagate, get_basic_sensmat
from .helperfuns import return_matrix



class CrossSectionMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:1-R1:') &
                   datatable['NODE'].str.match('exp_'))
        return np.array(expmask, dtype=bool)


    def propagate(self, datatable, refvals):
        propdic = self.__compute(datatable, refvals, 'propagate')
        propvals = np.full(datatable.shape[0], 0., dtype=float)
        propvals[propdic['idcs2']] = propdic['propvals']
        return propvals


    def jacobian(self, datatable, refvals, ret_mat=False):
        num_points = datatable.shape[0]
        Sdic = self.__compute(datatable, refvals, 'jacobian')
        return return_matrix(Sdic['idcs1'], Sdic['idcs2'], Sdic['coeffs'],
                  dims = (num_points, num_points),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, datatable, refvals, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = (datatable['REAC'].str.match('MT:1-R1:') &
                     datatable['NODE'].str.match('xsid_'))
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == curreac]
            exptable_red = exptable[exptable['REAC'] == curreac]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY']
            vals1 = refvals[priortable_red.index]
            idcs1red = priortable_red.index
            ens2 = exptable_red['ENERGY']
            idcs2red = exptable_red.index

            if what == 'jacobian':
                Sdic = get_basic_sensmat(ens1, vals1, ens2, ret_mat=False)
                Sdic['idcs1'] = idcs1red[Sdic['idcs1']]
                Sdic['idcs2'] = idcs2red[Sdic['idcs2']]
                idcs1 = concat([idcs1, Sdic['idcs1']])
                idcs2 = concat([idcs2, Sdic['idcs2']])
                coeff = concat([coeff, Sdic['x']])

            elif what == 'propagate':
                curvals = basic_propagate(ens1, vals1, ens2)
                idcs2 = concat([idcs2, idcs2red])
                propvals = concat([propvals, curvals])

        if what == 'jacobian':
            return {'idcs1': idcs1, 'idcs2': idcs2, 'coeffs': coeff}
        elif what == 'propagate':
            return {'idcs2': idcs2, 'propvals': propvals}

