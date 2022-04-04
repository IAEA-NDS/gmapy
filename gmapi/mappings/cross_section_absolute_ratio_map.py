import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
        basic_multiply_Sdic_rows)
from .helperfuns import return_matrix



class CrossSectionAbsoluteRatioMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:7-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable, refvals):
        propdic = self.__compute(priortable, exptable, refvals, 'propagate')
        propvals = np.full(exptable.shape[0], 0.)
        propvals[propdic['idcs2']] = propdic['propvals']
        return propvals


    def jacobian(self, priortable, exptable, refvals, ret_mat=False):
        jac = self.__compute(priortable, exptable, refvals, 'jacobian')
        return return_matrix(jac['idcs1'], jac['idcs2'], jac['coeffs'],
                  dims = (exptable.shape[0], priortable.shape[0]),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, priortable, exptable, refvals, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = priortable['REAC'].str.match('MT:1-R1:')
        priortable = priortable[priormask]
        expmask = self.is_responsible(exptable)
        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            # obtian the involved reactions
            string_groups = curreac.split('-')
            reac1id = int(string_groups[1].split(':')[1])
            reac2id = int(string_groups[2].split(':')[1])
            reac3id = int(string_groups[3].split(':')[1])
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            reac3str = 'MT:1-R1:' + str(reac3id)
            if (reac1str == reac2str or
                reac2str == reac3str or
                reac3str == reac1str):
                   raise IndexError('all three reactions in a/(b+c) must be different')
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'] == reac1str]
            priortable_red2 = priortable[priortable['REAC'] == reac2str]
            priortable_red3 = priortable[priortable['REAC'] == reac3str]
            # and in the exptable
            exptable_red = exptable[exptable['REAC'] == curreac]
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_idcs3 = priortable_red3.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            src_en3 = priortable_red3['ENERGY']
            src_vals1 = refvals[priortable_red1.index]
            src_vals2 = refvals[priortable_red2.index]
            src_vals3 = refvals[priortable_red3.index]
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']
            # calculate the sensitivity matrix
            # d(a/(b+c)) = 1/(b+c)*da - a/(b+c)^2*db - a/(b+c)^2*dc
            propvals1 = basic_propagate(src_en1, src_vals1, tar_en)
            propvals2 = basic_propagate(src_en2, src_vals2, tar_en)
            propvals3 = basic_propagate(src_en3, src_vals3, tar_en)

            if what == 'jacobian':
                Sdic1 = get_basic_sensmat(src_en1, src_vals1, tar_en, ret_mat=False)
                Sdic1['idcs1'] = src_idcs1[Sdic1['idcs1']]
                Sdic1['idcs2'] = tar_idcs[Sdic1['idcs2']]

                Sdic2 = get_basic_sensmat(src_en2, src_vals2, tar_en, ret_mat=False)
                Sdic2['idcs1'] = src_idcs2[Sdic2['idcs1']]
                Sdic2['idcs2'] = tar_idcs[Sdic2['idcs2']]

                Sdic3 = get_basic_sensmat(src_en3, src_vals3, tar_en, ret_mat=False)
                Sdic3['idcs1'] = src_idcs3[Sdic3['idcs1']]
                Sdic3['idcs2'] = tar_idcs[Sdic3['idcs2']]

                basic_multiply_Sdic_rows(Sdic1, 1 / (propvals2+propvals3))
                tmp = (-propvals1 / np.square(propvals2 + propvals3))
                basic_multiply_Sdic_rows(Sdic2, tmp)
                basic_multiply_Sdic_rows(Sdic3, tmp)

                Sdic = {'idcs1': concat([Sdic1['idcs1'], Sdic2['idcs1'], Sdic3['idcs1']]),
                        'idcs2': concat([Sdic1['idcs2'], Sdic2['idcs2'], Sdic3['idcs2']]),
                        'x': concat([Sdic1['x'], Sdic2['x'], Sdic3['x']])}

                idcs1 = concat([idcs1, Sdic['idcs1']])
                idcs2 = concat([idcs2, Sdic['idcs2']])
                coeff = concat([coeff, Sdic['x']])

            elif what == 'propagate':
                idcs2 = concat([idcs2, tar_idcs])
                curvals = propvals1 / (propvals2+propvals3)
                propvals = concat([propvals, curvals])

        if what == 'jacobian':
            return {'idcs1': idcs1, 'idcs2': idcs2, 'coeffs': coeff}
        elif what == 'propagate':
            return {'idcs2': idcs2, 'propvals': propvals}

