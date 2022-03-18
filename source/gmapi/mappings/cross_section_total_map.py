import numpy as np
from .basic_maps import get_sensmat_exact, propagate_exact
from .helperfuns import return_matrix



class CrossSectionTotalMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+')
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
            reac_groups = curreac.split('-')[1:]
            reacids = [int(x.split(':')[1]) for x in reac_groups] 
            reacstrs = ['MT:1-R1:' + str(rid) for rid in reacids]
            if len(np.unique(reacstrs)) < len(reacstrs):
                   raise IndexError('Each reaction must occur only once in reaction string')
            # retrieve the relevant reactions in the prior
            priortable_reds = [priortable[priortable['REAC'] == r] for r in reacstrs]
            # retrieve relevant rows in exptable
            exptable_red = exptable[exptable['REAC'] == curreac]
            # some abbreviations
            src_idcs_list = [pt.index for pt in priortable_reds]
            src_en_list = [pt['ENERGY'] for pt in priortable_reds]
            src_vals_list = [refvals[pt.index] for pt in priortable_reds]
            tar_idcs = exptable_red.index
            tar_en = exptable_red['ENERGY']
            # calculate the sensitivity matrix
            # d(a1+a2+a3+...) = 1*da1 + 1*da2 + 1*da3 + ...
            curvals = np.full(len(tar_idcs), 0., dtype=float)
            for i in range(len(src_idcs_list)):
                cur_src_en = src_en_list[i]
                cur_src_vals = src_vals_list[i]
                curvals += propagate_exact(cur_src_en, cur_src_vals, tar_en)

            if what == 'jacobian':
                Sdic = {'idcs1': np.empty(0, dtype=int),
                        'idcs2': np.empty(0, dtype=int),
                        'x': np.empty(0, dtype=float)}
                for i in range(len(src_en_list)):
                    cur_src_idcs = src_idcs_list[i]
                    cur_src_en = src_en_list[i]
                    cur_src_vals = src_vals_list[i]
                    curSdic = get_sensmat_exact(cur_src_en, tar_en,
                                                cur_src_idcs, tar_idcs)
                    Sdic['idcs1'] = concat([Sdic['idcs1'], curSdic['idcs1']])
                    Sdic['idcs2'] = concat([Sdic['idcs2'], curSdic['idcs2']])
                    Sdic['x'] = concat([Sdic['x'], curSdic['x']])

                idcs1 = concat([idcs1, Sdic['idcs1']])
                idcs2 = concat([idcs2, Sdic['idcs2']])
                coeff = concat([coeff, Sdic['x']])

            elif what == 'propagate':
                idcs2 = concat([idcs2, tar_idcs])
                propvals = concat([propvals, curvals])

        if what == 'jacobian':
            return {'idcs1': idcs1, 'idcs2': idcs2, 'coeffs': coeff}
        elif what == 'propagate':
            return {'idcs2': idcs2, 'propvals': propvals}

