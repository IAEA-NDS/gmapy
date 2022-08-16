import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
        basic_multiply_Sdic_rows)
from .helperfuns import return_matrix



class CrossSectionTotalMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:5(-R[0-9]+:[0-9]+)+', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)


    def propagate(self, datatable, refvals):
        propdic = self.__compute(datatable, refvals, 'propagate')
        propvals = np.full(datatable.shape[0], 0.)
        propvals[propdic['idcs2']] = propdic['propvals']
        return propvals


    def jacobian(self, datatable, refvals, ret_mat=False):
        jac = self.__compute(datatable, refvals, 'jacobian')
        return return_matrix(jac['idcs1'], jac['idcs2'], jac['coeffs'],
                  dims = (datatable.shape[0], datatable.shape[0]),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, datatable, refvals, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = (datatable['REAC'].str.match('MT:1-R1:', na=False) &
                     datatable['NODE'].str.match('xsid_', na=False))
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            # obtian the involved reactions
            reac_groups = curreac.split('-')[1:]
            reacids = [int(x.split(':')[1]) for x in reac_groups] 
            reacstrs = ['MT:1-R1:' + str(rid) for rid in reacids]
            if len(np.unique(reacstrs)) < len(reacstrs):
                   raise IndexError('Each reaction must occur only once in reaction string')
            # retrieve the relevant reactions in the prior
            priortable_reds = [priortable[priortable['REAC'].str.fullmatch(r, na=False)] for r in reacstrs]
            # retrieve relevant rows in exptable
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
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
                curvals += basic_propagate(cur_src_en, cur_src_vals, tar_en)

            if what == 'jacobian':
                Sdic = {'idcs1': np.empty(0, dtype=int),
                        'idcs2': np.empty(0, dtype=int),
                        'x': np.empty(0, dtype=float)}
                for i in range(len(src_en_list)):
                    cur_src_idcs = src_idcs_list[i]
                    cur_src_en = src_en_list[i]
                    cur_src_vals = src_vals_list[i]
                    curSdic = get_basic_sensmat(cur_src_en, cur_src_vals,
                                                tar_en, ret_mat=False)
                    curSdic['idcs1'] = cur_src_idcs[curSdic['idcs1']]
                    curSdic['idcs2'] = tar_idcs[curSdic['idcs2']]

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

