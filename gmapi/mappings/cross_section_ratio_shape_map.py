import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
        basic_multiply_Sdic_rows)
from .helperfuns import return_matrix



class CrossSectionRatioShapeMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:4-R1:[0-9]+-R2:[0-9]+', na=False) &
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
        priormask = np.logical_or(priormask, datatable['NODE'].str.match('norm_', na=False))
        priortable = datatable[priormask]
        expmask = self.is_responsible(datatable)
        exptable = datatable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            # obtian the involved reactions
            string_groups = curreac.split('-')
            reac1id = int(string_groups[1].split(':')[1])
            reac2id = int(string_groups[2].split(':')[1])
            reac1str = 'MT:1-R1:' + str(reac1id)
            reac2str = 'MT:1-R1:' + str(reac2id)
            # retrieve the relevant reactions in the prior
            priortable_red1 = priortable[priortable['REAC'].str.fullmatch(reac1str, na=False)]
            priortable_red2 = priortable[priortable['REAC'].str.fullmatch(reac2str, na=False)]
            # some abbreviations
            src_idcs1 = priortable_red1.index
            src_idcs2 = priortable_red2.index
            src_en1 = priortable_red1['ENERGY']
            src_en2 = priortable_red2['ENERGY']
            src_vals1 = refvals[priortable_red1.index]
            src_vals2 = refvals[priortable_red2.index]
            # cycle over the datasets as each of those has different normalization constant
            exptable_red = exptable[exptable['REAC'].str.fullmatch(curreac, na=False)]
            datasets = exptable_red['NODE'].unique()
            for ds in datasets:
                tar_idcs = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)].index
                tar_en = exptable_red[exptable_red['NODE'].str.fullmatch(ds, na=False)]['ENERGY']
                # obtain normalization and position in priortable
                normstr = ds.replace('exp_', 'norm_')
                norm_index = priortable[priortable['NODE'].str.fullmatch(normstr, na=False)].index
                if len(norm_index) != 1:
                    raise IndexError('Exactly one normalization factor must be present for a dataset')
                norm_fact = refvals[norm_index]
                # calculate the sensitivity matrix for cross section ratio shape
                # d(c*a/b) = c/b*da - c*a/b^2*db + a/b*dc
                propvals1 = basic_propagate(src_en1, src_vals1, tar_en)
                propvals2 = basic_propagate(src_en2, src_vals2, tar_en)

                if what == 'jacobian':
                    Sdic1 = get_basic_sensmat(src_en1, src_vals1, tar_en, ret_mat=False)
                    Sdic1['idcs1'] = src_idcs1[Sdic1['idcs1']]
                    Sdic1['idcs2'] = tar_idcs[Sdic1['idcs2']]

                    Sdic2 = get_basic_sensmat(src_en2, src_vals2, tar_en, ret_mat=False)
                    Sdic2['idcs1'] = src_idcs2[Sdic2['idcs1']]
                    Sdic2['idcs2'] = tar_idcs[Sdic2['idcs2']]

                    basic_multiply_Sdic_rows(Sdic1, 1 / propvals2 * norm_fact)
                    basic_multiply_Sdic_rows(Sdic2, (-propvals1 / np.square(propvals2)) * norm_fact)

                    Sdic = {'idcs1': concat([Sdic1['idcs1'], Sdic2['idcs1']]),
                            'idcs2': concat([Sdic1['idcs2'], Sdic2['idcs2']]),
                            'x': concat([Sdic1['x'], Sdic2['x']])}
                    # add the normalization sensitivity
                    src_norm_idcs = np.full(len(tar_idcs), norm_index)
                    norm_coeffs = propvals1 / propvals2
                    Sdic['idcs1'] = concat([Sdic['idcs1'], src_norm_idcs])
                    Sdic['idcs2'] = concat([Sdic['idcs2'], tar_idcs])
                    Sdic['x'] = concat([Sdic['x'], norm_coeffs])

                    # add everything to global triple list
                    idcs1 = concat([idcs1, Sdic['idcs1']])
                    idcs2 = concat([idcs2, Sdic['idcs2']])
                    coeff = concat([coeff, Sdic['x']])

                elif what == 'propagate':
                    idcs2 = concat([idcs2, tar_idcs])
                    curvals = propvals1 / propvals2 * norm_fact
                    propvals = concat([propvals, curvals])

        if what == 'jacobian':
            return {'idcs1': idcs1, 'idcs2': idcs2, 'coeffs': coeff}
        elif what == 'propagate':
            return {'idcs2': idcs2, 'propvals': propvals}

