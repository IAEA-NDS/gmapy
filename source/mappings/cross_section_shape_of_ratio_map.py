import numpy as np
from .basic_maps import get_sensmat_exact, propagate_exact
from .helperfuns import return_matrix



class CrossSectionShapeOfRatioMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:9-R1:[0-9]+-R2:[0-9]+-R3:[0-9]+')
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
        priormask = np.logical_or(priormask, priortable['NODE'].str.match('norm_'))
        priortable = priortable[priormask]
        expmask = self.is_responsible(exptable)
        exptable = exptable[expmask]
        reacs = exptable['REAC'].unique()

        for curreac in reacs:
            # obtain the involved reactions
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

            exptable_red = exptable[exptable['REAC'] == curreac]
            datasets = exptable_red['NODE'].unique()
            for ds in datasets:
                # subset another time exptable to get dataset info
                tar_idcs = exptable_red[exptable_red['NODE']==ds].index
                tar_en = exptable_red[exptable_red['NODE']==ds]['ENERGY']
                # obtain normalization and position in priortable
                normstr = ds.replace('exp_', 'norm_')
                norm_index = priortable[priortable['NODE']==normstr].index
                if len(norm_index) != 1:
                    raise IndexError('Exactly one normalization factor must be present for a dataset')
                norm_fact = refvals[norm_index]
                # calculate the sensitivity matrix
                # d(a/(b+c)) = 1/(b+c)*da - a/(b+c)^2*db - a/(b+c)^2*dc
                propvals1 = propagate_exact(src_en1, src_vals1, tar_en)
                propvals2 = propagate_exact(src_en2, src_vals2, tar_en)
                propvals3 = propagate_exact(src_en3, src_vals3, tar_en)

                if what == 'jacobian':
                    Sdic1 = get_sensmat_exact(src_en1, tar_en, src_idcs1, tar_idcs)
                    Sdic2 = get_sensmat_exact(src_en2, tar_en, src_idcs2, tar_idcs) 
                    Sdic3 = get_sensmat_exact(src_en3, tar_en, src_idcs3, tar_idcs) 
                    Sdic1['x'] = 1 / (propvals2+propvals3) * norm_fact
                    Sdic2['x'] = (-propvals1 / np.square(propvals2+propvals3)) * norm_fact
                    Sdic3['x'] = (-propvals1 / np.square(propvals2+propvals3)) * norm_fact

                    Sdic = {'idcs1': concat([Sdic1['idcs1'], Sdic2['idcs1'], Sdic3['idcs1']]),
                            'idcs2': concat([Sdic1['idcs2'], Sdic2['idcs2'], Sdic3['idcs2']]),
                            'x': concat([Sdic1['x'], Sdic2['x'], Sdic3['x']])}
                    # add the normalization sensitivity
                    src_norm_idcs = np.full(len(tar_idcs), norm_index)
                    norm_coeffs = propvals1 / (propvals2+propvals3)
                    Sdic['idcs1'] = concat([Sdic['idcs1'], src_norm_idcs])
                    Sdic['idcs2'] = concat([Sdic['idcs2'], tar_idcs])
                    Sdic['x'] = concat([Sdic['x'], norm_coeffs])
                    
                    # add everything to global triple list 
                    idcs1 = concat([idcs1, Sdic['idcs1']])
                    idcs2 = concat([idcs2, Sdic['idcs2']])
                    coeff = concat([coeff, Sdic['x']])

                elif what == 'propagate':
                    idcs2 = concat([idcs2, tar_idcs])
                    curvals = propvals1 / (propvals2+propvals3) * norm_fact
                    propvals = concat([propvals, curvals])

        if what == 'jacobian':
            return {'idcs1': idcs1, 'idcs2': idcs2, 'coeffs': coeff}
        elif what == 'propagate':
            return {'idcs2': idcs2, 'propvals': propvals}
