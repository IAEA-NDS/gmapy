import numpy as np
from .basic_maps import get_sensmat_exact, propagate_exact
from .helperfuns import return_matrix



class CrossSectionShapeMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:2-R1:')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable):
        vals = self.__compute(priortable, exptable, 'propagate')
        return vals


    def jacobian(self, priortable, exptable, ret_mat=False):
        jac = self.__compute(priortable, exptable, 'jacobian')
        ret = return_matrix(jac['idcs1'], jac['idcs2'], jac['x'],
                dims = (len(exptable.index), len(priortable.index)),
                how = 'csr' if ret_mat else 'dic')
        return ret


    def __compute(self, priortable, exptable, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        vals = np.empty(0, dtype=float)
        concat = np.concatenate

        isresp = self.is_responsible(exptable)
        reacs = exptable.loc[isresp, 'REAC'].unique()
        for curreac in reacs:
            priortable_red = priortable[priortable['REAC'] == \
                    curreac.replace('MT:2','MT:1')]
            exptable_red = exptable[exptable['REAC'] == curreac]
            ens1 = priortable_red['ENERGY']
            vals1 = priortable_red['PRIOR']
            idcs1red = priortable_red.index
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'] == dataset_id]
                # get the respective normalization factor from prior
                mask = priortable['NODE'] == dataset_id.replace('exp_', 'norm_')
                norm_index = priortable[mask].index
                if (len(norm_index) != 1):
                    raise IndexError('There are ' + str(len(norm_index)) +
                        ' normalization factors in prior for dataset ' + str(dataset_id))
                norm_fact = np.asscalar(priortable.loc[norm_index, 'PRIOR'])
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index
                # calculate the sensitivity matrix
                if what == 'jacobian':
                    Sdic = get_sensmat_exact(ens1, ens2, idcs1red, idcs2red)
                    curcoeff = np.array(Sdic['x']) * norm_fact
                    # add the sensitivity to normalization factor in prior
                    numel = len(Sdic['idcs2'])
                    propvals = propagate_exact(ens1, vals1, ens2)
                    curidcs1 = concat([Sdic['idcs1'], np.full(numel, norm_index)])
                    curidcs2 = concat([Sdic['idcs2'], Sdic['idcs2']])
                    if len(curidcs1) != len(curidcs2):
                        raise ValueError
                    curcoeff = concat([curcoeff, propvals])
                    idcs1 = concat([idcs1, curidcs1])
                    idcs2 = concat([idcs2, curidcs2])
                    coeff = concat([coeff, curcoeff])
                elif what == 'propagate':
                    idcs2 = concat([idcs1, idcs2red])
                    propvals = propagate_exact(ens1, vals1, ens2)
                    vals = concat([vals, propvals * norm_fact])
                else:
                    raise ValueError('argument what must be "propagate" or "jacobian"')


        if what == 'jacobian':
            Sdic =  {'idcs1': np.array(idcs1, dtype=int),
                     'idcs2': np.array(idcs2, dtype=int),
                     'x': np.array(coeff, dtype=float)}
            return Sdic

        elif what == 'propagate':
            # bring the elements into order
            vals = np.array(vals)
            perm = np.argsort(idcs2)
            vals = vals[perm]
            return vals

        else:
            raise ValueError('what must be either "propagate" or "jacobian"')

