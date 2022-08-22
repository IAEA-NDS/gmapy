import numpy as np
from .basic_maps import (basic_propagate, get_basic_sensmat,
        basic_multiply_Sdic_rows)
from .helperfuns import return_matrix



class CrossSectionShapeMap:

    def is_responsible(self, datatable):
        expmask = (datatable['REAC'].str.match('MT:2-R1:', na=False) &
                   datatable['NODE'].str.match('exp_', na=False))
        return np.array(expmask, dtype=bool)


    def propagate(self, datatable, refvals):
        vals = self.__compute(datatable, refvals, 'propagate')
        return vals


    def jacobian(self, datatable, refvals, ret_mat=False):
        jac = self.__compute(datatable, refvals, 'jacobian')
        ret = return_matrix(jac['idcs1'], jac['idcs2'], jac['x'],
                dims = (len(datatable.index), len(datatable.index)),
                how = 'csr' if ret_mat else 'dic')
        return ret


    def __compute(self, datatable, refvals, what):
        num_points = datatable.shape[0]

        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        vals = np.empty(0, dtype=float)
        concat = np.concatenate

        isresp = self.is_responsible(datatable)
        reacs = datatable.loc[isresp, 'REAC'].unique()
        for curreac in reacs:
            priormask = ((datatable['REAC'].str.fullmatch(curreac.replace('MT:2','MT:1'), na=False)) &
                         datatable['NODE'].str.match('xsid_', na=False))
            priortable_red = datatable[priormask]
            exptable_red = datatable[(datatable['REAC'].str.fullmatch(curreac, na=False) &
                                      datatable['NODE'].str.match('exp_'))]
            ens1 = priortable_red['ENERGY']
            vals1 = refvals[priortable_red.index]
            idcs1red = priortable_red.index
            # loop over the datasets
            dataset_ids = exptable_red['NODE'].unique()
            for dataset_id in dataset_ids:
                exptable_ds = exptable_red[exptable_red['NODE'].str.fullmatch(dataset_id, na=False)]
                # get the respective normalization factor from prior
                mask = datatable['NODE'].str.fullmatch(dataset_id.replace('exp_', 'norm_'), na=False)
                norm_index = datatable[mask].index
                if (len(norm_index) != 1):
                    raise IndexError('There are ' + str(len(norm_index)) +
                        ' normalization factors in prior for dataset ' + str(dataset_id))
                norm_fact = refvals[norm_index]
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index
                # calculate the sensitivity matrix
                if what == 'jacobian':
                    Sdic = get_basic_sensmat(ens1, vals1, ens2, ret_mat=False)
                    Sdic['idcs1'] = idcs1red[Sdic['idcs1']]
                    Sdic['idcs2'] = idcs2red[Sdic['idcs2']]
                    basic_multiply_Sdic_rows(Sdic, norm_fact)

                    curcoeff = np.array(Sdic['x']) * norm_fact
                    # add the sensitivity to normalization factor in prior
                    propvals = basic_propagate(ens1, vals1, ens2)
                    curidcs1 = concat([Sdic['idcs1'], np.full(len(idcs2red), norm_index)])
                    curidcs2 = concat([Sdic['idcs2'], idcs2red])
                    curcoeff = concat([Sdic['x'], propvals])

                    idcs1 = concat([idcs1, curidcs1])
                    idcs2 = concat([idcs2, curidcs2])
                    coeff = concat([coeff, curcoeff])

                elif what == 'propagate':
                    idcs2 = concat([idcs2, idcs2red])
                    propvals = basic_propagate(ens1, vals1, ens2)
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
            allvals = np.full(num_points, 0.)
            allvals[idcs2] = vals
            return allvals

        else:
            raise ValueError('what must be either "propagate" or "jacobian"')

