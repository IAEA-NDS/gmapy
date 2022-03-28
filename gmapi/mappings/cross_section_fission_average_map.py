import numpy as np
from .basic_maps import (propagate_fisavg, get_sensmat_fisavg,
        get_sensmat_fisavg_corrected)
from .helperfuns import return_matrix



class CrossSectionFissionAverageMap:

    def __init__(self, fix_jacobian=True):
        self._fix_jacobian = fix_jacobian


    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:6-R1:')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable, refvals):
        preds = np.full(len(exptable), 0., dtype=float) 
        mapdic = self.__compute(priortable, exptable, refvals, what='propagate')
        preds[mapdic['idcs2']] = mapdic['propvals'] 
        return preds


    def jacobian(self, priortable, exptable, refvals, ret_mat=False):
        num_exp_points = exptable.shape[0]
        num_prior_points = priortable.shape[0]
        Sdic = self.__compute(priortable, exptable, refvals, what='jacobian')
        return return_matrix(Sdic['idcs1'], Sdic['idcs2'], Sdic['coeff'],
                  dims = (num_exp_points, num_prior_points),
                  how = 'csr' if ret_mat else 'dic')


    def __compute(self, priortable, exptable, refvals, what):
        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        propvals = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = priortable['REAC'].str.match('MT:1-R1:')
        priormask = np.logical_or(priormask, priortable['NODE'] == 'fis')
        priortable = priortable[priormask]

        expmask = self.is_responsible(exptable)
        exptable = exptable[expmask]
        expids = exptable['NODE'].unique()

        # retrieve fission spectrum
        fistable = priortable[priortable['NODE']=='fis']
        ensfis = fistable['ENERGY'].to_numpy()
        valsfis = fistable['PRIOR'].to_numpy()

        for curexp in expids:
            exptable_red = exptable[exptable['NODE'] == curexp]
            if len(exptable_red) != 1:
                raise IndexError('None or more than one rows associated with a ' +
                        'fission average, which must not happen!')
            curreac = exptable_red['REAC'].values[0]
            preac = curreac.replace('MT:6-', 'MT:1-')
            priortable_red = priortable[priortable['REAC'] == preac]
            # abbreviate some variables
            ens1 = priortable_red['ENERGY'].to_numpy()
            vals1 = refvals[priortable_red.index]
            idcs1red = priortable_red.index
            idcs2red = exptable_red.index

            if what == 'propagate':
                curval = propagate_fisavg(ens1, vals1, ensfis, valsfis)
                idcs2 = concat([idcs2, idcs2red])
                propvals = concat([propvals, [curval]])
            elif what == 'jacobian':
                if self._fix_jacobian:
                    sensvec = get_sensmat_fisavg_corrected(ens1, vals1, ensfis, valsfis)
                else:
                    sensvec = get_sensmat_fisavg(ens1, vals1, ensfis, valsfis)
                idcs1 = concat([idcs1, idcs1red])
                idcs2 = concat([idcs2, np.full(len(idcs1red), idcs2red, dtype=int)])
                coeff = concat([coeff, sensvec])
            else:
                raise ValueError('what must be either "propagate" or "jacobian"')

        retdic = {}
        if what == 'jacobian':
            retdic['idcs1'] = idcs1
            retdic['idcs2'] = idcs2
            retdic['coeff'] = coeff
        elif what == 'propagate':
            retdic['idcs2'] = idcs2
            retdic['propvals'] = propvals

        return retdic

