import numpy as np
from .basic_maps import get_sensmat_exact, propagate_exact



class CrossSectionShapeMap:

    def is_responsible(self, exptable):
        expmask = exptable['REAC'].str.match('MT:2-R1:')
        return np.array(expmask, dtype=bool)


    def propagate(self, priortable, exptable):
        pass


    def jacobian(self, priortable, exptable):
        if not np.all(self.is_responsible(exptable)):
            raise TypeError('This handler can only map cross section ratios (MT=2)') 

        idcs1 = np.empty(0, dtype=int)
        idcs2 = np.empty(0, dtype=int)
        coeff = np.empty(0, dtype=float)
        concat = np.concatenate

        priormask = priortable['REAC'].str.match('MT:2-R1:')
        reacs = exptable['REAC'].unique()
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
                norm_fact = np.asscalar(priortable.loc[norm_index, 'PRIOR'])
                if (len(norm_index) != 1):
                    raise IndexError('More than one normalization in prior for dataset ' + str(dataset_id))
                # abbreviate some variables
                ens2 = exptable_ds['ENERGY']
                idcs2red = exptable_ds.index
                # calculate the sensitivity matrix
                Sdic = get_sensmat_exact(ens1, ens2, idcs1red, idcs2red)
                curcoeff = np.array(Sdic['x']) * norm_fact
                # add the sensitivity to normalization factor in prior
                numel = len(Sdic['idcs2'])
                propvals = propagate_exact(ens1, vals1, ens2)
                curidcs1 = concat([Sdic['idcs1'], np.full(numel, norm_index)])
                curidcs2 = concat([Sdic['idcs2'], Sdic['idcs2']])
                curcoeff = concat([curcoeff, propvals])
                if len(curidcs1) != len(curidcs2):
                    raise ValueError
                if len(curidcs1) != len(curcoeff):
                    raise ValueError

                idcs1 = concat([idcs1, curidcs1])
                idcs2 = concat([idcs2, curidcs2])
                coeff = concat([coeff, curcoeff])

            return {'idcs1': idcs1, 'idcs2': idcs2, 'x': coeff} 

