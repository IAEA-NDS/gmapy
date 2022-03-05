import numpy as np
import pandas as pd

from .helperfuns import return_matrix
from .cross_section_map import CrossSectionMap
from .cross_section_shape_map import CrossSectionShapeMap
from .cross_section_ratio_map import CrossSectionRatioMap
from .cross_section_ratio_shape_map import CrossSectionRatioShapeMap
from .cross_section_absolute_ratio_map import CrossSectionAbsoluteRatioMap
from .cross_section_shape_of_ratio_map import CrossSectionShapeOfRatioMap
from .cross_section_total_map import CrossSectionTotalMap
from .cross_section_shape_of_sum_map import CrossSectionShapeOfSumMap
from .cross_section_fission_average_map import CrossSectionFissionAverageMap



class CompoundMap:

    maplist = [CrossSectionMap(),
               CrossSectionShapeMap(),
               CrossSectionRatioMap(),
               CrossSectionRatioShapeMap(),
               CrossSectionAbsoluteRatioMap(),
               CrossSectionShapeOfRatioMap(),
               CrossSectionTotalMap(),
               CrossSectionShapeOfSumMap(),
               CrossSectionFissionAverageMap()]


    def is_responsible(self, exptable):
        resp = np.full(len(exptable.index), False, dtype=bool)
        for curmap in self.maplist:
            curresp = curmap.is_responsible(exptable)
            if np.any(np.logical_and(resp, curresp)):
                raise ValueError('Several maps claim responsibility')
            resp = np.logical_or(resp, curresp)
        return resp


    def propagate(self, priortable, exptable, refvals):
        if not np.all(self.is_responsible(exptable)):
            raise TypeError('No known link from prior to some experimental data points')

        treated = np.full(exptable.shape[0], False)
        propvals = np.full(exptable.shape[0], 0.)

        exptable = exptable.sort_index()
        for curmap in self.maplist:
            curresp = curmap.is_responsible(exptable)
            if np.any(np.logical_and(treated, curresp)):
                raise ValueError('Several maps claim responsibility for the same rows')
            treated[curresp] = True
            curvals = curmap.propagate(priortable, exptable, refvals)
            if np.any(np.logical_and(propvals!=0., curvals!=0.)):
                raise ValueError('Several maps contribute to same experimental datapoint')
            propvals += curvals

        return propvals


    def jacobian(self, priortable, exptable, refvals, ret_mat=False):
        if not np.all(self.is_responsible(exptable)):
            raise TypeError('No known link from prior to some experimental data points')

        concat = np.concatenate
        Sdic = {'idcs1': np.empty(0, dtype=int),
                'idcs2': np.empty(0, dtype=int),
                'x': np.empty(0, dtype=float)}
        for curmap in self.maplist:
            curSdic = curmap.jacobian(priortable, exptable, refvals)

            if len(curSdic['idcs1']) != len(curSdic['idcs2']):
                raise ValueError('Lengths of idcs1 and idcs2 not equal')
            if len(curSdic['idcs1']) != len(curSdic['x']):
                raise ValueError('Lengths of idcs1 and x not equal')

            Sdic['idcs1'] = concat([Sdic['idcs1'], curSdic['idcs1']])
            Sdic['idcs2'] = concat([Sdic['idcs2'], curSdic['idcs2']])
            Sdic['x'] = concat([Sdic['x'], curSdic['x']])

        return return_matrix(Sdic['idcs1'], Sdic['idcs2'], Sdic['x'],
                dims = (exptable.shape[0], priortable.shape[0]),
                how = 'csr' if ret_mat else 'dic')

