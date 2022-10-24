import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

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
from .cross_section_ratio_of_sacs_map import CrossSectionRatioOfSacsMap



class CompoundMap:

    def __init__(self, fix_sacs_jacobian=True, legacy_integration=False):
        self.maplist = [
                CrossSectionMap(),
                CrossSectionShapeMap(),
                CrossSectionRatioMap(),
                CrossSectionRatioShapeMap(),
                CrossSectionAbsoluteRatioMap(),
                CrossSectionShapeOfRatioMap(),
                CrossSectionTotalMap(),
                CrossSectionShapeOfSumMap(),
                CrossSectionFissionAverageMap(fix_jacobian=fix_sacs_jacobian,
                                              legacy_integration=legacy_integration),
                CrossSectionRatioOfSacsMap(atol=1e-5, rtol=1e-05, maxord=16)
            ]

    def is_responsible(self, datatable):
        resp = np.full(len(datatable.index), False, dtype=bool)
        for curmap in self.maplist:
            curresp = curmap.is_responsible(datatable)
            resp_overlap = np.logical_and(resp, curresp)
            if np.any(resp_overlap):
                print(datatable[resp_overlap])
                raise ValueError(f'Several maps claim responsibility ({str(curmap)})')
            resp = np.logical_or(resp, curresp)
        return resp


    def propagate(self, datatable, refvals):
        datatable = datatable.sort_index()
        isresp = self.is_responsible(datatable)
        isexp = datatable['NODE'].str.match('exp_')
        not_isresp = np.logical_not(isresp)
        if not np.all(isresp[isexp]):
            raise TypeError('No known link from prior to some experimental data points')

        treated = np.full(datatable.shape[0], False)
        propvals = np.full(datatable.shape[0], 0.)
        propvals[not_isresp] = refvals[not_isresp]

        for curmap in self.maplist:
            curresp = curmap.is_responsible(datatable)
            if np.any(np.logical_and(treated, curresp)):
                raise ValueError('Several maps claim responsibility for the same rows')
            treated[curresp] = True
            curvals = curmap.propagate(datatable, refvals)
            if np.any(np.logical_and(propvals!=0., curvals!=0.)):
                raise ValueError('Several maps contribute to same experimental datapoint')
            propvals += curvals

        return propvals


    def jacobian(self, datatable, refvals, ret_mat=False):
        isexp = datatable['NODE'].str.match('exp_')
        isresp = self.is_responsible(datatable)
        if not np.all(isresp[isexp]):
            raise TypeError('No known link from prior to some experimental data points')

        numel = len(datatable)
        idcs = np.arange(len(datatable))

        ones = np.full(len(datatable), 1., dtype=float)
        idmat = csr_matrix((ones, (idcs, idcs)),
                           shape=(numel, numel), dtype=float)
        compSmat = idmat
        concat = np.concatenate
        for curmap in self.maplist:
            curSmat = curmap.jacobian(datatable, refvals, ret_mat=True)
            compSmat = (idmat + curSmat) @ compSmat

        if ret_mat:
            return compSmat
        else:
            tmp = coo_matrix(compSmat)
            return {'idcs1': tmp.col, 'idcs2': tmp.row, 'x': tmp.data}

