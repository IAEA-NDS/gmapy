import numpy as np
from scipy.sparse import csr_matrix
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


def mapclass_with_params(origclass, *args, **kwargs):
    class WrapperClass(origclass):
        def __init__(self, datatable):
            super().__init__(datatable, *args, **kwargs)
    return WrapperClass


class CompoundMap:

    def __init__(self, fix_sacs_jacobian=True, legacy_integration=False):
        self.mapclasslist = [
                CrossSectionMap,
                CrossSectionShapeMap,
                CrossSectionRatioMap,
                CrossSectionRatioShapeMap,
                CrossSectionAbsoluteRatioMap,
                CrossSectionShapeOfRatioMap,
                CrossSectionTotalMap,
                CrossSectionShapeOfSumMap,
                mapclass_with_params(
                    CrossSectionFissionAverageMap,
                    fix_jacobian=fix_sacs_jacobian,
                    legacy_integration=legacy_integration
                ),
                mapclass_with_params(
                    CrossSectionRatioOfSacsMap,
                    atol=1e-5, rtol=1e-05, maxord=16
                )
            ]

    def is_responsible(self, datatable):
        resp = np.full(len(datatable.index), False, dtype=bool)
        for curclass in self.mapclasslist:
            curmap = curclass(datatable)
            curresp = curmap.is_responsible()
            resp_overlap = np.logical_and(resp, curresp)
            if np.any(resp_overlap):
                print(datatable[resp_overlap])
                raise ValueError(f'Several maps claim responsibility ({str(curmap)})')
            resp = np.logical_or(resp, curresp)
        sorted_resp = resp[datatable.index]
        return sorted_resp

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

        for curclass in self.mapclasslist:
            curmap = curclass(datatable)
            curresp = curmap.is_responsible()
            if not np.any(curresp):
                continue
            if np.any(np.logical_and(treated, curresp)):
                raise ValueError('Several maps claim responsibility for the same rows')
            treated[curresp] = True
            curvals = curmap.propagate(refvals)
            if np.any(np.logical_and(propvals!=0., curvals!=0.)):
                raise ValueError('Several maps contribute to same experimental datapoint')
            propvals += curvals

        return propvals

    def jacobian(self, datatable, refvals):
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
        for curclass in self.mapclasslist:
            curmap = curclass(datatable)
            if not np.any(curmap.is_responsible()):
                continue
            curSmat = curmap.jacobian(refvals)
            compSmat = (idmat + curSmat) @ compSmat

        return compSmat
