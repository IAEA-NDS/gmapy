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
from .helperfuns import mapclass_with_params


class CompoundMap:

    def __init__(self, datatable=None, fix_sacs_jacobian=True,
                 legacy_integration=False):
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
        self.maplist = None
        if datatable is not None:
            self.instantiate_maps(datatable)

    def instantiate_maps(self, datatable=None):
        if datatable is None:
            if self.maplist is None:
                raise TypeError('neither map list initialized nor datatable provided')
            return
        self.maplist = []
        self._dim = len(datatable)
        resp = np.full(len(datatable.index), False, dtype=bool)
        for curclass in self.mapclasslist:
            curmap = curclass(datatable)
            curresp = curmap.is_responsible()
            if not np.any(curresp):
                continue
            self.maplist.append(curmap)
            if np.any(np.logical_and(curresp, resp)):
                raise ValueError(f'Several maps claim responsibility ({str(curmap)})')
            resp = np.logical_or(resp, curresp)

    def is_responsible(self, datatable=None):
        self.instantiate_maps(datatable)
        resp = self.maplist[0].is_responsible()
        for curmap in self.maplist[1:]:
            curresp = curmap.is_responsible()
            resp = np.logical_or(resp, curresp)
        return resp

    def propagate(self, refvals, datatable=None):
        self.instantiate_maps(datatable)
        isresp = self.is_responsible()
        not_isresp = np.logical_not(isresp)
        propvals = np.full(datatable.shape[0], 0.)
        propvals[not_isresp] = refvals[not_isresp]
        for curmap in self.maplist:
            propvals += curmap.propagate(refvals)
        return propvals

    def jacobian(self, refvals, datatable=None):
        self.instantiate_maps(datatable)
        numel = self._dim
        idcs = np.arange(numel)
        ones = np.full(numel, 1., dtype=float)
        idmat = csr_matrix((ones, (idcs, idcs)),
                           shape=(numel, numel), dtype=float)
        compSmat = idmat
        for curmap in self.maplist:
            curSmat = curmap.jacobian(refvals)
            compSmat = (idmat + curSmat) @ compSmat

        return compSmat
