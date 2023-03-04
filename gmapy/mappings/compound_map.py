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
from .relative_error_map import RelativeErrorMap
from .mapping_elements import InputSelectorCollection, SumOfDistributors
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
        self.__input = None
        self.__output = None
        if datatable is not None:
            self.instantiate_maps(datatable)

    def instantiate_maps(self, datatable=None):
        if datatable is None:
            if self.__input is None or self.__output is None:
                raise TypeError('neither map list initialized nor datatable provided')
            return
        self._dim = len(datatable)
        resp = np.full(len(datatable.index), False, dtype=bool)
        selcol = InputSelectorCollection()
        distsum = SumOfDistributors()
        for curclass in self.mapclasslist:
            curmap = curclass(datatable, selector_list=selcol.get_selectors())
            curresp = curmap.is_responsible()
            if not np.any(curresp):
                continue
            if np.any(np.logical_and(curresp, resp)):
                raise ValueError(f'Several maps claim responsibility ({str(curmap)})')
            resp = np.logical_or(resp, curresp)
            selcol.add_selectors(curmap.get_selectors())
            distsum.add_distributors(curmap.get_distributors())
        # add the relative error map
        relerrmap = RelativeErrorMap(datatable, distsum,
                                     selector_list=selcol.get_selectors())
        relerr_sels = relerrmap.get_selectors()
        if len(relerr_sels) > 0:
            selcol.add_selectors(relerr_sels)
            relerr_dists = relerrmap.get_distributors()
            distsum = SumOfDistributors([distsum] + relerr_dists)
        # save everything for later
        self.__input = selcol
        self.__output = distsum
        self.__size = len(self.__output)

    def is_responsible(self, datatable=None):
        self.instantiate_maps(datatable)
        ret = np.full(self.__size, False)
        idcs = self.__output.get_indices()
        ret[idcs] = True
        return ret

    def propagate(self, refvals, datatable=None):
        self.instantiate_maps(datatable)
        isresp = self.is_responsible()
        self.__input.assign(refvals)
        propvals = refvals.copy()
        propvals[isresp] = self.__output.evaluate()[isresp]
        return propvals

    def jacobian(self, refvals, datatable=None, with_id=True):
        self.instantiate_maps(datatable)
        self.__input.assign(refvals)
        numel = self.__size
        Smat = self.__output.jacobian()
        if with_id:
            ones = np.full(numel, 1., dtype=float)
            idcs = np.arange(numel)
            idmat = csr_matrix((ones, (idcs, idcs)),
                               shape=(numel, numel), dtype=float)
            Smat += idmat

        return Smat
