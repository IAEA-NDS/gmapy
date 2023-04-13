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
from .energy_dependent_usu_map import EnergyDependentUSUMap
from .relative_error_map import RelativeErrorMap
from .mapping_elements import InputSelectorCollection, SumOfDistributors
from .helperfuns import mapclass_with_params


class CompoundMap:

    def __init__(self, datatable=None, fix_sacs_jacobian=True,
                 legacy_integration=False, reduce=False,
                 atol=1e-8, rtol=1e-5, maxord=16):
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
                    legacy_integration=legacy_integration,
                    atol=atol, rtol=rtol, maxord=maxord
                ),
                mapclass_with_params(
                    CrossSectionRatioOfSacsMap,
                    atol=atol, rtol=rtol, maxord=maxord
                )
            ]
        self.__reduce = reduce
        self.__input = None
        self.__output = None
        if datatable is not None:
            self.instantiate_maps(datatable, reduce)

    def instantiate_maps(self, datatable=None, reduce=False):
        if datatable is None:
            if self.__input is None or self.__output is None:
                raise TypeError('neither map list initialized nor datatable provided')
            return
        self._dim = len(datatable)
        resp = np.full(len(datatable.index), False, dtype=bool)
        selcol = InputSelectorCollection()
        tmp_distsum = SumOfDistributors()
        for curclass in self.mapclasslist:
            curmap = curclass(datatable, selcol=selcol, distsum=tmp_distsum,
                              reduce=reduce)
            curresp = curmap.is_responsible()
            if np.any(np.logical_and(curresp, resp)):
                raise ValueError(f'Several maps claim responsibility ({str(curmap)})')
            resp = np.logical_or(resp, curresp)

        # add the relative error map
        relerrmap = RelativeErrorMap(
            datatable, tmp_distsum, selcol=selcol, reduce=reduce
        )
        relerr_dists = relerrmap.get_distributors()
        # add the energy dependent USU error map
        usumap = EnergyDependentUSUMap(
            datatable, tmp_distsum, selcol=selcol, reduce=reduce
        )
        usumap_dists = usumap.get_distributors()
        # '+' means here to concatenate the lists
        all_dists = []
        all_dists.extend(relerr_dists)
        all_dists.extend(usumap_dists)
        all_dists.extend(tmp_distsum.get_distributors())
        distsum = SumOfDistributors(all_dists)
        # save everything for later
        self.__input = selcol
        self.__output = distsum
        self.__size = len(self.__output)

    def get_selectors(self):
        return self.__input.get_selectors()

    def get_distributors(self):
        return self.__output.get_distributors()

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
        outvals = self.__output.evaluate()
        if not self.__reduce:
            propvals = refvals.copy()
            propvals[isresp] = outvals[isresp]
            return propvals
        else:
            return outvals

    def jacobian(self, refvals, datatable=None, with_id=True):
        self.instantiate_maps(datatable)
        self.__input.assign(refvals)
        Smat = self.__output.jacobian()
        if with_id and not self.__reduce:
            numel = self.__size
            ones = np.full(numel, 1., dtype=float)
            idcs = np.arange(numel)
            idmat = csr_matrix((ones, (idcs, idcs)),
                               shape=(numel, numel), dtype=float)
            Smat += idmat

        return Smat
