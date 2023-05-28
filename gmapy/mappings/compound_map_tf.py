import numpy as np
from .cross_section_map_tf import CrossSectionMap
from .cross_section_shape_map_tf import CrossSectionShapeMap
from .cross_section_ratio_map_tf import CrossSectionRatioMap
from .cross_section_ratio_shape_map_tf import CrossSectionRatioShapeMap
from .cross_section_absolute_ratio_map_tf import CrossSectionAbsoluteRatioMap
from .cross_section_shape_of_ratio_map_tf import CrossSectionShapeOfRatioMap
from .cross_section_shape_of_sum_map_tf import CrossSectionShapeOfSumMap
from .cross_section_total_map_tf import CrossSectionTotalMap
from .cross_section_fission_average_map_tf import CrossSectionFissionAverageMap
from .relative_error_map_tf import RelativeErrorMap
import tensorflow as tf
from .mapping_elements_tf import (
    InputSelectorCollection,
    InputSelector,
    Distributor
)


class CompoundMap(tf.Module):

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
                CrossSectionShapeOfSumMap,
                CrossSectionTotalMap,
                CrossSectionFissionAverageMap
            ]
        dt = datatable
        self._reduce = reduce
        self._indep_idcs = dt.index[~dt.NODE.str.match('exp_', na=False)]
        self._datatable = dt
        selcol = InputSelectorCollection()
        self._selcol = selcol
        self._maplist = []
        for curclass in self.mapclasslist:
            curmap = curclass(dt, selcol=selcol, reduce=reduce)
            self._maplist.append(curmap)

    @tf.function
    def __call__(self, inputs):
        out_list = []
        for curmap in self._maplist:
            out_list.append(curmap(inputs))
        res = tf.add_n(out_list)
        if not self._reduce:
            inp = InputSelector(self._indep_idcs)(inputs)
            inpdist = Distributor(self._indep_idcs, len(res))(inp)
            res = res + inpdist
        # modifier maps (rely on the output of the other maps)
        if RelativeErrorMap.is_applicable(self._datatable):
            curmap = RelativeErrorMap(
                self._datatable, res, self._selcol, self._reduce
            )
            res = res + curmap(inputs)
        return res
