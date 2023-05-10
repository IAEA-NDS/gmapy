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
import tensorflow as tf
from .mapping_elements_tf import (
    InputSelectorCollection,
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
        self._reduce = reduce
        self._datatable = datatable
        selcol = InputSelectorCollection()
        self._maplist = []
        for curclass in self.mapclasslist:
            curmap = curclass(datatable, selcol=selcol, reduce=reduce)
            self._maplist.append(curmap)

    @tf.function
    def __call__(self, inputs):
        out_list = []
        for curmap in self._maplist:
            out_list.append(curmap(inputs))
        res = tf.add_n(out_list)
        return res
