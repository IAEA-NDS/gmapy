import numpy as np
from .cross_section_map_tf import CrossSectionMap
from .cross_section_shape_map_tf import CrossSectionShapeMap
from .cross_section_ratio_map_tf import CrossSectionRatioMap
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
