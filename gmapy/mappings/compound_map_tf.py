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
from .cross_section_ratio_of_sacs_map_tf import CrossSectionRatioOfSacsMap
from .relative_error_map_tf import RelativeErrorMap
from .energy_dependent_usu_map_tf import EnergyDependentUSUMap
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
                CrossSectionFissionAverageMap,
                CrossSectionRatioOfSacsMap
            ]
        dt = datatable
        self._reduce = reduce
        self._indep_idcs = dt.index[~dt.NODE.str.match('exp_', na=False)]
        self._datatable = dt
        selcol = InputSelectorCollection()
        self._selcol = selcol
        self._maplist = []
        for curclass in self.mapclasslist:
            if curclass.is_applicable(dt):
                curmap = curclass(dt, selcol=selcol, reduce=reduce)
                self._maplist.append(curmap)

    def _orig_propagate(self, inputs):
        out_list = []
        for curmap in self._maplist:
            out_list.append(curmap(inputs))
        res = tf.add_n(out_list)
        if not self._reduce:
            inp = InputSelector(self._indep_idcs)(inputs)
            inpdist = Distributor(self._indep_idcs, len(res))(inp)
            res = res + inpdist
        return res

    def _apply_modifier_maps(self, inputs, orig_propvals):
        # modifier maps (rely on the output of the other maps)
        adj_list = []
        if RelativeErrorMap.is_applicable(self._datatable):
            curmap = RelativeErrorMap(
                self._datatable, orig_propvals, self._selcol, self._reduce
            )
            adj_list.append(curmap(inputs))
        if EnergyDependentUSUMap.is_applicable(self._datatable):
            curmap = EnergyDependentUSUMap(
                self._datatable, orig_propvals, self._selcol, self._reduce
            )
            adj_list.append(curmap(inputs))

        if len(adj_list) == 0:
            return orig_propvals
        else:
            res = orig_propvals + tf.add_n(adj_list)
            return res

    def __call__(self, inputs):
        orig_propvals = self._orig_propagate(inputs)
        res = self._apply_modifier_maps(inputs, orig_propvals)
        return res

    def _orig_jacobian(self, inputs):
        first = True
        res = None
        for curmap in self._maplist:
            curjac = curmap.jacobian(inputs)
            if first is True:
                first = False
                res = curjac
            else:
                res = tf.sparse.add(res, curjac)
        return res

    def jacobian(self, inputs):
        orig_propvals = self._orig_propagate(inputs)
        orig_jac = self._orig_jacobian(inputs)
        final_jac = orig_jac
        if RelativeErrorMap.is_applicable(self._datatable):
            curmap = RelativeErrorMap(
                self._datatable, orig_propvals, self._selcol, self._reduce
            )
            curjac = curmap.jacobian(inputs, orig_jac)
            final_jac = tf.sparse.add(final_jac, curjac)

        if EnergyDependentUSUMap.is_applicable(self._datatable):
            curmap = EnergyDependentUSUMap(
                self._datatable, orig_propvals, self._selcol, self._reduce
            )
            curjac = curmap.jacobian(inputs, orig_jac)
            final_jac = tf.sparse.add(final_jac, curjac)
        return final_jac
