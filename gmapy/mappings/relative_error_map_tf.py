import pandas as pd
from .cross_section_modifier_base_map_tf import CrossSectionModifierBaseMap


class RelativeErrorMap(CrossSectionModifierBaseMap):

    @classmethod
    def is_applicable(cls, datatable):
        return (
            (datatable['NODE'].str.match('exp_', na=False)).any() &
            (datatable['NODE'].str.match('relerr_([0-9]+)$')).any()
        ).any()

    def _prepare_propagate(self, priortable, exptable):
        priormask = priortable['NODE'].str.match('relerr_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]
        # determine the source and target indices of the mapping
        expids = exptable['NODE'].str.extract(r'exp_([0-9]+)$')
        ptidx = exptable['PTIDX']
        rerr_expids = priortable['NODE'].str.extract(r'relerr_([0-9]+)$')
        rerr_ptidx = priortable['PTIDX']
        mapdf1 = pd.concat([expids, ptidx], axis=1)
        mapdf1.columns = ('expid', 'ptidx')
        mapdf1.reset_index(inplace=True, drop=False)
        mapdf1.set_index(['expid', 'ptidx'], inplace=True)
        mapdf2 = pd.concat([rerr_expids, rerr_ptidx], axis=1)
        mapdf2.columns = ('expid', 'ptidx')
        mapdf2.reset_index(inplace=True, drop=False)
        mapdf2.set_index(['expid', 'ptidx'], inplace=True)
        source_indices = mapdf2['index'].to_numpy()
        target_indices = mapdf1.loc[list(mapdf2.index), 'index'].to_numpy()
        propfun = self._generate_atomic_propagate()
        self._add_lists([source_indices], target_indices, propfun)

    def _generate_atomic_propagate(self):
        def _atomic_propagate(propvals, inpvars):
            return propvals * inpvars
        return _atomic_propagate
