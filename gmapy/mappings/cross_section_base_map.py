import numpy as np
from .mapping_elements import (
    InputSelectorCollection
)
from .priortools import prepare_prior_and_exptable


class CrossSectionBaseMap:

    def __init__(self, datatable, selcol=None, distsum=None, reduce=False,
                 more_prepare_args=None):
        if more_prepare_args is None:
            more_prepare_args = {}
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        self._input, self._output = self._base_prepare(
            datatable, selcol, reduce, more_prepare_args
        )
        if distsum is not None:
            distsum.add_distributors(self._output.get_distributors())

    def is_responsible(self):
        ret = np.full(self.__numrows, False)
        if self._output is not None:
            idcs = self._output.get_indices()
            ret[idcs] = True
        return ret

    def propagate(self, refvals):
        self._input.assign(refvals)
        return self._output.evaluate()

    def jacobian(self, refvals):
        self._input.assign(refvals)
        return self._output.jacobian()

    def get_selectors(self):
        return self._input.get_selectors()

    def get_distributors(self):
        return self._output.get_distributors()

    def _base_prepare(self, datatable, selcol, reduce, more_prepare_args):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(datatable, reduce)
        self._src_len = src_len
        self._tar_len = tar_len
        inp, out = self._prepare(
            priortable, exptable, selcol, **more_prepare_args
        )
        return inp, out

    def _prepare(self, priortable, exptable, selcol, **more_prepare_args):
        raise NotImplementedError(
            'Please implement this method in the derived class'
        )
