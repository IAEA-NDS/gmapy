import numpy as np
import pandas as pd
from .mapping_elements import (
    InputSelectorCollection,
    Selector,
    Distributor,
    SumOfDistributors,
    LinearInterpolation
)
from .priortools import prepare_prior_and_exptable


class EnergyDependentReacUSUMap:

    def __init__(self, datatable, distributor_like,
                 selcol=None, distsum=None, reduce=False):
        if type(distributor_like) not in (Distributor, SumOfDistributors):
            raise TypeError('distributor_like must be of class Distributor '
                            'or class SumOfDistributors')
        self.__numrows = len(datatable)
        if selcol is None:
            selcol = InputSelectorCollection()
        self.__input, self.__output = self.__prepare(
            datatable, distributor_like, selcol, reduce
        )
        if distsum is not None:
            distsum.add_distributors(self.__output.get_distributors())

    def is_responsible(self):
        ret = np.full(self.__numrows, False)
        if self.__output is not None:
            idcs = self.__output.get_indices()
            ret[idcs] = True
        return ret

    def propagate(self, refvals):
        self.__input.assign(refvals)
        return self.__output.evaluate()

    def jacobian(self, refvals):
        self.__input.assign(refvals)
        return self.__output.jacobian()

    def get_selectors(self):
        return self.__input.get_selectors()

    def get_distributors(self):
        return self.__output.get_distributors()

    def __prepare(self, datatable, distributor_like, selcol, reduce):
        priortable, exptable, src_len, tar_len = \
            prepare_prior_and_exptable(datatable, reduce)

        priormask = priortable['NODE'].str.match('endep_reac_usu_', na=False)
        priortable = priortable[priormask]
        expmask = exptable['NODE'].str.match('exp_', na=False)
        exptable = exptable[expmask]

        inp = InputSelectorCollection()
        out = SumOfDistributors()
        if len(priortable) == 0 or len(exptable) == 0:
            return inp, out

        # construct the selectors
        if type(distributor_like) == SumOfDistributors:
            aux_dists = distributor_like.get_distributors()
        else:
            aux_dists = [distributor_like]
        aux_distsum = SumOfDistributors(aux_dists)

        # establish the mapping for each USU component
        # associated with an experimental dataset
        tmp_out = SumOfDistributors()
        rerr_reacids = priortable['NODE'].str.extract(r'endep_reac_usu_(.+)$')
        rerr_reacids = pd.unique(rerr_reacids.iloc[:, 0])
        for reacid in rerr_reacids:
            srcnode = 'endep_reac_usu_' + reacid
            srcdt = priortable[priortable['NODE'] == srcnode]
            tardt = exptable[exptable['REAC'] == reacid]
            inpvar = selcol.define_selector(srcdt.index, src_len)
            intres = LinearInterpolation(
                inpvar, srcdt['ENERGY'], tardt['ENERGY'], zero_outside=True
            )
            outvar = Distributor(intres, tardt.index, tar_len)
            inp.add_selector(inpvar)
            tmp_out.add_distributor(outvar)

        # target_indices = tmp_out.get_indices()
        expquants = aux_distsum
        abserrors = tmp_out * expquants
        abserrors_dist = Distributor(abserrors, np.arange(tar_len), tar_len)
        out.add_distributor(abserrors_dist)
        return inp, out


def attach_endep_reac_usu_df(datatable, reacs, energies, uncs):
    dt = datatable.copy()
    dt.sort_index()
    red_dt = dt[dt.NODE.str.match('exp_[0-9]+') & dt.REAC.isin(reacs)]
    reacids = pd.unique(red_dt.REAC)
    dt_list = [dt]
    for reacid in reacids:
        curnode = 'endep_reac_usu_' + reacid
        new_usu_dt = pd.DataFrame.from_dict({
            'NODE': curnode,
            'REAC': reacid,
            'ENERGY': energies,
            'PRIOR': 0.0,
            'DATA': np.nan,
            'UNC': uncs
        })
        dt_list.append(new_usu_dt)

    res_dt = pd.concat(dt_list, axis=0, ignore_index=True)
    return res_dt