import sys
sys.path.append('../../../')
import sys
sys.path.append('../../..')
import pandas as pd
from scipy.sparse import block_diag
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gmapy.data_management.object_utils import (
    save_objects
)
from gmapy.data_management.uncfuns import (
    create_experimental_covmat,
    create_datablock_covmat_list,
    create_prior_covmat
)
from gmapy.mappings.tf.compound_map_tf \
    import CompoundMap as CompoundMapTF
from gmapy.data_management.database_IO import read_gma_database
from gmapy.data_management.tablefuns import (
    create_prior_table,
    create_experiment_table,
)
from gmapy.mappings.priortools import (
    attach_shape_prior,
    initialize_shape_prior,
    remove_dummy_datasets
)
from gmapy.mappings.tf.restricted_map import RestrictedMap
from gmapy.mappings.tf.energy_dependent_absolute_usu_map_tf import (
    EnergyDependentAbsoluteUSUMap,
    create_endep_abs_usu_df
)
from gmapy.tf_uq.custom_distributions import (
    MultivariateNormal,
    MultivariateNormalLikelihood,
    DistributionForParameterSubset,
    UnnormalizedDistributionProduct
)
from gmapy.gmap import run_gmap_simplified
from sksparse.cholmod import cholesky
from scipy.sparse import csr_matrix, csc_matrix, diags

# db_path = '../../../tests/testdata/data_and_sacs.json'
# db_path = '../../../legacy-tests/test_004/input/data.gma'
db_path = '../../../legacy-tests/test_002/input/data.gma'
db = read_gma_database(db_path)
remove_dummy_datasets(db['datablock_list'])

priortable = create_prior_table(db['prior_list'])
priorcov = create_prior_covmat(db['prior_list'])

# prepare experimental quantities
exptable = create_experiment_table(db['datablock_list'])
expcov = create_experimental_covmat(db['datablock_list'], relative=True)
exptable['UNC'] = np.sqrt(expcov.diagonal())

# initialize the normalization errors
priortable, priorcov = attach_shape_prior((priortable, exptable), covmat=priorcov, raise_if_exists=False)
compmap = CompoundMapTF((priortable, exptable), reduce=True)
initialize_shape_prior((priortable, exptable), compmap)

# some convenient shortcuts
priorvals = priortable.PRIOR.to_numpy()
expvals = exptable.DATA.to_numpy()

# generate a restricted mapping blending out the fixed parameters
is_adj = priorcov.diagonal() != 0.
adj_idcs = np.where(is_adj)[0]
fixed_pars = priorvals[~is_adj]
fixed_idcs = np.where(~is_adj)[0]
restrimap = RestrictedMap(
    len(priorvals), compmap.propagate, compmap.jacobian,
    fixed_params=fixed_pars, fixed_params_idcs=fixed_idcs
)
propfun = tf.function(restrimap.propagate)
jacfun = tf.function(restrimap.jacobian)


def do_gls_iters(remove_idcs=None, num_iters=1, ppp_fix=True):
    curexpcov = expcov.copy()
    addvar_vec = np.zeros(curexpcov.shape[0])
    if remove_idcs is not None:
        addvar_vec[remove_idcs] = 1e6
        curexpcov += diags(addvar_vec)
    res = priorvals[is_adj]
    for i in range(num_iters):
        print(i)
        jacmat_tf = jacfun(res)
        jacmat = tf.sparse.to_dense(jacmat_tf).numpy()
        propvals = propfun(res).numpy()
        if ppp_fix:
            expcov_abs = curexpcov.toarray() * propvals.reshape(-1, 1) * propvals.reshape(1, -1)
        else:
            expcov_abs = curexpcov.toarray() * expvals.reshape(-1, 1) * expvals.reshape(1, -1)
        expcov_abs = csc_matrix(expcov_abs)
        d = expvals - propvals
        factor = cholesky(expcov_abs)
        x1 = factor(d)
        x2 = jacmat.T @ x1
        z = jacmat.T @ factor(jacmat)
        res = priorvals[is_adj] + np.linalg.solve(z, x2)
    return res

##################################################
#  Investigate rise of thermal neutron constant
##################################################

# refres = do_gls_iters(remove_idcs=[4914,4908])
red_priortable = priortable.loc[is_adj].reset_index(drop=True)
refres = do_gls_iters(remove_idcs=[4908, 4914])
idcs = red_priortable.index[red_priortable.REAC.str.match('MT:1-R1:[89]') & (red_priortable.ENERGY == 2.53e-8)]
refres[idcs]

expids = exptable.NODE.unique()
sensar = np.zeros((np.sum(is_adj), len(expids)), dtype=np.float64)
for i, expid in enumerate(expids):
    print(f'{i} of {len(expids)}: removing dataset exp {expid}')
    rem_idcs = set(np.array(exptable.index[exptable.NODE == expid]))
    rem_idcs.add(4908)
    rem_idcs.add(4914)
    rem_idcs = np.array(tuple(rem_idcs))
    sensar[:, i] = do_gls_iters(rem_idcs)

# look at a particular quantity
red_priortable[red_priortable.DESCR=='PU9(n,f)']
# sensar_bckup = sensar.copy()
diffres = sensar - refres.reshape(-1, 1)
myord = np.argsort(diffres[675,:])
ordered_expids = expids[myord]
diffres[675, myord]

considered_expids = set(ordered_expids[:20])
other_expids = considered_expids.copy()
expids_combi = []
sensar2 = np.zeros((np.sum(is_adj), len(considered_expids)*(len(considered_expids)-1)//2), dtype=np.float64)
count = 0
for i, expid1 in enumerate(considered_expids):
    idcs1 = exptable.index[exptable.NODE == expid1]
    other_expids.remove(expid1)
    for j, expid2 in enumerate(other_expids):
        curcombi = (expid1, expid2)
        expids_combi.append(curcombi)
        print(f'{count}: considering {curcombi}')
        idcs2 = exptable.index[exptable.NODE == expid2]
        all_idcs = set(np.concatenate([idcs1, idcs2]))
        all_idcs.update([4908, 4914])
        all_idcs = np.array(tuple(all_idcs))
        sensar2[:, count] = do_gls_iters(all_idcs)
        count += 1

diffres2 = sensar2 - refres.reshape(-1, 1)
myord2 = np.argsort(diffres2[675,:])
diffres2[675, myord2]
np.array(expids_combi)[myord2]

# remove impactful datasets
# idcs = exptable.index[exptable.NODE.isin(('exp_403', 'exp_586', 'exp_602', 'exp_614', 'exp_547', 'exp_588', 'exp_712', 'exp_551'))]

unused_idcs = considered_expids.copy()
# used_expids = set(considered_expids).difference([expid])
# 4, (5), 13, 14
# unused_expids = np.array(list(considered_expids))[np.array([4, 5, 13, 14])]
unused_expids = np.array(list(considered_expids))[np.array([4, 13, 14])]
# unused_expids = np.array(list(considered_expids))[np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
not_used_idcs = exptable.index[exptable.NODE.isin(unused_expids)]
# not_used_idcs = list(set([4908, 4914] + list(not_used_idcs)))
not_used_idcs = list(list(not_used_idcs))
hypores = do_gls_iters(remove_idcs=not_used_idcs, num_iters=30)
idcs = red_priortable.index[(red_priortable.NODE.str.match('xsid_')) & red_priortable.REAC.str.match('MT:1-R1:[89]') & (red_priortable.ENERGY == 2.53e-8)]
print(hypores[idcs])

hypopred = propfun(hypores)
exptable[exptable.REAC.str.match('MT:6-')]
# whatever is removed in the upper code
hypopred[2601] / hypopred[2596]
hypopred[4908]
hypopred[4914]

# SF-PU9 and SF-U5 removed
refpred = propfun(refres)
refpred[2601] / refpred[2596]
refpred[4908]
refpred[4914]

# nothing removed
hypores = do_gls_iters(num_iters=3, ppp_fix=False)
hypopred = restrimap(hypores)
u5_sacs = hypopred[exptable.index[exptable.REAC == 'MT:6-R1:8'][0]]
pu9_sacs = hypopred[exptable.index[exptable.REAC == 'MT:6-R1:9'][0]]
pu9_sacs / u5_sacs
sf_u5 = hypopred[exptable.index[(exptable.REAC == 'MT:1-R1:8') & (exptable.ENERGY == 2.53e-8)][0]]
sf_pu9 = hypopred[exptable.index[(exptable.REAC == 'MT:1-R1:9') & (exptable.ENERGY == 2.53e-8)][0]]
sf_u5
sf_pu9


res = run_gmap_simplified(dbfile=db_path, dbtype='legacy', num_iter=4, remove_dummy=False)
hypores = res['table']['POST'].to_numpy()[np.where(is_adj)[0]]



exptable[(exptable.REAC.str.match('MT:1-R1:[89]')) & (exptable.ENERGY == 2.53e-8)]



unused_expids
exptable.loc[exptable.NODE.isin(unused_expids), ['NODE', 'REAC', 'AUTHOR']].drop_duplicates()
# 1357  exp_403  MT:9-R1:9-R2:3-R3:4  L.W.Weston et al.
# 2078  exp_547       MT:4-R1:9-R2:1  C.WAGEMANS ET AL.
# 5197  exp_536       MT:4-R1:9-R2:8  L.W.WESTON+J.H.TODD
# 2148  exp_602       MT:3-R1:9-R2:8  J.W.MEADOWS
exptable.loc[exptable.NODE=='exp_602']

red_exptable = exptable.loc[exptable.REAC=='MT:9-R1:9-R2:3-R3:4']
plt.scatter(red_exptable.ENERGY, red_exptable.DATA)
plt.show()

idcs = exptable.loc[exptable.NODE == 'exp_403']
red_exptable = exptable.loc[exptable.NODE == 'exp_403']
red_idcs = red_exptable.index
red_expcov = expcov[np.ix_(red_idcs, red_idcs)].toarray()
red_expunc = np.sqrt(red_expcov.diagonal())
red_css = red_exptable.DATA.to_numpy()

red_expunc / red_css
red_expcor = red_expcov / (red_expunc.reshape(1, -1) * red_expunc.reshape(-1, 1))

import matplotlib.pyplot as plt
plt.matshow(red_expcor)
plt.show()





hypores_list = []
for curexpid in considered_expids:
    print(curexpid)
    idcs = exptable.index[exptable.NODE.isin([curexpid])]
    all_idcs = list(set([4908, 4914] + list(idcs)))
    hypores = do_gls_iters(remove_idcs=all_idcs)
    idcs = red_priortable.index[red_priortable.REAC.str.match('MT:1-R1:[9]') & (red_priortable.ENERGY == 2.53e-8)]
    hypores_list.append(hypores[idcs])

hypores_list

# all_idcs = list(set([4908, 4914] + list(idcs_589)))

refres[idcs]
hypores[idcs]




perm_exptable = exptable.set_index('NODE')
perm_exptable = perm_exptable.loc[ordered_expids].copy()
perm_exptable[-200:-150]

tmp = exptable.groupby(['NODE']).agg(reac=('REAC', lambda x: x.iloc[0]), minE=('ENERGY', min), maxE=('ENERGY', max))
tmp = tmp.loc[ordered_expids[::-1]].copy()
tmp
tmp[:30]


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
cnt = 0
for idx, currow in tmp.iterrows():
    cnt += 1
    if cnt > 20:
        break
    y_start = currow['minE']
    y_end = currow['maxE']
    label = idx
    print(idx)
    print(y_start)
    print(y_end)
    if y_start == y_end:
        y_end = y_start + 0.1
        ax.hlines(label, xmin=y_start, xmax=y_end, linewidth=4, color='red')
    else:
        ax.hlines(label, xmin=y_start, xmax=y_end, linewidth=4)

plt.xlim(-0.5, 5)
plt.show()

tmp.loc[:'exp_631']


diffres[675, 377]
expids[377]

exptable.loc[myord]

