import sys
sys.path.append('..')
from gmapy.gma_database_class import GMADatabase
from gmapy.posterior_usu import PosteriorUSU
from gmapy.mcmc_inference import mh_algo
from gmapy.inference import lm_update
from gmapy.mappings.energy_dependent_reac_usu_map import (
    attach_endep_reac_usu_df
)
from gmapy.mappings.priortools import (
    prepare_prior_and_likelihood_quantities,
    create_propagate_source_mask
)
from gmapy.mcmc_inference import (
    load_mcmc_result,
    compute_effective_sample_size
)
from gmapy.inference import lm_update
from gmapy.mappings.compound_map import CompoundMap
import matplotlib.pyplot as plt
import scipy.sparse as sps
import numpy as np


gmadb = GMADatabase('../../legacy-tests/test_004/input/data.gma')

dt = gmadb.get_datatable()
covmat = gmadb.get_covmat()

# attach the usu error contributions here
# red_dt = dt.loc[dt.REAC.str.match('MT:.*-R1:(8|9|10)(-R2:(8|9|10))?$')]
mod_dt = dt[dt.NODE.str.match('xsid_')]
exp_dt = dt[dt.NODE.str.match('exp_')]
exp_dt.groupby('REAC')['REAC'].count()
exp_dt.groupby('REAC')['ENERGY'].max()

myreac = 'MT:1-R1:9'
red_exp_dt = exp_dt.loc[exp_dt.REAC==myreac]
red_mod_dt = mod_dt.loc[mod_dt.REAC==myreac]
red_exp_dt

plt.errorbar(red_exp_dt.ENERGY, red_exp_dt.DATA, red_exp_dt.UNC, fmt='bo', ls='none')
plt.xlim(0.1, 25)
plt.ylim(-1, 3)
# plt.xscale('log')
plt.show()

cdt = dt.copy()
cdt = attach_endep_reac_usu_df(cdt, ['MT:1-R1:9'], [0, 7, 12.5, 20], [0.1]*4)
cdt = attach_endep_reac_usu_df(cdt, ['MT:3-R1:9-R2:8'], [0, 7, 12.5, 20], [0.1]*4)
cdt = attach_endep_reac_usu_df(cdt, ['MT:4-R1:9-R2:8'], [0, 7, 12.5, 20], [0.1]*4)
usu_uncs = cdt.loc[cdt.NODE.str.match('endep_'), 'UNC'].to_numpy()
usu_cov = sps.diags(usu_uncs**2)
ccovmat = sps.block_diag([covmat, usu_cov], format='csr')

# set up quantities to construct posterior distribution object
q = prepare_prior_and_likelihood_quantities(cdt, ccovmat)
priordt = q['priortable']
priorvals = q['priorvals']
priorcov = q['priorcov']
expvals = q['expvals']
expcov = q['expcov']

unc_dt = priordt.loc[priordt.NODE.str.match('endep_reac_usu'), ['NODE', 'REAC', 'ENERGY']]
unc_idcs = unc_dt.index.to_numpy()
unc_group_assoc = np.full(len(unc_idcs), 'one_group')

m = CompoundMap(cdt, reduce=True)
source_mask = create_propagate_source_mask(priordt)
postdist = PosteriorUSU(
    priorvals, priorcov, m, expvals, expcov,
    relative_exp_errors=True, source_mask=source_mask,
    unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
)

uncvec = np.full(len(postdist._groups), 0.1)
startvals = np.concatenate([priorvals + 1e-4, uncvec])
lm_res = lm_update(postdist, startvals=startvals, print_status=True, rtol=1e-5,
                   maxiter=1, must_converge=False)

mh_startvals = lm_res['upd_vals'].copy()
endep_usu_idcs = priordt.index[priordt.NODE.str.match('endep_usu')]
mh_startvals[endep_usu_idcs] = 0.02
mh_startvals[-len(uncvec):] = 0.025

propfun, prop_logpdf = postdist.generate_proposal_fun(mh_startvals, scale=0.06, rho=0.5)

print('Starting sampling... This takes a long time... \n' +
      'Intermediate results are stored in `.pkl` (pickle) files ' +
      'every `save_batchsize=100` samples')
mh_res = mh_algo(mh_startvals, postdist.logpdf, propfun, 20,
                 log_transition_pdf=prop_logpdf, thin_step=100, attempt_resume=True,
                 save_dir='results-009a', save_prefix='mh_res_001_', save_batchsize=100, seed=424239)
# mh_res = load_mcmc_result('mh_res_000_', '.pkl', 'results-009a')

mh_res['elapsed_time']
mh_res['accept_rate']
smpl = mh_res['samples']

plt.plot(np.arange(len(mh_res['logprob_hist'])), mh_res['logprob_hist'])
plt.show()

plt.plot(np.arange(smpl[:,6000:].shape[1]), smpl[-1, 6000:])
plt.show()

meanvec = np.mean(smpl[:, 6000:], axis=1)
stdvec = np.std(smpl[:, 6000:], axis=1)
meanvec[-15:]
stdvec[-15:]
priordt['POST'] = meanvec[:-len(postdist._groups)]
priordt['UNC'] = stdvec[:-len(postdist._groups)]
priordt['RELUNC'] = meanvec[:-len(postdist._groups)] / stdvec[:-len(postdist._groups)]
priordt[priordt.NODE.str.match('xsid_8') & (priordt.ENERGY > 1) & (priordt.ENERGY < 20)]


plt.hist(smpl[-1, 6000:], bins=20)
plt.show()

stdvec[mod_dt.index]
