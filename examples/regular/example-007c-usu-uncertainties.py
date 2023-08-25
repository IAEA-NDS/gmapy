import sys
sys.path.append('..')
from gmapy.gma_database_class import GMADatabase
from gmapy.posterior_usu import PosteriorUSU
from gmapy.mcmc_inference import mh_algo
from gmapy.inference import lm_update
from gmapy.mappings.energy_dependent_usu_map import attach_endep_usu_df
from gmapy.mappings.priortools import (
    prepare_prior_and_likelihood_quantities,
    create_propagate_source_mask
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
red_dt = dt.loc[dt.REAC.str.match('MT:.*-R1:(8|9|10)(-R2:(8|9|10))?$')]
red_dt = red_dt[red_dt.NODE.str.match('exp_')]
red_dt.groupby('REAC')['REAC'].count()
red_dt.groupby('REAC')['ENERGY'].max()

reac_dt = red_dt.loc[red_dt.REAC=='MT:1-R1:9']

plt.errorbar(reac_dt.ENERGY, reac_dt.DATA, reac_dt.UNC, fmt='bo', ls='none')
plt.xlim(1, 20) 
plt.ylim(0.75, 3) 
# plt.xscale('log')
plt.show()

cdt = attach_endep_usu_df(dt, ['MT:1-R1:9'], [1, 7, 12.5, 20], [0.1]*4)
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

unc_dt = priordt.loc[priordt.NODE.str.match('endep_usu'), ['NODE', 'ENERGY']]
unc_idcs = unc_dt.index.to_numpy()
unc_group_assoc = np.full(len(unc_idcs), 'usu_group')

m = CompoundMap(cdt, reduce=True)
source_mask = create_propagate_source_mask(priordt)
postdist = PosteriorUSU(
    priorvals, priorcov, m, expvals, expcov,
    relative_exp_errors=True, source_mask=source_mask,
    unc_idcs=unc_idcs, unc_group_assoc=unc_group_assoc
)

uncvec = np.full(len(postdist._groups), 0.1)
startvals = np.concatenate([priorvals + 1e-4, uncvec])
lm_res = lm_update(postdist, startvals=startvals, print_status=True, rtol=1e-5)

mh_startvals = lm_res['upd_vals'].copy()
endep_usu_idcs = priordt.index[priordt.NODE.str.match('endep_usu')]
mh_startvals[endep_usu_idcs] = 0.02
mh_startvals[-len(uncvec):] = 0.025

propfun, prop_logpdf = postdist.generate_proposal_fun(mh_startvals, scale=0.055, rho=0.5)

print('Starting sampling... This takes a long time... \n' +
      'Intermediate results are stored in `.pkl` (pickle) files ' +
      'every `save_batchsize=1000` samples')
mh_res = mh_algo(
    mh_startvals, postdist.logpdf, propfun, 9000,
    log_transition_pdf=prop_logpdf, thin_step=100,
    attempt_resume=True, save_batchsize=1000
)
# import pickle
# with open('data-example-007c-30000-mh-res.pkl', 'wb') as fout:
#     pickle.dump(mh_res, fout)

mh_res['accept_rate']
smpl = mh_res['samples']

plt.plot(np.arange(len(mh_res['logprob_hist'])), mh_res['logprob_hist'])
plt.show()

plt.plot(np.arange(smpl.shape[1]), smpl[-2, :])
plt.show()

meanvec = np.mean(smpl[:, 5000:], axis=1)
stdvec = np.std(smpl[:, 5000:], axis=1)
meanvec[-10:]
stdvec[-10:]

plt.hist(smpl[-2,5000:])
plt.show()

