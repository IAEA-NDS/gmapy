import time
import matplotlib.pyplot as plt
from gmapy.gma_database_class import GMADatabase
from gmapy.posterior_usu import PosteriorUSU, Posterior
from gmapy.mcmc_inference import (
    symmetric_mh_algo,
    parallel_mh_algo
)
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
from scipy.stats import multivariate_normal
from gmapy.mcmc_inference import compute_effective_sample_size


def step_out(x, z, u, scl=1.5):
    alpha = -1
    x = x.flatten()
    z = z.flatten()
    nl_steps = 1
    while postdist.logpdf(x + alpha*z) > u:
        alpha *= scl 
        nl_steps += 1
    beta = 1
    nr_steps = 1
    while postdist.logpdf(x + beta*z) > u:
        beta *= scl 
        nr_steps += 1
    return alpha, beta, nl_steps, nr_steps  


def step_in(x, z, u, alpha, beta):
    x = x.flatten()
    z = z.flatten()
    cur_alpha = alpha
    cur_beta = beta
    nsteps = 0
    while True:
        nsteps += 1
        g = np.random.uniform(low=cur_alpha, high=cur_beta) 
        prop = x + g*z
        if postdist.logpdf(prop) > u:
            break
        if g < 0:
            cur_alpha = g 
        else:
            cur_beta = g
    return prop, nsteps


def slice_sample(size, x, scl=1.5):
    x = x.reshape(-1, 1).copy()
    res = np.zeros((x.shape[0], size), dtype=float)
    tot_evals = 0
    for i in range(size):
        if i % 100 == 0:
            print(f'obtained {i} samples')
        f = postdist.logpdf(x)
        u = np.log(np.random.uniform()) + f 
        z = propfun(np.zeros(len(x), dtype=float))
        alpha, beta, s1, s2 = step_out(x, z, u, scl)
        # print(s1)
        # print(s2)
        x, s3 = step_in(x, z, u, alpha, beta)
        # print(s3)
        # (new_x - x) / np.sqrt(tmpcov.diagonal())
        res[:, i] = x
        tot_evals += s1 + s2 + s3
    print(tot_evals)
    return res


if __name__ == '__main__':
    gmadb = GMADatabase('../../legacy-tests/test_004/input/data.gma')

    dt = gmadb.get_datatable()
    covmat = gmadb.get_covmat()

    # set up quantities to construct posterior distribution object
    q = prepare_prior_and_likelihood_quantities(dt, covmat)
    priordt = q['priortable']
    priorvals = q['priorvals']
    priorcov = q['priorcov']
    expvals = q['expvals']
    expcov = q['expcov']

    unc_dt = priordt.loc[priordt.NODE.str.match('endep_usu'), ['NODE', 'ENERGY']]
    unc_idcs = unc_dt.index.to_numpy()
    unc_group_assoc = unc_dt.ENERGY.to_numpy()

    m = CompoundMap(dt, reduce=True)
    source_mask = create_propagate_source_mask(priordt)
    postdist = Posterior(
        priorvals, priorcov, m, expvals, expcov,
        relative_exp_errors=True, source_mask=source_mask
    )

    startvals = priorvals.copy()
    lm_res = lm_update(postdist, startvals=startvals, print_status=True, rtol=1e-5, maxiter=20, lmb=1e-2)
    x = lm_res['upd_vals']
    postdist.logpdf(x)

    import pickle
    # with open('slice_results/burnin_smpl.pkl', 'wb') as fout:
    #     pickle.dump(smpl, fout) 
    with open('slice_results/burnin_smpl.pkl', 'rb') as fin:
        smpl = pickle.load(fin)

    st = time.time()
    propfun = postdist.generate_proposal_fun(lm_res['upd_vals'], scale=0.05)
    slice_smpl = slice_sample(100000, smpl[:,-1], scl=1.5)
    ft = time.time()
    print(ft-st)

    propfun = postdist.generate_proposal_fun(lm_res['upd_vals'], scale=0.075, reflect_prop=1e-10)
    mh_smpl = symmetric_mh_algo(smpl[:,-1], postdist.logpdf, propfun, 1000, thin_step=100, print_info=True, save_dir='slice_results')
    mh_smpl['accept_rate']

    propfun = postdist.generate_proposal_fun(lm_res['upd_vals'], scale=2)
    xref = lm_res['upd_vals'].reshape(-1, 1).copy()
    z = propfun(xref)
    d = z - xref
    rz = xref - d
    postdist.logpdf(z)
    postdist.logpdf(rz)

    cursmpl = mh_smpl['samples']
    compute_effective_sample_size(cursmpl[5, :])

    meanvec = np.mean(cursmpl, axis=1)
    meanvec[0]

    plt.hist(cursmpl[0,:])
    plt.show()

    logprob_hist = np.zeros(100000)
    for i, j  in enumerate(np.arange(0, 100000, dtype=int)):
        logprob_hist[i] = postdist.logpdf(cursmpl[:,j])

    plt.plot(np.arange(len(logprob_hist[::100])), logprob_hist[::100])
    plt.show()
