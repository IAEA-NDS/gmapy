import numpy as np
from statsmodels.tsa.stattools import acf
from .mappings.compound_map import CompoundMap
from .inference import superseded_lm_update
from .posterior import Posterior
from multiprocessing import Process, Pipe
import time


def mh_algo(startvals, log_probdens, proposal, num_samples,
            log_transition_pdf=None, num_burn=0, thin_step=1,
            seed=None, print_info=False):
    if seed is not None:
        np.random.seed(seed)
    startvals = startvals.reshape(-1, 1)
    dim = startvals.shape[0]
    num_total = num_samples + num_burn
    samples = np.zeros((dim, num_samples), dtype=float)
    logprob_hist = np.zeros(num_samples, dtype=float)
    start_time = time.time()
    curvals = startvals
    cur_logprob = log_probdens(curvals)
    num_acc = 0
    i = -num_burn
    j = 0
    while i < num_samples:
        j += 1
        candidate = proposal(curvals)
        cand_logprob = log_probdens(candidate)
        log_alpha = cand_logprob - cur_logprob
        if log_transition_pdf is not None:
            log_alpha += log_transition_pdf(candidate, curvals)
            log_alpha -= log_transition_pdf(curvals, candidate)
        log_u = np.log(np.random.uniform())
        if log_u < log_alpha:
            curvals = candidate
            cur_logprob = cand_logprob
            if i >= 0:
                num_acc += 1
        if j >= thin_step:
            if i >= 0:
                samples[:, i:i+1] = curvals
                logprob_hist[i] = cur_logprob
            j = 0
            i += 1
            print(f'Obtained sample number {i}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    accept_rate = num_acc / (num_samples*thin_step)
    result = {
        'samples': samples,
        'accept_rate': accept_rate,
        'logprob_hist': logprob_hist,
        'elapsed_time': elapsed_time,
        'num_burn': num_burn,
        'thin_step': thin_step,
        'seed': seed
    }
    return result


def symmetric_mh_algo(startvals, log_probdens, proposal, num_samples,
                      num_burn=0, thin_step=1, seed=None, print_info=False):
    return mh_algo(
        startvals, log_probdens, proposal, num_samples,
        num_burn=num_burn, thin_step=thin_step,
        seed=seed, print_info=print_info
    )


def symmetric_mh_worker(con, mh_args, mh_kwargs):
    mh_res = symmetric_mh_algo(*mh_args, **mh_kwargs)
    con.send(mh_res)
    con.close()


def parallel_symmetric_mh_algo(num_workers, startvals, log_probdens, proposal, num_samples,
                               num_burn=0, thin_step=1, seeds=None, print_info=False):
    if seeds is None:
        seeds = np.random.randint(low=0, high=65535, size=num_workers)
    elif len(seeds) != num_workers:
        raise ValueError('number of seed values must equal num_workers')
    mh_args = (startvals, log_probdens, proposal, num_samples)
    mh_kwargs = {'num_burn': num_burn, 'thin_step': thin_step}
    pipe_parents = []
    pipe_children = []
    procs = []
    for i, seed in zip(range(num_workers), seeds):
        mh_kwargs['seed'] = seed
        pipe_parent, pipe_child = Pipe()
        pipe_parents.append(pipe_parent)
        pipe_children.append(pipe_child)
        proc = Process(target=symmetric_mh_worker,
                       args=(pipe_child, mh_args, mh_kwargs))
        proc.start()
        procs.append(proc)

    mh_res_list = []
    for pipe_parent, proc in zip(pipe_parents, procs):
        mh_res_list.append(pipe_parent.recv())
        proc.join()
    return mh_res_list


def compute_acceptance_rate(samples):
    num_samples = samples.shape[1]
    num_reject = np.sum(np.sum(samples[:, 1:] - samples[:, :-1], axis=0) == 0)
    return 1 - num_reject / num_samples


def compute_effective_sample_size(arr):
    n = len(arr)
    acfvec = acf(arr, nlags=n, fft=True)
    sums = 0
    for k in range(1, len(acfvec)):
        sums = sums + (n-k)*acfvec[k]/n

    return n/(1+2*sums)


def gmap_mh_inference(datatable, covmat, num_samples, prop_scaling,
                      startvals=None, num_burn=0, thin_step=1, int_rtol=1e-4,
                      relative_exp_errors=False, num_workers=1):
    if not np.all(datatable.index == np.sort(datatable.index)):
        raise IndexError('index of datatable must be sorted')
    # prepare the relevant objects, e.g., prior values and prior covariance matrix
    dt = datatable.sort_index()
    exp_sel = dt['NODE'].str.match('exp_', na=False)
    exp_idcs = dt[exp_sel].index
    prior_idcs = dt[~exp_sel].index
    priorvals = dt.loc[prior_idcs, 'PRIOR'].to_numpy()
    priorcov = covmat.tocsr()[prior_idcs,:].tocsc()[:,prior_idcs]
    expvals = dt.loc[exp_idcs, 'DATA'].to_numpy()
    expcov = covmat.tocsr()[exp_idcs,:].tocsc()[:,exp_idcs]
    if np.any(expcov.diagonal() <= 0):
        raise ValueError('observed data must have non-zero uncertainty')
    if startvals is None:
        print('Determine initial values for MCMC chain...')
        mapping = CompoundMap(datatable, rtol=int_rtol, reduce=False)
        lmres = superseded_lm_update(mapping, datatable, covmat, print_status=True)
        startvals = np.empty(len(datatable), dtype=float)
        startvals[prior_idcs] = priorvals
        startvals[lmres['idcs']] = lmres['upd_vals']
        startvals = startvals[prior_idcs]
    else:
        if startvals.size != len(datatable):
            raise IndexError('startvals must be of same length as datatable')
        startvals = startvals[prior_idcs]
    # intialize objects to sample and to obtain values from log posterior pdf
    mapping = CompoundMap(datatable, rtol=int_rtol, reduce=True)
    post = Posterior(priorvals, priorcov, mapping, expvals, expcov,
                     relative_exp_errors=relative_exp_errors)
    propfun = post.generate_proposal_fun(startvals, scale=prop_scaling)

    print('Construct the MCMC chain...')
    mh_args = (startvals, post.logpdf, propfun)
    mh_kwargs = {'num_samples': num_samples, 'num_burn': num_burn,
                 'thin_step': thin_step}
    if num_workers == 1:
        mh_res = symmetric_mh_algo(*mh_args, **mh_kwargs)
        return [mh_res]
    else:
        mh_res_list = parallel_symmetric_mh_algo(num_workers, *mh_args, **mh_kwargs)
        return mh_res_list
