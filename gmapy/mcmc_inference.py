import os
import numpy as np
import pickle
from statsmodels.tsa.stattools import acf
from .mappings.compound_map import CompoundMap
from .inference import superseded_lm_update
from .posterior import Posterior
from multiprocessing import Process, Pipe
from copy import deepcopy
import time


def _mh_algo(
    random_generator, startvals, log_probdens, proposal,
    num_samples, log_transition_pdf=None, num_burn=0, thin_step=1,
):
    initial_rg_state = deepcopy(random_generator.bit_generator.state)
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
        candidate = proposal(curvals, random_state=random_generator)
        cand_logprob = log_probdens(candidate)
        log_alpha = cand_logprob - cur_logprob
        if log_transition_pdf is not None:
            t_logprob, inv_t_logprob = log_transition_pdf(curvals, candidate)
            log_alpha += inv_t_logprob
            log_alpha -= t_logprob
        log_u = np.log(random_generator.uniform())
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
    end_time = time.time()
    final_rg_state = deepcopy(random_generator.bit_generator.state)
    elapsed_time = end_time - start_time
    accept_rate = num_acc / (num_samples*thin_step)
    result = {
        'samples': samples,
        'accept_rate': accept_rate,
        'logprob_hist': logprob_hist,
        'elapsed_time': elapsed_time,
        'num_burn': num_burn,
        'thin_step': thin_step,
        'initial_rg_state': initial_rg_state,
        'final_rg_state': final_rg_state
    }
    return result


def _get_chain_files(prefix, suffix, save_dir):
    chain_files = os.listdir(save_dir)
    chain_files = [fn for fn in chain_files if fn.startswith(prefix)]
    chain_files = [fn for fn in chain_files if fn.endswith(suffix)]
    countstrs = [fn[len(prefix):-len(suffix)] for fn in chain_files]
    should_include = np.array([len(c) == 6 and c.isdigit() for c in countstrs])
    if len(should_include) == 0:
        return None
    chain_files = np.array(chain_files)[should_include]
    countstrs = np.array(countstrs)[should_include]
    count_seq = np.array([int(c) for c in countstrs])
    order = np.argsort(count_seq)
    sorted_count_seq = count_seq[order]
    if (not np.all(np.diff(sorted_count_seq) == 1) or
            sorted_count_seq[0] != 0):
        raise FileNotFoundError('Missing files in the MCMC chain, aborting')
    sorted_chain_files = chain_files[order]
    return sorted_chain_files


def _get_resume_info(prefix, suffix, save_dir, attempt_resume):
    chain_files = _get_chain_files(prefix, suffix, save_dir)
    if chain_files is not None and not attempt_resume:
        raise FileExistsError('Strongly refusing to overwrite existing files')
    if chain_files is None:
        return None
    sample_count = 0
    for i, curfile in enumerate(chain_files):
        curpath = os.path.join(save_dir, curfile)
        with open(curpath, 'rb') as fin:
            curcont = pickle.load(fin)
        sample_count += curcont['samples'].shape[1]
        # NOTE: overriding startvals
        startvals = curcont['samples'][:, -1]
        final_rg_state = curcont['final_rg_state']
    # initial_rg_state = last_mh_res['initial_rng_state']
    rg = np.random.Generator(np.random.PCG64())
    rg.bit_generator.state = final_rg_state
    cur_file_idx = len(chain_files)
    return {
        'random_generator': rg,
        'cur_file_idx': cur_file_idx,
        'sample_count': sample_count,
        'startvals': startvals
    }


def load_mcmc_result(prefix, suffix, save_dir, idcs=None):
    chain_files = _get_chain_files(prefix, suffix, save_dir)
    if chain_files is None:
        raise FileNotFoundError('files of the chain not found')
    if idcs is not None:
        idcs_set = set(idcs)
    offset = 0
    sample_list = []
    logprob_hist_list = []
    cum_elapsed_time = 0
    num_accepted = 0
    num_total = 0
    for i, curfile in enumerate(chain_files):
        curpath = os.path.join(save_dir, curfile)
        with open(curpath, 'rb') as fin:
            curcont = pickle.load(fin)
        cursamples = curcont['samples']
        cum_elapsed_time += curcont['elapsed_time']
        logprob_hist_list.append(curcont['logprob_hist'])
        num_total += cursamples.shape[1] * curcont['thin_step']
        num_accepted += \
            curcont['accept_rate'] * cursamples.shape[1] * curcont['thin_step']
        if idcs is None:
            sample_list.append(cursamples)
        else:
            curidcs = np.arange(cursamples.shape[1]) + offset
            is_sel = [idx in idcs_set for idx in curidcs]
            sel_idcs = curidcs[is_sel]
            sample_list.append(cursamples[:, sel_idcs])
    mcmc_res = {
        'samples': np.hstack(sample_list),
        'logprob_hist': np.hstack(logprob_hist_list),
        'elapsed_time': cum_elapsed_time,
        'num_total': num_total,
        'accept_rate': num_accepted / num_total
    }
    if idcs is not None:
        mcmc_res['idcs'] = idcs.copy()
    return mcmc_res


def mh_algo(
    startvals, log_probdens, proposal, num_samples,
    log_transition_pdf=None, thin_step=1,
    seed=None, print_info=True, attempt_resume=False,
    save_prefix='mh_res_', save_suffix='.pkl', save_dir='.',
    save_batchsize=1000, return_type='chain'
):
    num_burn = 0
    resume_info = _get_resume_info(
        save_prefix, save_suffix, save_dir, attempt_resume
    )
    if resume_info is None:
        rg = np.random.Generator(np.random.PCG64(seed))
        cur_file_idx = 0
        sample_count = 0
    else:
        rg = resume_info['random_generator']
        cur_file_idx = resume_info['cur_file_idx']
        sample_count = resume_info['sample_count']
        startvals = resume_info['startvals']

    total_count = num_burn + num_samples
    samples_obtained = sample_count
    remaining_sample_count = num_burn + num_samples - sample_count
    while remaining_sample_count > 0:
        curbatchsize = min(save_batchsize, remaining_sample_count)
        mh_res = _mh_algo(
            rg, startvals, log_probdens, proposal, curbatchsize,
            log_transition_pdf=log_transition_pdf, num_burn=0,
            thin_step=thin_step
        )
        startvals = mh_res['samples'][:,-1].copy()
        if cur_file_idx == 0:
            mh_res['seed'] = seed
        cur_samples_obtained = min(curbatchsize, mh_res['samples'].shape[1])
        remaining_sample_count -= cur_samples_obtained
        samples_obtained += cur_samples_obtained
        curfile = save_prefix + '{:06d}'.format(cur_file_idx) + save_suffix
        curpath = os.path.join(save_dir, curfile)
        if os.path.isfile(curpath):
            raise FileExistsError(f'file {curpath} already exists')
        with open(curpath, 'wb') as fout:
            pickle.dump(mh_res, fout)
        cur_file_idx += 1
        if print_info:
            print(f'obtained {samples_obtained} samples of {total_count}')

    if return_type == 'chain':
        mcmc_res = load_mcmc_result(save_prefix, save_suffix, save_dir)
        return mcmc_res
    else:
        return None


def symmetric_mh_algo(
    startvals, log_probdens, proposal, num_samples,
    thin_step=1, seed=None, print_info=False,
    attempt_resume=False, save_prefix='mh_res_', save_suffix='.pkl',
    save_dir='.', save_batchsize=1000, return_type='chain'
):
    return mh_algo(
        startvals, log_probdens, proposal, num_samples,
        thin_step=thin_step, seed=seed, print_info=print_info,
        attempt_resume=attempt_resume, save_prefix=save_prefix,
        save_suffix=save_suffix, save_dir=save_dir,
        save_batchsize=save_batchsize, return_type=return_type
    )


def _mh_worker(con, mh_args, mh_kwargs):
    mh_res = mh_algo(*mh_args, **mh_kwargs)
    con.send(mh_res)
    con.close()


def parallel_mh_algo(
    num_workers, startvals, log_probdens, proposal,
    num_samples, log_transition_pdf=None, thin_step=1,
    seeds=None, print_info=False, attempt_resume=False,
    save_prefix='mh_res_', save_suffix='.pkl', save_dir='.',
    save_batchsize=1000, return_type='chain'
):
    if seeds is None:
        seeds = np.random.randint(low=0, high=65535, size=num_workers)
    elif len(seeds) != num_workers:
        raise ValueError('number of seed values must equal num_workers')
    mh_args = (startvals, log_probdens, proposal, num_samples)
    mh_kwargs = {'log_transition_pdf': log_transition_pdf,
                 'thin_step': thin_step,
                 'print_info': print_info,
                 'attempt_resume': attempt_resume,
                 'save_dir': save_dir,
                 'save_batchsize': save_batchsize,
                 'return_type': return_type,
                 }
    pipe_parents = []
    pipe_children = []
    procs = []
    for i, seed in zip(range(num_workers), seeds):
        mh_kwargs['seed'] = seed
        mh_kwargs['save_prefix'] = save_prefix + '{:03d}_'.format(i)
        pipe_parent, pipe_child = Pipe()
        pipe_parents.append(pipe_parent)
        pipe_children.append(pipe_child)
        proc = Process(target=_mh_worker,
                       args=(pipe_child, mh_args, mh_kwargs))
        proc.start()
        procs.append(proc)

    mh_res_list = []
    for pipe_parent, proc in zip(pipe_parents, procs):
        mh_res_list.append(pipe_parent.recv())
        proc.join()
    if return_type == 'chain':
        return mh_res_list
    else:
        return None


def parallel_symmetric_mh_algo(
    num_workers, startvals, log_probdens, proposal,
    num_samples, thin_step=1, seeds=None, print_info=True,
    attempt_resume=False, save_prefix='mh_res_', save_suffix='.pkl',
    save_dir='.', save_batchsize=1000, return_type='chain'
):
    return parallel_mh_algo(
        num_workers, startvals, log_probdens, proposal,
        num_samples, thin_step=1, seeds=seeds, print_info=print_info,
        attempt_resume=attempt_resume, save_prefix=save_prefix,
        save_suffix=save_suffix, save_dir=save_dir,
        save_batchsize=save_batchsize, return_type=return_type
    )


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
    raise NotImplementedError("""
    TODO: this function needs to be rewritten to make use of the
          new lm_update function and to account for the changed
          interface of the mh_algo and related functions'
    """)
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
