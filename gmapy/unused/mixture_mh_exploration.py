import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ks_2samp


# define conditional distribution
def get_cond_dist(a, src_idcs):
    a = np.array(a)
    src_idcs = np.array(src_idcs)
    sel = np.full(N, True)
    sel[src_idcs] = False
    tar_idcs = np.where(sel)[0]
    m1 = target_mean[tar_idcs]
    s11 = target_cov[np.ix_(tar_idcs, tar_idcs)]
    s12 = target_cov[np.ix_(tar_idcs, src_idcs)]
    s22 = target_cov[np.ix_(src_idcs, src_idcs)]
    m2 = target_mean[src_idcs]
    cmean1 = m1 + s12 @ np.linalg.inv(s22) @ (a - m2)
    ccov11 = s11 - s12 @ np.linalg.inv(s22) @ s12.T
    return multivariate_normal(mean=cmean1, cov=ccov11)


def sample_gibbs(size):
    smpl = np.full((N, size), 0.)
    curvals = target_mean.copy()
    for i in range(size):
        cdist2 = get_cond_dist(curvals[idcs1], idcs1)
        curvals[idcs2] = cdist2.rvs()
        cdist1 = get_cond_dist(curvals[idcs2], idcs2)
        curvals[idcs1] = cdist1.rvs()
        smpl[:, i] = curvals
    return smpl


def proposal1(x):
    cdist1 = get_cond_dist(x[idcs2], idcs2)
    x = x.copy()
    x[idcs1] = cdist1.rvs()
    return x


def proposal1_pdf(x, propx):
    if not np.all(x[idcs2] == propx[idcs2]):
        return 0.
    cdist1 = get_cond_dist(x[idcs2], idcs2)
    return cdist1.pdf(propx[idcs1])


def proposal2(x):
    cdist2 = get_cond_dist(x[idcs1], idcs1)
    x = x.copy()
    x[idcs2] = cdist2.rvs()
    return x


def proposal2_pdf(x, propx):
    if not np.all(x[idcs1] == propx[idcs1]):
        return 0.
    cdist2 = get_cond_dist(x[idcs1], idcs1)
    return cdist2.pdf(propx[idcs2])


def proposal(x):
    z = np.random.rand(1)
    if rho < z:
        return proposal1(x)
    else:
        return proposal2(x)


def proposal_pdf(x, propx):
    p1 = proposal1_pdf(x, propx)
    p2 = proposal2_pdf(x, propx)
    return rho * p1 + (1-rho) * p2


def sample_mh(size):
    smpl = np.full((N, size), 0.)
    curval = target_mean.copy()
    curprob = target_dist.pdf(curval)
    num_acc = 0
    for i in range(size):
        propval = proposal(curval)
        newprob = target_dist.pdf(propval)
        prob_to_prop = proposal_pdf(curval, propval)
        prob_from_prop = proposal_pdf(propval, curval)
        a1 = prob_from_prop / prob_to_prop
        a2 = newprob / curprob
        a = a1 * a2
        if np.random.rand() < a:
            curval = propval
            curprob = newprob
            num_acc += 1
        smpl[:, i] = curval
    print(f'acceptance rate: {num_acc / size}')
    return smpl


if __name__ == '__main__':
    rho = 0.7
    idcs1 = np.array([0, 2])
    idcs2 = np.array([1])

    # define target distribution
    np.random.seed(38)
    N = 3
    L = np.random.rand(N*N).reshape(N, N)
    target_mean = np.random.rand(N)
    target_cov = L.T @ L
    target_dist = multivariate_normal(mean=target_mean, cov=target_cov)

    # employ different ways of sampling
    print('# employ different sampling techniques')
    print('direct sampling...')
    smpl1 = target_dist.rvs(size=500000).T
    print('Gibbs sampling....')
    smpl2 = sample_gibbs(500000)
    print('Metropolis-Hastings sampling...')
    smpl3 = sample_mh(500000)

    nburn = 5000
    print('# Comparison of mean vectors')
    print(np.mean(smpl1[:, nburn:], axis=1))
    print(np.mean(smpl2[:, nburn:], axis=1))
    print(np.mean(smpl3[:, nburn:], axis=1))
    print('# Comparison of standard deviations')
    print(np.std(smpl1[:, nburn:], axis=1))
    print(np.std(smpl2[:, nburn:], axis=1))
    print(np.std(smpl3[:, nburn:], axis=1))

    print('# Perform Kolmogorov-Smirnov test')
    for i in range(N):
        print(f'- for element at position {i}')
        print('-- result for direct vs Gibbs sampling')
        print(ks_2samp(smpl1[i, nburn::2000], smpl2[i, nburn::2000]))
        print('-- result for direct vs MH sampling')
        print(ks_2samp(smpl1[i, nburn::2000], smpl3[i, nburn::2000]))
