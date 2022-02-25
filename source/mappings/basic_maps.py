import numpy as np
from scipy.sparse import csr_matrix



def get_sensmat_exact(ens1, ens2, idcs1=None, idcs2=None):
    """Compute sensitivity matrix to map
    values given on energy mesh ens1 to
    the mesh given by ens2. It is assumed
    that the energies in ens2 are a 
    subset of the energies in ens1."""
    ens1 = np.array(ens1)
    ens2 = np.array(ens2)
    ord = np.argsort(ens1)
    ens1 = ens1[ord]
    ridcs = np.searchsorted(ens1, ens2, side='left')
    if np.any(ridcs >= len(ens1)):
        raise ValueError('too large index')

    if not np.all(ens1[ridcs] == ens2):
        raise ValueError('mismatching energies encountered' +
                str(ens1[ridcs]) + ' vs ' + str(ens2))

    curidcs2 = np.arange(len(ens2))
    curidcs1 = ord[ridcs]
    coeff = np.ones(len(ens2))
    if idcs1 is not None:
        curidcs1 = idcs1[curidcs1]
    if idcs2 is not None:
        curidcs2 = idcs2[curidcs2]
    return {'idcs1': np.array(curidcs1, dtype=int),
            'idcs2': np.array(curidcs2, dtype=int),
            'x': np.array(coeff, dtype=float)}



def propagate_exact(ens1, vals1, ens2):
    """Propagate values vals1 given on
    energy mesh ens1 to the mesh given
    by ens2. It is assumed that ens2 is
    a subset of ens1."""
    Sraw = get_sensmat_exact(ens1, ens2)
    S = csr_matrix((Sraw['x'], (Sraw['idcs2'], Sraw['idcs1'])),
              shape = (len(ens2), len(ens1)))
    return S @ vals1



def propagate_fisavg(ens, vals, ensfis, valsfis):
    ord = np.argsort(ens)
    ens = ens[ord]
    vals = vals[ord]
    ordfis = np.argsort(ensfis)
    ensfis = ensfis[ordfis]
    valsfis = valsfis[ordfis]

    # we skip one element, because all energy meshes contain
    # as lowest energy 2.53e-8 MeV
    fidx = np.searchsorted(ensfis[1:], ens[1]) + 1

    lentmp = len(ensfis)
    ensfis = np.concatenate([ensfis, np.full(100, 0.)])
    valsfis = np.concatenate([valsfis, np.full(100, 0.)])

    uhypidx = fidx+len(ens)-1
    urealidx = min(fidx+len(ens)-1, len(ensfis))
    ulimidx2 = len(ens) - (uhypidx - urealidx)
    # TODO: This check fails for the 9Pu(n,f) cross section fission average
    #       because the energy 0.235 MeV is missing in 9Pu(n,f) but present
    #       in the fission spectrum
    # if not np.all(np.isclose(ens[1:ulimidx2], ensfis[fidx:urealidx], atol=0, rtol=1e-05)):
    #   raise ValueError('energies of excitation function and fission spectrum do not match')

    fl = 0.
    sfl = 0.
    for i in range(1, ulimidx2-1):
        fl = fl + valsfis[fidx-1+i]
        el1 = 0.5 * (ens[i-1] + ens[i])
        el2 = 0.5 * (ens[i] + ens[i+1])
        de1 = 0.5 * (ens[i] - el1)
        de2 = 0.5 * (el2 - ens[i])
        ss1 = 0.5 * (vals[i] + 0.5*(vals[i-1] + vals[i]))
        ss2 = 0.5 * (vals[i] + 0.5*(vals[i] + vals[i+1]))
        cssli = (ss1*de1 + ss2*de2) / (de1+de2)
        sfl = sfl + cssli*valsfis[fidx-1+i]

    fl = fl + valsfis[0] + valsfis[urealidx-1]
    sfl = sfl + valsfis[0]*vals[0] + valsfis[urealidx-1]*vals[-1]
    sfis = sfl / fl

    if not np.isclose(1., fl, atol=0, rtol=1e-4):
        print('fission normalization: ' + str(fl))
        raise ValueError('fission spectrum not normalized')

    return sfis



def get_sensmat_fisavg(ens, vals, ensfis, valsfis):
    ord = np.argsort(ens)
    ens = ens[ord]
    vals = vals[ord]
    ordfis = np.argsort(ensfis)
    ensfis = ensfis[ordfis]
    valsfis = valsfis[ordfis]

    # we skip one element, because all energy meshes contain
    # as lowest energy 2.53e-8 MeV
    fidx = np.searchsorted(ensfis[1:], ens[1]) + 1

    uhypidx = fidx+len(ens)-1
    urealidx = min(fidx+len(ens)-1, len(ensfis))
    ulimidx2 = len(ens) - (uhypidx - urealidx)
    # TODO: This check fails for the 9Pu(n,f) cross section fission average
    #       because the energy 0.235 MeV is missing in 9Pu(n,f) but present
    #       in the fission spectrum
    # if not np.all(np.isclose(ens[1:ulimidx2], ensfis[fidx:urealidx], atol=0, rtol=1e-05)):
    #     raise ValueError('energies of excitation function and fission spectrum do not match')

    lentmp = len(ensfis)
    ensfis = np.concatenate([ensfis, np.full(100, 0.)])
    valsfis = np.concatenate([valsfis, np.full(100, 0.)])

    sensvec = np.full(len(ens), 0., dtype=float)

    fl = 0.
    for i in range(0, len(ens)): 
        fl = fl + valsfis[fidx+i]
        if i == 0 or i == len(ens)-1:
            cssj = vals[i]
        else:
            el1 = 0.5 * (ens[i-1] + ens[i])
            el2 = 0.5 * (ens[i] + ens[i+1])
            de1 = 0.5 * (ens[i] - el1)
            de2 = 0.5 * (el2 - ens[i])
            ss1 = 0.5 * (vals[i] + 0.5*(vals[i-1] + vals[i]))
            ss2 = 0.5 * (vals[i] + 0.5*(vals[i] + vals[i+1]))
            cssj = (ss1*de1 + ss2*de2) / (de1+de2)

        sensvec[i] = valsfis[fidx+i-1] * cssj / vals[i]

        #     # TODO: This is probably the correct way of calculating
        #     #       the sensitivity matrix.
        #     # ss1di = 0.75
        #     # ss1dim1 = 0.25
        #     # ss1dip1 = 0. 
        #     # ss2di = 0.75
        #     # ss2dip1 = 0.25
        #     # ss2dim1 = 0
        #     # coeff1 = de1/(de1+de2) * valsfis[fidx+i]
        #     # coeff2 = de2/(de1+de2) * valsfis[fidx+i]
        #     # sensvec[i] += ss1di * coeff1 + ss2di * coeff2
        #     # sensvec[i-1] += ss1dim1 * coeff1
        #     # sensvec[i+1] += ss2dip1 * coeff2

    # sensvec /= fl

    sensvec[ord] = sensvec.copy()
    return sensvec

