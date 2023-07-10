import numpy as np


def create_relunc_vector(datablock):
    relcovmat = create_relative_datablock_covmat(datablock)
    return np.sqrt(relcovmat.diagonal())


def create_relative_datablock_covmat(datablock):
    if datablock['type'] != 'simple-experiment-datablock':
        raise TypeError(
            f'this function cannot create a relative '
            f'covariance matrix for a datablock of type {datablock["type"]}.')
    if 'relative_covariance_matrix' in datablock:
        relcovmat = np.array(datablock['relative_covariance_matrix'],
                             dtype=np.float64)
    elif 'percentual_uncertainties' in datablock:
        reluncs = np.array(datablock['percentual_uncertainties'],
                           dtype=np.float64) / 100.
        if 'lower_triagonal_correlation_matrix' in datablock:
            tmp = datablock['lower_triagonal_correlation_matrix']
            cormat = np.empty([len(tmp), len(tmp)], dtype=np.float64)
            for i, currow in enumerate(tmp):
                cormat[i, :i+1] = currow
                cormat[:i+1, i] = currow
            cormat /= 100
            relcovmat = cormat * reluncs.reshape(-1, 1) * reluncs.reshape(1, -1)
        else:
            relcovmat = np.diag(np.square(reluncs), k=0)

    return relcovmat
