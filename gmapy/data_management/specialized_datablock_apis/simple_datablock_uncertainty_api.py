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
        reluncs = np.array('percentual_uncertainties', dtype=np.float64) / 100.
        cormat = np.array(datablock['correlation_matrix'], dtype=np.float64)
        relcovmat = (cormat * reluncs.reshape(-1, 1)) * reluncs.reshape(1, -1)
    return relcovmat
