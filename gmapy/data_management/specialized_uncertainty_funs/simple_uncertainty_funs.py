import numpy as np



def create_relunc_vector(datablock):
    relcovmat = create_relative_datablock_covmat(datablock)
    return relcovmat.diagonal()



def create_relative_datablock_covmat(datablock):
    if datablock['type'] != 'simple-experiment-datablock':
        raise TypeError(f'this function cannot create a relative '
                'covariance matrix for a datablock of type {datablock["type"]}.')

    relcovmat = np.array(datablock['relative_covmat'])
    return relcovmat
