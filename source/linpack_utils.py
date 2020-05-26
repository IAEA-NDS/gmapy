import numpy as np


def pack_symmetric_matrix(mat):
    assert len(mat.shape)==2
    assert mat.shape[0] == mat.shape[1]
    N = mat.shape[0]
    numel = N*(N+1)//2
    res = np.zeros(numel, dtype='float64', order='F')
    k = 0
    for i in range(mat.shape[1]):
        k = k + i
        res[k:(k+i+1)] = mat[0:(i+1),i]
    return res


def unpack_symmetric_matrix(mat):
    numel = len(mat)
    N = (int(np.sqrt(1+8*numel))-1)//2
    res = np.zeros((N,N), dtype='float64', order='F')
    k = 0
    for i in range(N):
        k = k+i
        res[:(i+1),i] = mat[k:(k+i+1)]
        res[i,:(i+1)] = res[:(i+1),i]
    return res


def unpack_utriang_matrix(mat):
    numel = len(mat)
    print('numel: ' + str(numel))
    N = (int(np.sqrt(1+8*numel))-1)//2
    print('N: ' + str(N))
    res = np.zeros((N,N), dtype='float64', order='F')
    k = 0
    for i in range(N):
        k = k+i
        res[:(i+1),i] = mat[k:(k+i+1)]
    return res

