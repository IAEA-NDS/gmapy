import numpy as np
from scipy.sparse import csr_matrix


SHAPE_MT_IDS = (2,4,8,9)


def return_matrix(idcs1, idcs2, vals, dims, how):
    """Return a matrix defined by triples in desired format."""
    Sdic = {'idcs1': np.array(idcs1, dtype=int),
            'idcs2': np.array(idcs2, dtype=int),
            'x': np.array(vals, dtype=float)}
    if how == 'csr':
        S = csr_matrix((Sdic['x'], (Sdic['idcs2'], Sdic['idcs1'])), shape=dims)
        return S
    elif how == 'dic':
        return Sdic
    else:
        raise ValueError('invalid value of parameter "how"')



#     Numerical Jacobian using Richardson extrapolation
#     Jay Kahn - University of Rochester, November 19, 2012
#     f - function to take derivative over
#     x - point (vector) at which to evaluate derivatives
#     o - order of error term desired relative to f
#     h1 - starting factor
#     *control - any extra arguments to be passed to f
def richardson(f, x0, o, h1, v, *control):
    x=np.array(x0)
    d=x.shape[0]
    i=1
    r=o/2
    while i <= d:
        j=1
        while j <= r:
            if j==1:
                h=h1
            else:
                h=h/v

            idd = np.zeros(d)
            idd[i-1] = h
            xup=x+idd
            xdown=x-idd
            fat=f(x,*control)
            fup=f(xup,*control)
            fdown=f(xdown,*control)
            ddu=fup-fat
            ddd=fdown-fat
            hp=h
            if j==1:
                dds=np.array([ddu, ddd])
                hhs=np.array([[hp, -hp]])
            else:
                dds=np.concatenate((dds, np.array([ddu, ddd])),0)
                hhs=np.concatenate((hhs, np.array([[hp, -hp]])),1)
            j=j+1

        mat=hhs
        j=2
        while j<=o:
            mat=np.concatenate((mat, np.power(hhs,j)/np.math.factorial(j)),0)
            j=j+1

        der = np.dot(np.transpose(np.linalg.inv(mat)),dds)

        if i==1:
            g=der
        else:
            g=np.concatenate((g,der),1)

        i=i+1
    return g



#     Jacobian running as shell of Richardson. Ends up with matrix
#     whose rows are derivatives with respect to different elements
#     of x and columns are derivatives of different elements of f(x).
#     For scalar valued f(x) simplifies to column gradient.
#     Jay Kahn - University of Rochester, November 19, 2012
#     f - function to take derivative over
#     x - point (vector) at which to evaluate derivatives
#     o - order of error term desired relative to f
#     h1 - starting factor
#     *control - any extra arguments to be passed to f
def numeric_jacobian(f, x0, o=4, h1=0.01, v=2, *control):
    fn=f(x0,*control).shape[0]
    x=np.array(x0)
    xn=x.shape[0]
    J=np.zeros((xn,fn))
    g=richardson(f, x, o, h1, v, *control)
    j=0
    while j<=xn-1:
        i=0
        while i<=fn-1:
            J[j,i]=g[0,i+j*fn]
            i=i+1
        j=j+1
    return J.T

