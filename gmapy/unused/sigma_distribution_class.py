from scipy.special import gammaincc, gamma
from scipy.stats import invgamma, ks_2samp
import numpy as np
from ..mappings.helperfuns import numeric_jacobian


class SigmaDist:

    def __init__(self, a=1, n=10):
        self.__a = a
        self.__n = n
        self.__norm = 1.
        self.__xmax, self.__norm = self._determine_normalization()

    def probfun(self, x):
        a = self.__a
        n = self.__n
        if x == 0:
            return 0.
        res = 1 / (x**n) * np.exp(-a / (x**2))
        res /= self.__norm
        return res

    def distfun(self, x):
        a = self.__a
        n = self.__n
        if x == 0.:
            return 0.
        res = 0.5 * x**(1-n)
        res *= (a/x**2)**(0.5-0.5*n)
        res *= gammaincc((n-1)/2, a/(x**2))
        res *= gamma((n-1)/2)
        res /= self.__norm
        return res

    def quantfun(self, y, eps=1e-8):
        if y < 0. or y > 1.:
            raise ValueError('y must be between 0 and 1')
        xmin, xmax = self._find_interval(
            self.distfun, y, 0., self.__xmax, eps=1e-3
        )
        xcur = (xmin + xmax) / 2.
        steps = 0
        while True:
            steps += 1
            xnew = xcur + (y-self.distfun(xcur)) / self.probfun(xcur)
            if np.abs(xnew - xcur) < eps:
                break
            if xnew < xmin:
                xnew = xmin
            elif xnew > xmax:
                xnew = xmax
            xcur = xnew
        return xnew

    def sample(self, n=1):
        u = np.random.rand(n)
        res = np.empty(n, dtype=float)
        for i in range(n):
            res[i] = self.quantfun(u[i])
        return res

    def _find_interval(self, fun, y, xmin, xmax, eps=1e-2, maxsteps=100):
        steps = 0
        while xmax - xmin > eps and steps < maxsteps:
            steps += 1
            xcur = (xmax + xmin) / 2
            curval = fun(xcur)
            if curval <= y:
                xmin = xcur
            elif curval > y:
                xmax = xcur
        return xmin, xmax

    def _determine_normalization(self, abseps=1e-14):
        xcur = 1.
        curval = self.distfun(xcur)
        for i in range(100):
            xnew = xcur * 10
            newval = self.distfun(xnew)
            if np.abs(curval - newval) < abseps:
                break
            curval = newval
            xcur = xnew
        return xnew, newval


if __name__ == '__main__':
    n = 12
    a = 25 

    dist = SigmaDist(a=a, n=n)
    dist.probfun(3)

    # testing that derivative of distribution function is probability density function
    print('testing numerical derivatives...')
    print(numeric_jacobian(dist.distfun, np.array([2.])) / dist.probfun(2))
    print(numeric_jacobian(dist.distfun, np.array([4.])) / dist.probfun(4))
    print(numeric_jacobian(dist.distfun, np.array([10.])) / dist.probfun(10))

    # testing that quantile function is indeed inverse of distribution function
    print('testing quantile function...')
    x = dist.quantfun(0.89)
    y = dist.distfun(x)
    print(f'distfun(quantfun(0.89)) = {y}')

    # generating two sets of samples from the two distributions
    smpl = dist.sample(100000)
    alpha = n/2 - 1/2
    beta = a
    rv = invgamma(alpha)
    smpl2 = rv.rvs(size=100000)
    tsmpl2 = np.sqrt(smpl2 * beta)

    # checking of the moments
    print('result of Kolmogorov-Smirnov two sample test')
    np.mean(smpl) - np.mean(tsmpl2)
    np.std(smpl) - np.std(tsmpl2)
    print(ks_2samp(smpl, tsmpl2))
