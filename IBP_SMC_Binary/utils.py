import  numpy as np
import numpy.random as npr
from scipy.special import gammaln


def nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n/k * nchoosek(n-1, k-1)
    return round(r)


def multinomial_resampling(weights, N=None):
    if N is None:
        N = len(weights)
    u = npr.rand(N)
    bins = np.cumsum(weights)
    return np.digitize(u,bins).astype(int)

# log prior on Z: IBP prior
def logPZ(Z, alpha):
    N, K = Z.shape

    HN = np.sum(1 / np.array(range(1, N + 1)))
    m = np.sum(Z, axis=0)
    lP = K * np.log(alpha) -alpha*HN + sum(gammaln(m) + gammaln(N-m+1) - gammaln(N+1))

    return lP


def logPXYZ(X, Y, Z, alpha, epsilon, lmd):
    # we don't want to include lPY, since Y is a big infinite matrix anyway
    #nY = np.prod(Y.shape)
    #nY1 = len(Y[Y == 1])
    #nY0 = nY - nY1
    #lPY = nY1 * log(p) + nY0 * log(1 - p)

    lPZ = logPZ(Z, alpha)
    ZY = np.matmul(Z, Y)

    lPX = np.sum(X * np.log(1 - (1 -lmd)**ZY * (1 - epsilon)) + (1 - X) * np.log((1 -lmd)**ZY * (1 - epsilon)) )
    lP = lPX + lPZ

    return lP


def inferstats(Zsamples, Z=None, start=0):

    ksum = Zsamples[start].shape[1]
    ZZt = np.matmul(Zsamples[start], Zsamples[start].T)
    num_samples = len(Zsamples)

    for i in range(start+1, num_samples):
        ksum = ksum + Zsamples[i].shape[1]
        ZZt = ZZt + np.matmul(Zsamples[i], Zsamples[i].T)

    Ek = ksum / (num_samples - start)
    EZZt = ZZt / (num_samples - start)

    if Z is not None:
        truth = np.matmul(Z, Z.T)

        in_degree_error = sum(abs(np.diag(truth) - np.diag(EZZt)))
        structure_error = sum(sum(abs(np.triu(truth, 1) - np.triu(EZZt, 1))))

        in_degree_error_ratio = in_degree_error / sum(np.diag(truth))
        total_link_measure = (sum(sum(np.triu(truth, 1))))
        if total_link_measure !=0:
            structure_error_ratio = structure_error / total_link_measure
        else:
            structure_error_ratio = structure_error

        return Ek, EZZt, in_degree_error, structure_error, in_degree_error_ratio, structure_error_ratio

    return Ek, EZZt


def inv_logit(log_p):
    value = 0
    if log_p > 0:
        value = 1. / (1. + np.exp(-log_p))
    elif log_p <= 0:
        value = np.exp(log_p) / (1 + np.exp(log_p))

    return value


def cannonize(Z, Y):
    """
    convert the matrix Z into LOF form, according arrange the rows of Y
    :param Z:
    :param Y:
    :return: LOF of Z, and Y
    """
    i = np.array(range((len(Z) - 1), -1, -1))
    p = 2 ** (i)
    sv = np.matmul(p, Z)

    i = np.argsort(sv)[::-1]

    retZ = Z[:, i]
    retY = Y[i, :]
    return retZ, retY


def clean(Z, Y):
    """
    Remove the empty columns form Z, according arrange the rows of Y
    :param Z:
    :param Y:
    :return: cleaned Z and Y
    """
    columncounts = np.sum(Z, axis=0)
    nzc = np.where(columncounts>0)[0]
    retZ = Z[:, nzc]
    retY = Y[nzc, :]
    return retZ, retY


if __name__ == '__main__':
    w = [.1, .4, .5]
    resamp = multinomial_resampling(w, 1)
    print(resamp)
    # code for unit test
    X = np.array([[1, 1], [1, 1]])
    Z = np.array([[0, 1], [1, 0]])
    Y = np.array([[1, 1], [0, 0]])
    Z, Y = cannonize(Z, Y)
    #print(Z)
    Z, Y = clean(Z, Y)
    #print(Z)
    p = .2
    epsilon = .0001
    lmd = .5
    alpha = 2.2
    lpz = logPXYZ(X, Y, Z, alpha, epsilon, lmd)
    print(lpz)
    lpz = logPZ(Z, alpha)
    print(lpz)