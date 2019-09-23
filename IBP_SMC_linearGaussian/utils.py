import  numpy as np
import numpy.random as npr
from numpy.linalg import inv, det
from scipy.special import gammaln


def multinomial_resampling(weights):
    u = npr.rand(len(weights))
    bins = np.cumsum(weights)
    return np.digitize(u,bins).astype(int)


def clean(Z):
    """
    Remove the empty columns form Z
    :param Z:
    return: cleaned Z
    """
    columncounts = np.sum(Z, axis=0)
    nzc = np.where(columncounts>0)[0]
    retZ = Z[:, nzc]

    return retZ


def logPX(X, Z, sigma_x, sigma_A):
    Z_plus = Z
    K_plus = Z.shape[1]
    N, D = X.shape

    Z_T_Z = np.matmul(Z_plus.T, Z_plus)
    logZ = (N * D / 2) * np.log(2 * np.pi) + ((N - K_plus) * D) * np.log(sigma_x) + (K_plus * D) * np.log(sigma_A) + \
           (D / 2) * np.log(det(Z_T_Z + (sigma_x ** 2) / (sigma_A ** 2) * np.eye(K_plus)))

    inv_z = inv(Z_T_Z + (sigma_x ** 2) / (sigma_A ** 2) * np.eye(K_plus))
    inv_z = np.matmul(Z_plus, inv_z)
    inv_z = np.matmul(inv_z, Z_plus.T)
    inv_z = np.eye(N) - inv_z
    # I−Z+(ZT+Z+σ2Xσ2YIK+)−1ZT
    temp_z = np.matmul(X.T, inv_z)
    temp_z = np.matmul(temp_z, X)
    logExp = -1 / (2 * sigma_x ** 2) * np.trace(temp_z)

    logLike = logExp - logZ

    lP = logLike
    return lP


def logPXZ(X,Z,sigma_x, sigma_A, alpha):
    Z_plus = Z
    K_plus = Z.shape[1]
    N, D = X.shape

    m_k = np.sum(Z, axis=0)
    HN = np.sum(1 / np.array(range(1, N + 1)))

    logZPrior = K_plus*np.log(alpha) -alpha*HN + sum(gammaln(m_k)+gammaln(N-m_k+1)-gammaln(N+1))

    z_temp = np.matmul(Z_plus.T, Z_plus) + (sigma_x**2)/(sigma_A**2)*np.eye(K_plus)

    logZ = (N*D/2) * np.log(2*np.pi) + ((N-K_plus)*D) * np.log(sigma_x) +  (K_plus * D) * np.log(sigma_A) + \
    (D/2)* np.log (det(z_temp))
    z_temp = np.eye(N)- np.matmul(np.matmul(Z_plus, inv(z_temp)), Z_plus.T)
    logExp = -1/(2*sigma_x**2) * np.trace(np.matmul(np.matmul(X.T, z_temp), X) )
    logLike = logExp - logZ

    lP = logLike + logZPrior

    return lP


def resample(Zparticles, w, N):
    retZparticles = []
    #indexes = np.zeros_like(Zparticles)
    cdf = np.cumsum(w)
    p = npr.rand(N)
    p = np.sort(p)
    
    picked = np.zeros_like(Zparticles)
    for i in range(N):
        pind = np.where(cdf > p[i])[0][0]
        picked[pind] = picked[pind] + 1

    picked = picked.astype(int)
    for i in range(N):
        if picked[i] > 0:
            for j in range(picked[i]):
                retZparticles.append(Zparticles[i])

    return retZparticles


def inferstats(Zsamples, Z, start):

    ksum = Zsamples[start].shape[1]
    ZZt = np.matmul(Zsamples[start], Zsamples[start].T)
    num_samples = len(Zsamples)

    for i in range(start+1, num_samples):
        ksum = ksum + Zsamples[i].shape[1]
        ZZt = ZZt + np.matmul(Zsamples[i], Zsamples[i].T)

    Ek = ksum / (num_samples - start)
    EZZt = ZZt / (num_samples - start)


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

def inv_logit(log_p):
    value = 0
    if log_p > 0:
        value = 1. / (1. + np.exp(-log_p))
    elif log_p <= 0:
        value = np.exp(log_p) / (1 + np.exp(log_p))

    return value