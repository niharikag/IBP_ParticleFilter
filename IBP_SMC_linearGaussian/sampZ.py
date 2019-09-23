import  numpy as np
import numpy.random as npr
from numpy.linalg import inv, det, pinv
from scipy.special import gammaln
from math import log, pi
from utils import inv_logit, clean

def sampZ(X, Z, sigma_x, sigma_A, alpha):
    K = Z.shape[1]
    N, D = X.shape
    Zsamp = Z.copy()

    for i in range(N):
        for k in range(K):
            Zsamp = sampleIndividualEntry(X, Zsamp, sigma_x, sigma_A, i, k)

        Zsamp = sampleNewColumns(X, Zsamp, sigma_x, sigma_A, alpha, i)

        Zsamp = clean(Zsamp)
        K = Zsamp.shape[1]

    return Zsamp


def sampleIndividualEntry(X, Zsamp, sigma_x, sigma_A, i, k):
    Z_temp = np.delete(Zsamp, i, axis=0)
    mk = np.sum(Z_temp, axis=0)[k]

    Z_temp = np.copy(Zsamp)
    K = Zsamp.shape[1]
    N, D = X.shape

    if mk > 0:
        pi_k = mk / N
        log_r_p = np.log(pi_k / (1 - pi_k))

        Z_temp[i, k] = 0
        Z_T_Z = np.matmul(Z_temp.T, Z_temp)
        temp_z = Z_T_Z + (sigma_x ** 2) / (sigma_A ** 2) * np.eye(K)
        try:
            z_inv = inv(temp_z)
        except:
            z_inv = pinv(temp_z)

        logZpart = (D / 2) * log(det(temp_z))
        temp_z = np.matmul(np.matmul(Z_temp, z_inv), Z_temp.T)
        temp_z = np.eye(N) - temp_z
        logExp = -1 / (2 * sigma_x ** 2) * np.trace(np.matmul(np.matmul(X.T, temp_z), X))

        log_l_0 = logExp - logZpart

        Z_temp[i, k] = 1
        Z_T_Z = np.matmul(Z_temp.T, Z_temp)
        temp_z = Z_T_Z + (sigma_x ** 2) / (sigma_A ** 2) * np.eye(K)
        z_inv = inv(temp_z)

        logZpart = (D / 2) * log(det(temp_z))
        temp_z = np.matmul(np.matmul(Z_temp, z_inv), Z_temp.T)
        temp_z = np.eye(N) - temp_z
        logExp = -1 / (2 * sigma_x ** 2) * np.trace(np.matmul(np.matmul(X.T, temp_z), X))

        log_l_1 = logExp - logZpart

        logrprop = log_r_p + log_l_1 - log_l_0
        probzis1 = inv_logit(logrprop)
        Z_temp[i, k] = npr.uniform() < probzis1
    else:
        pass
        #print("do nothing")
    return Z_temp


def sampleNewColumns(X, Zsamp, sigma_x, sigma_A, alpha, i):
    Z_temp = np.delete(Zsamp, i, axis=0)
    m = np.sum(Z_temp, axis=0)

    Z_temp = np.copy(Zsamp)
    K = Zsamp.shape[1]
    N, D = X.shape

    for k_indx, m_item in enumerate(m):
        if m_item == 0:
            Z_temp[i, k_indx] = 0

    cdfr = npr.rand()

    lpnewK = np.zeros(10)

    for newK in range(10):
        Z_T_Z = np.matmul(Z_temp.T, Z_temp)
        temp_z = Z_T_Z + (sigma_x ** 2) / (sigma_A ** 2) * np.eye(K)

        logZ = (N * D / 2) * log(2 * pi) + ((N - K) * D) * log(sigma_x) + \
               (K * D) * log(sigma_A) * (D / 2) * log(det(temp_z))

        z_inv = inv(temp_z)
        temp_z = np.matmul(np.matmul(Z_temp, z_inv), Z_temp.T)
        temp_z = np.eye(N) - temp_z
        logExp = -1 / (2 * sigma_x ** 2) * np.trace(np.matmul(np.matmul(X.T, temp_z), X))

        logLike = logExp - logZ

        lpnewK[newK] = logLike - alpha / N + (K + newK) * log(alpha / N) - gammaln(K + newK + 1)

    logmax = max(lpnewK)
    pdf = np.exp(lpnewK - logmax)
    cdf = np.cumsum(pdf/sum(pdf))
    newK = len(np.where(cdf<cdfr)[0])

    print(newK)
    if newK > 0:
        Z_temp = np.column_stack((Z_temp, np.zeros((len(Z_temp), newK))))
        Z_temp[i, K:] = 1

    return Z_temp


if __name__ == '__main__':
    #X, Z, A, sigma_X = generate_test_data(10)
    # print(X.shape)
    # print(Z)
    # lp = logPX(X, Z, sigma_X, 1)
    # print(lp)


    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,1]])
    A = np.array([[1, 0], [0, 1]])
    Zsamp = sampZ(X, Z, 0.5, 1, 4)

    print(Zsamp)
    #lp = logPX(X, Z, sigma_X, 1)
    #print(lp)
    #lpz = logPXZ(X, Z, sigma_X, 1, 4)
    #print(lpz)


    # Zparticles = particle_filter(X, 10, 4, sigma_X, 1)
    # print(Zparticles[0])

    # for matlab
    #X = [[1, 1, 1]; [1, 1, 1]]
    #Z = [[1, 0]; [0, 1]]