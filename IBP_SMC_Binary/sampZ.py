import numpy as np
import numpy.random as npr
from scipy.special import gammaln
from utils import inv_logit, nchoosek, clean
from numpy.matlib import repmat

def sampZ(X, Y, Z, alpha, epsilon, lmd, p, Temp=1):
    K = Z.shape[1]
    N, T = X.shape
    Zsamp = Z.copy()
    Ysamp = Y.copy()

    for i in range(N):
        for k in range(K):
            Zsamp, Ysamp = sampleIndividualEntry(X, Zsamp, Ysamp, epsilon,lmd, i, k)

        Zsamp, Ysamp = sampleNewColumns(X, Zsamp, Ysamp, alpha, epsilon, lmd, p, i)

        Zsamp, Ysamp = clean(Zsamp, Ysamp)

        K = Zsamp.shape[1]

    return Zsamp, Ysamp


def sampleIndividualEntry(X, Zsamp, Ysamp, epsilon,lmd, i, k):
    Z_temp = np.delete(Zsamp, i, axis=0)
    mk = np.sum(Z_temp, axis=0)[k]

    N, K = Zsamp.shape
    Z_temp = np.copy(Zsamp)

    if mk > 0:
        pi_k = mk / N
        log_r_p = np.log(pi_k / (1 - pi_k))

        Z_temp[i, k] = 0

        e = np.matmul(Z_temp[i, 0:K], Ysamp[0:K, :])
        log_l_0 = sum(X[i,:] * np.log(1 - ((1 - lmd) ** e) * (1 - epsilon)) \
                          + (1 - X[i, :]) * np.log(((1 - lmd) ** e) * (1 - epsilon)))

        Z_temp[i, k] = 1
        e = np.matmul(Z_temp[i, 0:K], Ysamp[0:K, :])
        log_l_1 = sum(X[i, :] * np.log(1 - ((1 - lmd) ** e) * (1 - epsilon)) \
                          + (1 - X[i, :]) * np.log(((1 - lmd) ** e) * (1 - epsilon)))

        logrprop = log_r_p + log_l_1 - log_l_0
        probzis1 = inv_logit(logrprop)
        Z_temp[i, k] = npr.uniform() < probzis1
    else:
        pass
        #print("do nothing")
    return Z_temp, Ysamp


def sampleNewColumns(X, Zsamp, Ysamp, alpha, epsilon, lmd, p, i):
    K = Zsamp.shape[1]
    N, T = X.shape

    Z_temp = np.delete(Zsamp, i, axis=0)

    m = np.sum(Z_temp, axis=0)
    Z_temp = np.copy(Zsamp)

    for k_indx, m_item in enumerate(m):
        if m_item == 0:
            Z_temp[i, k_indx] = 0

    cdfr = npr.rand()
    e = np.matmul(Z_temp[i, 0:K], Ysamp[0:K, :])
    oneinds = np.ravel(np.argwhere(X[i,:] == 1))
    zeroinds = np.setdiff1d(np.array(range(T)), oneinds)

    lpnewK = np.zeros(10)

    for newK in range(10):
        lpXiT = np.sum(np.log(1 - (1 - epsilon) * ((1 -lmd)** e[oneinds]) * ((1 -lmd * p)**newK)))
        lpXiT = lpXiT + np.sum(np.log((1 - epsilon) * ((1 -lmd)** e[zeroinds]) * ((1 -lmd * p)** newK)))
        lpnewK[newK] = lpXiT - alpha / N + (K + newK) * np.log(alpha / N) - gammaln(K + newK + 1)

    logmax = max(lpnewK)
    #print(logmax)
    pdf = np.exp(lpnewK - logmax)
    #pdf = pdf / sum(pdf)
    #cdf = pdf[0]
    #newK = 0
    #ii = 0
    cdf = np.cumsum(pdf/sum(pdf))
    #cdf = np.sort(cdf)
    newK = len(np.where(cdf<cdfr)[0])
    #while cdf < cdfr:
    #    ii = ii + 1
    #    cdf = cdf + pdf[ii]
    #    newK = newK + 1

    print(newK)
    if newK:
        #kplus = Z_temp.shape[1]
        Z_temp = np.column_stack((Z_temp, np.zeros((len(Z_temp), newK))))
        Z_temp[i, K:] = 1
        #  the new values of Y should be drawn jointly from their
        #  posterior distribution given Z
        Ysampnew = np.row_stack((Ysamp, np.zeros((newK, Ysamp.shape[1]))))

        newprobs = np.zeros((T, newK + 1))
        newprobs = ((1 - epsilon) * (1 -lmd)** (repmat(e, newK + 1, 1) + \
                    repmat( np.array(range(newK+1)).reshape(-1,1), 1,T)) )
        newprobs[:, oneinds] = 1 - newprobs[:, oneinds]

        temp_prob = np.zeros((newK+1, T))
        for temp_k in range(newK+1):
            temp = nchoosek(newK, temp_k) * (p ** temp_k) * ((1 - p) ** (newK - temp_k))
            temp_prob[temp_k, :] = repmat(temp, 1, T)

        print(newprobs.shape)
        print(temp_prob.shape)
        newprobs = np.matmul(newprobs.T, temp_prob)
        #newprobs = newprobs** Temp

        newprobs = newprobs / repmat(np.sum(newprobs, axis=0), len(newprobs), 1)
        newprobs = np.cumsum(newprobs, axis=0)

        for j in range(T):
            m = min(np.where(npr.rand() < newprobs[:,j])[0])
            Ysampnew[len(Ysampnew) - int(m): , j] = 1

        Ysamp = Ysampnew

    return Z_temp, Ysamp


if __name__ == '__main__':
    p=.2
    mat = np.zeros((3, 10))
    for k in range(2+1):
        temp = nchoosek(2, k)* p**k * (1-p)**(2-k)
        mat[k,:] = repmat(temp, 1,10)

    #print(mat.shape)

    #print(repmat(np.sum(mat, axis=0), 2 + 1, 1))

    #print(nchoosek(2, k))
    #X, Z, A, sigma_X = generate_test_data(10)
    # print(X.shape)
    # print(Z)
    # lp = logPX(X, Z, sigma_X, 1)
    # print(lp)

    p = .2
    epsilon = .0001
    lmd = .5
    alpha = 2.2

    X = np.array([[1,1],[1,0]])
    Z = np.array([[1,0],[0,0]])
    #Z = np.array([[1], [0]])
    Y = np.array([[1, 1], [0, 1]])
    Zsamp, Ysamp = sampZ(X, Y, Z,alpha,epsilon,lmd,p, Temp=1)


    print(Zsamp)
    print(Ysamp)
