import  numpy as np
import numpy.random as npr

def sampY(X, Y, Z, epsilon, lmd, p):
    N, T = X.shape
    K = Z.shape[1]

    pYjointprop = np.zeros(2)

    Ysamp = Y.copy()
    for t in range(T):
        for k in range(K):
            for a in range(2):
                Ysamp[k,t] = a

                Ypriorpartl = Ysamp[k,t]*np.log(p)+(1-Ysamp[k,t])*np.log(1-p)

                e = np.matmul(Z, Ysamp[:, t])
                Xpriorpartl = sum(X[:, t] * np.log(1 - ((1 - lmd) ** e) * (1 - epsilon)) \
                                  + (1 - X[:, t]) * np.log(((1 - lmd) ** e) * (1 - epsilon)))

                pYjointprop[a] = Ypriorpartl+Xpriorpartl

            pY0 = 1 / (1 + np.exp(-(pYjointprop[0]-pYjointprop[1])))
            Ysamp[k,t] = pY0 < npr.rand()

    return Ysamp


def sampY_newrows_only(X, Y, Z, epsilon, lmd, p, start_row):

    N, T = X.shape
    K = Z.shape[1]

    pYjointprop = np.zeros(2)

    Ysamp = Y.copy()

    for t in range(T):
        for k in range(start_row, K):
            for a in range(2):
                Ysamp[k,t] = a

                Ypriorpartl = Ysamp[k,t]*np.log(p)+(1-Ysamp[k,t])*np.log(1-p)

                e = np.matmul(Z, Ysamp[:,t])
                Xpriorpartl = sum(X[:,t] * np.log(1-((1-lmd)**e) * (1-epsilon)) \
                    + (1-X[:,t]) * np.log(((1-lmd)**e) *(1-epsilon)))

                pYjointprop[a] = Ypriorpartl+Xpriorpartl

            pY0 = 1/(1 + np.exp(-(pYjointprop[0]-pYjointprop[1])))
            Ysamp[k, t] = pY0 < npr.rand()

    return Ysamp

if __name__ == '__main__':
    sigma_X = 0.5

    p = .2
    epsilon = .0001
    lmd = .5
    alpha = 2.2
    #T = 1000
    #N = 20

    X = np.array([[1,1],[1,0]])
    Z = np.array([[1,0],[0,1]])
    Y = np.array([[1, 0], [0, 1]])

    Ysamp = sampY(X,Y,Z,epsilon,lmd,p)
    #Ysamp = sampY_newrows_only(X,Y,Z,epsilon,lmd,p,0)
    print(Ysamp)