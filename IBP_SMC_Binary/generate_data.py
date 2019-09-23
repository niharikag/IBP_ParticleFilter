import  numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from utils import cannonize


def ibp_generate(N, alpha, K_specified):
    """
    :param N: 
    :param alpha: 
    :param K_specified: 
    :return: Z, K
    It produces a  matrix Z according to the Indian Buffet Process (with the dirichlet
    hyperparameter alpha) by generated a bunch of random matrixes until
    one of the correct dimensionality K_specified pops up.  if K_specified
    isn't specified then the function returns a matrix of output
    dimensionality N but random K.  K_specified = -1 means unspecified.
 
    """
    tempK = 100
    K = np.inf

    while (K != K_specified):
        Z = np.zeros((N, tempK))

        nfirst = npr.poisson(alpha, size=1)
        if nfirst == 0:
            continue

        nfirst = int(nfirst)
        Z[0, 0:nfirst] = 1
        m = np.sum(Z, axis=0)
        max_c = nfirst

        for i in range(1, N):
            rns = npr.rand(tempK)
            pr = m / (i+1)
            Z[i,:] = pr > rns

            k_more = int(npr.poisson(alpha / (i+1), size=1))
            if k_more >= 1:
                Z[i, max_c:(max_c + k_more)] = 1

            max_c = int(max_c + k_more)
            m = np.sum(Z, axis=0)

        K = max_c

        Z = Z[:, : K]

        if K_specified == -1:
            break

    return Z


def plot_ibp_matrices(X,Z,Y):

    plt.subplot(1,3,1)
    plt.imshow(X)
    plt.title('X ')
    #plt.ylim((0,100))
    plt.subplot(1,3,2)
    plt.imshow(Z)
    plt.title('Z ')

    plt.subplot(1,3,3)
    plt.imshow(Y)
    plt.title('Y ')

    plt.show()

def generate_test_data(N, T, alpha, lmd, epsilon, p):
    K_specified = -1

    Z= ibp_generate(N, alpha, K_specified)
    K = Z.shape[1]

    Y_temp = npr.rand(K, T)
    Y = np.ones((K, T))
    Y[Y_temp < (1 - p)] = 0

    pX = 1 - ((1 -lmd) * np.matmul(Z, Y)) * (1 - epsilon)
    flips = npr.rand(N,T)
    X = np.zeros(pX.shape)
    X[flips < pX] = 1

    Z, Y = cannonize(Z, Y)

    plot_ibp_matrices(X, Z, Y)
    return X, Z, Y


if __name__ == '__main__':
    N = 6  # number of rows in observation matrix X
    T = 100  # number of columns in observation matrix X
    alpha = 2  # IBP concentration parameter
    lmd = .8  # noisy-or parameters (see paper)
    epsilon = .1  #
    p = .4  #
    num_samples = 20  # samples to draw using the Gibbs and RJMCMC sampler

    # generate test data from the model
    X, Z, Y = generate_test_data(N, T, alpha, lmd, epsilon, p)

    print(Z.shape[1])