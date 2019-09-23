import matplotlib.pyplot as plt
import numpy as np


def plot_and_save_graphs(X, Z, EZZt):
    plt.figure(1)
    plt.imshow(X)
    # title('Training data X')
    plt.xlabel('T')
    plt.ylabel('N')
    plt.show()
    plt.clf()

    plt.imshow(np.matmul(Z, Z.T))
    plt.show()
    # title('Z*Z'' training data')


    plt.clf()
    plt.imshow(EZZt)
    plt.show()
    # title('EZZt')



