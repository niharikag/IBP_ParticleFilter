import  numpy as np
from numpy.linalg import pinv
from math import ceil, sqrt
from generate_data import generate_test_data
from hyper_sampler import hyper_sampler
from particle_filter import particle_filter
import matplotlib.pyplot as plt


X, Z_plus, A, sigma_x = generate_test_data(100)

iter_count = 100
Z_sample, lP_sample, K_sample, alpha_sample, sigma_x_sample, sigma_A_sample = hyper_sampler(X,
                                                                                            iter_count,
                                                                                            Z_plus,
                                                                                            2,
                                                                                            sigma_x,
                                                                                            1)

Z_last = Z_sample[iter_count-1]
temp_z = np.matmul(Z_last.T, Z_last)
temp_z = np.matmul(pinv(temp_z), Z_last.T)
Aest = np.matmul(temp_z, X)
print(len(Aest))
num_figs = ceil(sqrt(len(Aest)))
print(num_figs)

for i in range(len(Aest)):
    plt.subplot(num_figs,num_figs, i+1)
    #subplot(num_figs,num_figs,i)
    plt.imshow(Aest[i,:].reshape(6,6), cmap="gray")

plt.show()

Zparticles = particle_filter(X, 100, 1, .5, .4)

Z_last_pf = Zparticles[0]
temp_z = np.matmul(Z_last_pf.T, Z_last_pf)
temp_z = np.matmul(pinv(temp_z), Z_last_pf.T)

Aest = np.matmul(temp_z, X)
num_figs = ceil(sqrt(len(Aest)))

print(num_figs)

for i in range(len(Aest)):
    plt.subplot(num_figs,num_figs, i+1)
    #subplot(num_figs,num_figs,i)
    plt.imshow(Aest[i,:].reshape(6,6), cmap="gray")

plt.show()

