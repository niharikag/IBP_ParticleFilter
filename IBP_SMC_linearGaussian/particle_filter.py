import  numpy as np
import numpy.random as npr
from utils import logPX, multinomial_resampling, inferstats
from numpy.linalg import inv


def particle_filter(X, num_particles, true_alpha, true_sigma_x, true_sigma_A):
    Zparticles = [ [] for _ in range(num_particles)]
    N = len(X)

    for n in range(N):
        particle_weight = np.ones(num_particles) / num_particles

        if n == 0:
            for pind in range(num_particles):
                num_first_dishes = npr.poisson(true_alpha)
                Z_local = np.ones(num_first_dishes).reshape(1,-1)
                if num_first_dishes == 0:
                    Z_local = np.zeros(1).reshape(1,-1)

                Zparticles[pind] = Z_local.copy()
                particle_weight[pind] = logPX(X[n,:].reshape(1,-1), Z_local, true_sigma_x, true_sigma_A)

            logmax = max(particle_weight)
            particle_weight = np.exp(particle_weight - logmax)
            particle_weight = particle_weight / sum(particle_weight)
        else:
            for pind in range(num_particles):
                local_Z = Zparticles[pind]

                m_k = np.sum(local_Z, axis=0)
                K = local_Z.shape[1]

                if sum(m_k) > 0:
                    shared_choices = np.where((m_k / (n+1)) > npr.rand(len(m_k)))[0]
                else:
                    shared_choices= []

                new_choices = npr.poisson(true_alpha / (n+1))

                Z_temp = np.zeros(K+new_choices)
                Z_temp[K:] = 1

                if len(shared_choices) >= 1 :
                    #print(K, m_k, shared_choices)
                    Z_temp[shared_choices] = 1

                if new_choices > 0:
                    if n == 1:
                        Zparticles[pind] = np.column_stack((Zparticles[pind].reshape(1,-1), np.zeros((1, new_choices))))
                    else:
                        Zparticles[pind] = np.column_stack((Zparticles[pind], np.zeros(((n), new_choices))))

                Zparticles[pind] = np.row_stack((Zparticles[pind], Z_temp))

                Z_local = Zparticles[pind]
                K = Z_local.shape[1]
                Z_T_Z = np.matmul(Z_local.T, Z_local)
                inv_factor = Z_T_Z + (true_sigma_x**2 / true_sigma_A**2) * np.eye(K)
                z_inv =  inv(inv_factor)
                #z_temp = np.matmul(z_inv, Z_local.T)
                #M = np.eye(n) - np.matmul(Z_local, z_temp)

                A = np.eye(n+1) / true_sigma_x ** 2
                Ainv = np.eye(n+1) * true_sigma_x ** 2

                B = -z_inv/true_sigma_x**2
                Binv = -(inv_factor)*true_sigma_x**2
                Xli = Z_local

                z_temp = np.matmul(np.matmul(Z_local.T, Ainv), Xli)
                x_inv = np.matmul(Z_local, inv(Binv+z_temp))
                x_inv = np.matmul(x_inv, Z_local.T)
                C_N1_inv = Ainv- np.matmul(np.matmul(Ainv, x_inv), Ainv)
                C_N1 = A+ np.matmul(np.matmul(Z_local, B), Z_local.T)

                C_N_inv = C_N1[:-1, :-1] - 1 / C_N1[-1, -1] * C_N1[:-1, -1] * C_N1[-1, :-1]

                ACCB = C_N1_inv
                C = ACCB[:-1, -1]
                B = ACCB[-1, -1]

                x = X[:n,:]
                y = X[n,:]

                mean = np.matmul(np.matmul(C.T,C_N_inv), x)
                covar = B - np.matmul(np.matmul(C.T,C_N_inv), C)

                particle_weight[pind] = (-(y - mean).dot((y - mean).T)/(2*covar)) - \
                                        0.5*len(y)*(np.log(2*np.pi*covar)).T

            logmax = max(particle_weight)
            particle_weight = np.exp(particle_weight - logmax)

            particle_weight = particle_weight / sum(particle_weight)

        resample_index = multinomial_resampling(particle_weight)
        Zparticles  = [Zparticles[i] for i in resample_index]

    return Zparticles



if __name__ == '__main__':
    sigma_X = 0.5
    X = np.array([[1,0],[1,1]])
    Z = np.array([[1,0],[0,1]])
    A = np.array([[1, 0], [0, 1]])
    lp = logPX(X, Z, sigma_X, 1)

    Zparticles = particle_filter(X, 10, 4, sigma_X, 1)

    Ek, EZZt, in_err, s_err, in_r, s_r = inferstats(Zparticles, Z, 0)

    print(Ek, EZZt, in_err, s_err, in_r, s_r)