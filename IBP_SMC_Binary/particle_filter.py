import  numpy as np
import numpy.random as npr
from utils import multinomial_resampling, inferstats
from sampY import sampY_newrows_only


def particle_filter(X,Z,Y,alpha,epsilon,lmd,p,num_particles):

    Zparticles = [[] for _ in range(num_particles)]
    Yparticles = [[] for _ in range(num_particles)]

    N, T = X.shape

    X_one_inds = [[] for _ in range(N)]
    X_zero_inds = [[] for _ in range(N)]

    for n in range(N):
        X_one_inds[n] = np.argwhere(X[n,:] == 1)
        if X_one_inds[n] is not None:
            X_one_inds[n] = np.ravel(X_one_inds[n])
        X_zero_inds[n] = np.argwhere(X[n, :] == 0)
        if  X_zero_inds[n] is not None:
            X_zero_inds[n] =  np.ravel(X_zero_inds[n])


    for n in range(N):
        print('Particle filter row ' + str(n+1) + '/' + str(N))
        particle_weight = np.ones(num_particles) / num_particles

        if n == 0:
            for pind in range(num_particles):
                num_first_dishes = npr.poisson(alpha)
                Zparticles[pind] = np.ones(num_first_dishes).reshape(1,-1)
                if num_first_dishes == 0:
                    Zparticles[pind] = np.zeros((1,1))

                lpXiT = sum(X[0,:])*(np.log(1 - (1 - epsilon) * ((1 -lmd * p)** num_first_dishes)))
                lpXiT = lpXiT + sum(1 - X[0,:])*(np.log((1 - epsilon) * ((1 - lmd * p)**num_first_dishes)))
                particle_weight[pind] = lpXiT

            logmax = max(particle_weight)
            particle_weight = np.exp(particle_weight - logmax)
            particle_weight = particle_weight / sum(particle_weight)
        else:
            for pind in range(num_particles):
                local_Z = Zparticles[pind]

                m_k = np.sum(local_Z, axis=0)
                K = local_Z.shape[1]

                p_k = m_k / (n + 1)
                if sum(m_k) > 0:
                    shared_choices = np.argwhere( p_k > npr.rand(len(m_k)))
                else:
                    shared_choices= []

                new_choices = npr.poisson(alpha / (n+1))

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

                new_Z = Zparticles[pind]
                local_Y = Yparticles[pind]
               
                lpXiT = 0
                
                e = np.ravel(np.matmul(new_Z[n, 0:K].reshape(1,-1), local_Y))
                oneinds = X_one_inds[n]
                
                zeroinds = X_zero_inds[n]
                
                lpXiT = lpXiT + sum( \
                    np.log(1 - (1 - epsilon) * ((1 -lmd)** e[oneinds]) * ((1 -lmd * p)** new_choices)))
                lpXiT = lpXiT + sum( \
                    np.log((1 - epsilon) * ((1 -lmd)** e[zeroinds]) * ((1 -lmd * p)** new_choices)))

                particle_weight[pind] = lpXiT

            logmax = max(particle_weight)
            particle_weight = np.exp(particle_weight - logmax)

            particle_weight = particle_weight / sum(particle_weight)

        resample_index = multinomial_resampling(particle_weight)

        for pind in range(num_particles):
            try:
                num_new_rows = Zparticles[pind].shape[1] - len(Yparticles[pind])
                Yparticles[pind] = np.row_stack((Yparticles[pind], np.zeros((num_new_rows, T))) )
            except:
                #num_new_rows = len(Zparticles[pind]) - 0
                Yparticles[pind] = np.zeros((num_new_rows, T))
                loc_Z = Zparticles[pind].reshape(1,-1)

            if num_new_rows>0:
                Yparticles[pind] = sampY_newrows_only(X[0:n, :], Yparticles[pind] , loc_Z,
                                                   epsilon, lmd , p, len(Yparticles))


        Zparticles  = [Zparticles[i] for i in resample_index]
        Yparticles = [Yparticles[i] for i in resample_index]


    #resample_index = multinomial_resampling(particle_weight, 1)
    #index = resample_index[0]
    #return Zparticles[index], Yparticles[index]
    return Zparticles, Yparticles


if __name__ == '__main__':
    sigma_X = 0.5

    p = .2
    epsilon = .0001
    lmd = .5
    alpha = 2.2
    #T = 1000
    #N = 20

    X = np.array([[1,0],[0,1]])
    Z = np.array([[1,0],[0,1]])
    Y = np.array([[1, 0], [0, 1]])

    #Zsamp = sampZ(X, Z, 0.5, 1, 4)
    Zparticles, Yparticles = particle_filter(X,Z,Y,alpha,epsilon,lmd,p,10)

    #Ek, EZZt, in_err, s_err, in_r, s_r = inferstats(Zparticles, Z=None, start=0)
    Ek, EZZt = inferstats(Zparticles, Z=None, start=0)

    print(Ek, EZZt)