import  numpy as np
import numpy.random as npr
from utils import logPX, logPXZ, inferstats
from numpy.linalg import inv
from sampZ import sampZ
from math import log


def hyper_sampler(X, num_samples,
                  true_Z,
                  true_alpha,
                  true_sigma_x,
                  true_sigma_A):

    N = len(X)
    # build data structures to hold samples
    Z_sample = [ [] for _ in range(num_samples)]
    lP_sample = np.zeros(num_samples)
    K_sample = np.zeros(num_samples)
    sigma_x_sample = np.zeros(num_samples)
    sigma_A_sample = np.zeros(num_samples)
    alpha_sample = np.zeros(num_samples)
    structure_error = np.zeros(num_samples)
    in_degree_error = np.zeros(num_samples)

    # initialize sampler with smart guesses
    Z_sample[0] = np.round(npr.rand(N)).reshape(-1,1)
    sigma_x_sample[0] = .5
    alpha_sample[0] = true_alpha
    sigma_A_sample[0] = .5

    K_sample[0] = Z_sample[0].shape[1]

    lP_sample[0] = logPXZ(X, Z_sample[0], sigma_x_sample[0], sigma_A_sample[0], alpha_sample[0])

    # % there are two metropolis (vs. Gibbs) samplings steps per sweep, one
    # % for lambda and one for epsilon -- make variables to keep track of
    # % the acceptance ratio so as to scale the proposal variance according to
    # % that ratio


    num_sigma_x_moves_accepted = 0
    sigma_x_move_acceptance_ratio = np.zeros(num_samples)
    sigma_x_move_acceptance_ratio[0] = 0
    sigma_x_proposal_variance = .01
    sigma_x_bound =[0, np.inf]

    num_sigma_A_moves_accepted = 0
    sigma_A_move_acceptance_ratio = np.zeros(num_samples)
    sigma_A_move_acceptance_ratio[0] = 0
    sigma_A_proposal_variance = .01
    sigma_A_bound =[0, np.inf]

    num_alpha_moves_accepted = 0
    alpha_move_acceptance_ratio = np.zeros(num_samples)
    alpha_move_acceptance_ratio[0] = 0
    alpha_proposal_variance = .1
    alpha_bound =[0, np.inf]



    # if the truth is provided, compute the ground truth score

    #true_model_score = logPXZ(X, true_Z, true_sigma_x, true_sigma_A, true_alpha)


    #Hn = sum(1 / np.array(range(1,N+1)))

    for sweep in range(1,num_samples):
        if sweep%2 == 0:
            print('Sweep '+str(sweep)+'/'+str(num_samples))

        Z_sample[sweep] = sampZ(X, Z_sample[sweep - 1], sigma_x_sample[sweep - 1],
                                sigma_A_sample[sweep - 1], alpha_sample[sweep - 1])
        # epsilon metropolis step
        sigma_A_proposal = sigma_A_sample[sweep - 1] + npr.randn() * sigma_A_proposal_variance

        if (sigma_A_proposal > sigma_A_bound[0]) and (sigma_A_proposal < sigma_A_bound[1]):

            lp_sigma_A_proposal = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep - 1],
                                sigma_A_proposal, alpha_sample[sweep - 1])
            lp_sigma_A = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep - 1],
                                sigma_A_sample[sweep - 1], alpha_sample[sweep - 1])


            log_acceptance_ratio = lp_sigma_A_proposal - lp_sigma_A

            if log(npr.rand()) < min(log_acceptance_ratio, 0):
                sigma_A_sample[sweep] = sigma_A_proposal
                num_sigma_A_moves_accepted = num_sigma_A_moves_accepted + 1
            else:
                sigma_A_sample[sweep] = sigma_A_sample[sweep - 1]
        else:
            sigma_A_sample[sweep] = sigma_A_sample[sweep - 1]

        sigma_A_move_acceptance_ratio[sweep] = num_sigma_A_moves_accepted / sweep

        # sigma_x metropolis step
        sigma_x_proposal = sigma_x_sample[sweep - 1] + npr.randn() * sigma_x_proposal_variance

        if (sigma_x_proposal > sigma_x_bound[0]) and (sigma_x_proposal < sigma_x_bound[1]):
            lp_sigma_x_proposal = logPXZ(X, Z_sample[sweep], sigma_x_proposal,
                                        sigma_A_sample[sweep], alpha_sample[sweep - 1])
            lp_sigma_x = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep - 1],
                               sigma_A_sample[sweep], alpha_sample[sweep - 1])

            log_acceptance_ratio = lp_sigma_x_proposal - lp_sigma_x

            if log(npr.rand()) < min(log_acceptance_ratio, 0):
                sigma_x_sample[sweep] = sigma_x_proposal
                num_sigma_x_moves_accepted = num_sigma_x_moves_accepted + 1
            else:
                sigma_x_sample[sweep] = sigma_x_sample[sweep - 1]
        else:
            sigma_x_sample[sweep] = sigma_x_sample[sweep - 1]

        sigma_x_move_acceptance_ratio[sweep] = num_sigma_x_moves_accepted / sweep

        # alpha gibbs step
        alpha_proposal = alpha_sample[sweep - 1] + npr.randn() * alpha_proposal_variance

        if (alpha_proposal > alpha_bound[0]) and (alpha_proposal < sigma_x_bound[1]):
            lp_alpha_proposal = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep],
                                        sigma_A_sample[sweep], alpha_proposal)
            lp_alpha = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep],
                               sigma_A_sample[sweep], alpha_sample[sweep - 1])

            log_acceptance_ratio = lp_alpha_proposal - lp_alpha

            if log(npr.rand()) < min(log_acceptance_ratio, 0):
                alpha_sample[sweep] = alpha_proposal
                num_alpha_moves_accepted = num_alpha_moves_accepted + 1
            else:
                alpha_sample[sweep] = alpha_sample[sweep - 1]
        else:
            alpha_sample[sweep] = alpha_sample[sweep - 1]

        alpha_move_acceptance_ratio[sweep] = num_alpha_moves_accepted / sweep


        lP_sample[sweep] = logPXZ(X, Z_sample[sweep], sigma_x_sample[sweep],
                                  sigma_A_sample[sweep], alpha_sample[sweep])
        K_sample[sweep] = Z_sample[sweep].shape[1]


    #[Ek, EZZt, cur_in_degree_error, cur_structure_error] = inferstats(Z_sample[1:sweep], true_Z)
    #structure_error[sweep] = cur_structure_error
    #in_degree_error[sweep] = cur_in_degree_error

    return Z_sample, lP_sample, K_sample, alpha_sample, sigma_x_sample, sigma_A_sample


if __name__ == '__main__':
    #X, Z, A, sigma_X = generate_test_data(10)
    # print(X.shape)
    # print(Z)
    # lp = logPX(X, Z, sigma_X, 1)
    # print(lp)


    X = np.array([[1,1,1],[1,1,1]])
    #Z = np.array([[1,0],[0,1]])
    Z = np.array([[1], [0]])
    A = np.array([[1, 0], [0, 1]])
    Z_sample, lP_sample, K_sample, alpha_sample, sigma_x_sample, sigma_A_sample = hyper_sampler(X, 2, Z, 4, 0.5, 1)

    print(Z_sample, lP_sample, K_sample, alpha_sample, sigma_x_sample, sigma_A_sample)
    #lp = logPX(X, Z, sigma_X, 1)
    #print(lp)
    #lpz = logPXZ(X, Z, sigma_X, 1, 4)
    #print(lpz)


    # Zparticles = particle_filter(X, 10, 4, sigma_X, 1)
    # print(Zparticles[0])

    # for matlab
    #X = [[1, 1, 1]; [1, 1, 1]]
    #Z = [[1, 0]; [0, 1]]