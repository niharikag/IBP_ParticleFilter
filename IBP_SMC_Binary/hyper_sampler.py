import  numpy as np
import numpy.random as npr
from utils import logPZ, logPXYZ, inferstats, cannonize, clean
from sampZ import sampZ
from sampY import sampY
from math import log


def hyper_sampler(X,num_samples,Y,Z,alpha,epsilon,lmd,p):

    N = len(X)
    # build data structures to hold samples
    Z_sample = [[] for _ in range(num_samples)]
    Y_sample = [[] for _ in range(num_samples)]
    lP_sample = np.zeros(num_samples)
    K_sample = np.zeros(num_samples)
    epsilon_sample = np.zeros(num_samples)
    lambda_sample = np.zeros(num_samples)
    alpha_sample = np.zeros(num_samples)
    p_sample = np.zeros(num_samples)
    structure_error = np.zeros(num_samples)
    in_degree_error = np.zeros(num_samples)

    # initialize sampler with smart guesses
    Z_sample[0] = np.round(npr.rand(N)).reshape(-1,1)
    Y_sample[0] = np.zeros((Z_sample[0].shape[1],X.shape[1]))
    epsilon_sample[0] = npr.rand()
    alpha_sample[0] = npr.gamma(1,1)
    lambda_sample[0] = npr.rand()
    p_sample[0] = npr.beta(1, 1)

    K_sample[0] = Z_sample[0].shape[1]
    lP_sample[0] = logPXYZ(X, Y_sample[0], Z_sample[0], alpha_sample[0],
                           epsilon_sample[0], lambda_sample[0])

    #  there are two metropolis (vs. Gibbs) samplings steps per sweep, one
    #  for lambda and one for epsilon -- make variables to keep track of
    #  the acceptance ratio so as to scale the proposal variance according to
    #  that ratio


    num_epsilon_moves_accepted = 0
    epsilon_move_acceptance_ratio = np.zeros(num_samples)
    epsilon_move_acceptance_ratio[0] = 0
    epsilon_proposal_variance = .01
    epsilon_bound =[0, 1]

    num_lambda_moves_accepted = 0
    lambda_move_acceptance_ratio = np.zeros(num_samples)
    lambda_move_acceptance_ratio[0] = 0
    lambda_proposal_variance = .01
    lambda_bound =[0, 1]

    num_alpha_moves_accepted = 0
    alpha_move_acceptance_ratio = np.zeros(num_samples)
    alpha_move_acceptance_ratio[0] = 0

    # if the truth is provided, compute the ground truth score
    #true_model_score = logPXZ(X, true_Z, true_epsilon, true_lambda, true_alpha)


    Hn = sum(1 / np.array(range(1,N+1)))

    for sweep in range(1,num_samples):
        if sweep%2 == 0:
            print('Sweep '+str(sweep)+'/'+str(num_samples))

        Z_sample[sweep], Y_sample[sweep] = sampZ(X, Y_sample[sweep - 1], Z_sample[sweep - 1],
                                                 alpha_sample[sweep - 1],
                                                 epsilon_sample[sweep - 1],
                                                 lambda_sample[sweep - 1],
                                                 p_sample[sweep - 1])

        Y_sample[sweep] = sampY(X, Y_sample[sweep], Z_sample[sweep],
                                 epsilon_sample[sweep - 1],
                                 lambda_sample[sweep - 1],
                                 p_sample[sweep - 1])

        Z_sample[sweep], Y_sample[sweep] = cannonize(Z_sample[sweep], Y_sample[sweep])
        Z_sample[sweep], Y_sample[sweep] = clean(Z_sample[sweep], Y_sample[sweep])

        # epsilon metropolis step
        epsilon_proposal = epsilon_sample[sweep - 1] + npr.randn() * epsilon_proposal_variance

        if (epsilon_proposal > epsilon_bound[0]) and (epsilon_proposal < epsilon_bound[1]):
            lp_epsilon_proposal = logPXYZ(X, Y_sample[sweep], Z_sample[sweep],
                                 alpha_sample[sweep - 1],
                                 epsilon_proposal,
                                 lambda_sample[sweep - 1])

            lp_epsilon = logPXYZ(X, Y_sample[sweep], Z_sample[sweep],
                                 alpha_sample[sweep - 1],
                                 epsilon_sample[sweep - 1],
                                 lambda_sample[sweep - 1])


            log_acceptance_ratio = lp_epsilon_proposal - lp_epsilon

            if log(npr.rand()) < min(log_acceptance_ratio, 0):
                epsilon_sample[sweep] = epsilon_proposal
                num_epsilon_moves_accepted = num_epsilon_moves_accepted + 1
            else:
                epsilon_sample[sweep] = epsilon_sample[sweep - 1]
        else:
            epsilon_sample[sweep] = epsilon_sample[sweep - 1]

        epsilon_move_acceptance_ratio[sweep] = num_epsilon_moves_accepted / sweep

        # lambda M-H step
        # epsilon metropolis step
        lambda_proposal = lambda_sample[sweep - 1] + npr.randn() * lambda_proposal_variance

        if (lambda_proposal > lambda_bound[0]) and (lambda_proposal < lambda_bound[1]):

            lp_lambda_proposal = logPXYZ(X, Y_sample[sweep], Z_sample[sweep],
                                 alpha_sample[sweep - 1],
                                 epsilon_sample[sweep],
                                 lambda_proposal)

            lp_lambda = logPXYZ(X, Y_sample[sweep], Z_sample[sweep],
                                 alpha_sample[sweep - 1],
                                 epsilon_sample[sweep ],
                                 lambda_sample[sweep - 1])


            log_acceptance_ratio = lp_lambda_proposal - lp_lambda

            if log(npr.rand()) < min(log_acceptance_ratio, 0):
                lambda_sample[sweep] = lambda_proposal
                num_lambda_moves_accepted = num_lambda_moves_accepted + 1
            else:
                lambda_sample[sweep] = lambda_sample[sweep - 1]
        else:
            lambda_sample[sweep] = lambda_sample[sweep - 1]

        lambda_move_acceptance_ratio[sweep] = num_lambda_moves_accepted / sweep

        # p gibbs step
        num_ones_in_Y = np.sum(Y_sample[sweep])
        num_zeros_in_Y = np.prod(Y_sample[sweep].shape) - num_ones_in_Y
        p_sample[sweep] = npr.beta(num_ones_in_Y + 1, num_zeros_in_Y + 1)

        # alpha gibbs step
        K_plus = Z_sample[sweep].shape[1]
        alpha_sample[sweep] = npr.gamma(1 + K_plus, 1 / (1 + Hn))

        lP_sample[sweep] = logPXYZ(X, Y_sample[sweep], Z_sample[sweep], alpha_sample[sweep],
                                   epsilon_sample[sweep], lambda_sample[sweep])

        K_sample[sweep] = Z_sample[sweep].shape[1]


    #[Ek, EZZt, cur_in_degree_error, cur_structure_error] = inferstats(Z_sample[1:sweep], true_Z)
    #structure_error[sweep] = cur_structure_error
    #in_degree_error[sweep] = cur_in_degree_error

    return Z_sample, Y_sample, lP_sample, K_sample, alpha_sample, epsilon_sample, lambda_sample, p_sample


if __name__ == '__main__':
    #X, Z, A, sigma_X = generate_test_data(10)
    # print(X.shape)
    # print(Z)
    # lp = logPX(X, Z, sigma_X, 1)
    # print(lp)

    p = .2
    epsilon = .0001
    lmd = .5
    alpha = 2.2
    # T = 1000
    # N = 20

    X = np.array([[1, 1], [1, 0]])
    Z = np.array([[1, 0], [0, 1]])
    Y = np.array([[1, 0], [0, 1]])

    Z_sample, Y_sample, lP_sample, K_sample, alpha_sample, epsilon_sample, lmd_sample, p_sample = hyper_sampler(X,10,Y,Z,alpha,epsilon,lmd,p)

