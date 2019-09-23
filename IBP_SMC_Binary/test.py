from generate_data import generate_test_data
from hyper_sampler import hyper_sampler
from particle_filter import particle_filter
from utils import inferstats
from plot_save_graphs import  plot_and_save_graphs


N = 6                  # number of rows in observation matrix X
T = 100                # number of columns in observation matrix X
alpha = 2              # IBP concentration parameter
lmd = .8            # noisy-or parameters (see paper)
epsilon = .1           #
p = .4                 #
num_samples = 20    # samples to draw using the Gibbs and RJMCMC sampler
    
# generate test data from the model
X,Z,Y = generate_test_data(N,T,alpha,lmd,epsilon,p)
print(len(X))
# run the Gibbs sampler (called hyper_sampler because this sampler samples
# the hyperparameters in the model as well
Z_sample, Y_sample, lP_sample, K_sample, alpha_sample, epsilon_sample, lmd_sample, p_sample = hyper_sampler(X,num_samples,Y,Z,alpha,epsilon,lmd,p)

# compute some posterior averages
[Ek, EZZt] = inferstats(Z_sample,None,0)
plot_and_save_graphs(X, Z, EZZt)
# run the RJMCMC sampler
#[Z_sample, Y_sample, lP_sample, K_sample, alpha_sample, epsilon_sample, lmd_sample, p_sample] = rjmcmc_sampler(X,num_samples,1,Y,Z,alpha,epsilon,lmd,p)

num_particles = 100   # number of particles to use in the SIS sampler

# run the SIS sampler
Zparticles, Yparticles = particle_filter(X,Z,Y,alpha,epsilon,lmd,p,num_particles)
Ek, EZZt = inferstats(Zparticles, Z=None, start=0)

plot_and_save_graphs(X, Z, EZZt)
