import numpy as np
import emcee
import corner
import os
import time
import matplotlib as plt

# from likelihood import lnprob
from itm.likelihood import lnprob
 
start_time = time.time()

# chute da posicao inicial dos walkers. a posicao de cada walker eh diferente. o p1_0 eh o centro da bola
p0 = [24.96, 0.69, 0.022, 0.12]

nwalkers = 2**3  # no. de walkers
steps = 1000  # no. de passos
burnin = 100  # tempo (passos) para os walker passarem o burn in

ndim = len(p0)  # no. de parametros
# nthreads = 4

# which data?
file_name = 'lcdm'
base_path = "/Users/fabio/code/fchibana/memos"

filename = "test.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


# MCMC =============================================================================================

print("MCMC")
print("walkers: ", nwalkers)
print("steps: ", steps)



# initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)
# j = 0
# while os.path.exists("results/chains/chains_"+file_name+"_%s.dat" % j) or os.path.exists("results/chains_"+file_name+"_%s.txt" % j) or  os.path.exists("results/triagle_scf_"+file_name+"_%s.png" % j) or os.path.exists("results/triangle_"+file_name+"_%s.png" % j) :
#     j += 1
#     print(j)

# f = open("results/chains/chains_"+file_name+"_%s.dat" %j, "w")
# f.close()

# condicoes iniciais dos walkers dentro da bola de centro p1_0
pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

max_n = 100000

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    print(tau)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
    
# roda MCMC
# print("Running burn in chains...")
# state = sampler.run_mcmc(pos, burnin, progress=True)
# sampler.reset()

# print("Sampling the posterior...")
# sampler.run_mcmc(state, int(steps), progress=True)
# print("Done!")
# Serialization
#     position = result[0]
# f = open("results/chains/chains_"+file_name+"_%s.dat" %j, "a")
# for k in range(position.shape[0]):
#     np.savetxt(f, position[k],fmt='%1.4e')
# f.close()

# Analysis ===========================================================================================

# TODO: get autocorrelation time
tau = sampler.get_autocorr_time()
print(tau)

flat_samples = sampler.get_chain(discard=1, thin=15, flat=True)
print(flat_samples.shape)


# TODO: compute sigmas

# Calcula as medias de cada parametro
# medias = []
# for i in range(ndim):
#     medias.append(np.percentile(samples[:,i],50))

# Calcula os devios para cima e para baixo de cada parametro
# desvios = []
# for i in range(ndim):
#     desvios.append([np.percentile(samples[:,i],84.13) - np.percentile(samples[:,i],50), np.percentile(samples[:,i],50) - np.percentile(samples[:,i],15.869999999999997)])

# print("""MCMC result:
#     M           = {0[0]} +{1[0][0]} -{1[0][1]}
#     h          = {0[1]} +{1[1][0]} -{1[1][1]}
#     omega0_b    = {0[2]} +{1[2][0]} -{1[2][1]}
#     omega0_cdm  = {0[3]} +{1[3][0]} -{1[3][1]}
# """.format(medias, desvios))

fig = corner.corner(flat_samples,
                    labels=["M", "$h$",
                            "$\Omega_{b} h^2$", "$\Omega_{c} h^2$"],
                    quantiles=(0.16, 0.5, 0.84), show_titles=True,
                    title_kwargs={"fontsize": 12})
fig.suptitle('walkers: %s steps: %s' % (nwalkers, steps))
# fig = corner.corner(flat_samples)

fig.savefig("test.png")

# create output files ==================================================================================
# end_time = time.time() - start_time
# execution_time = 'Execution time: '+str(end_time)+'s'

# np.savetxt("results/chains_"+file_name+"_%s.txt" % j, samples, fmt='%1.4e', header=execution_time)
# fig.savefig("results/triangle_"+file_name+"_%s.png" % j)
