from datetime import datetime

import corner
import emcee
import matplotlib as plt
import numpy as np

import itm.likelihood
import itm.utils
 
config_params = itm.utils.load_config("/Users/fabio/code/fchibana/tachyons/config.yaml")

print(config_params)

mcmc_params = config_params["mcmc_params"]


p0 = [24.96, 0.69, 0.022, 0.12]         # initial guess 
ndim = len(p0)  
nwalkers =  mcmc_params["n_walkers"] 

out_name = "results/" + datetime.now().strftime("%Y%m%d_%H%M%S")
print(out_name)

backend = emcee.backends.HDFBackend(out_name + ".h5")
backend.reset(nwalkers, ndim)


# MCMC =============================================================================================

print("MCMC")
print("walkers: ", nwalkers)



# initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, itm.likelihood.lnprob, backend=backend)

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
    steps = sampler.iteration
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
    


# Analysis ===========================================================================================

tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
print(flat_samples.shape)

fig = corner.corner(flat_samples,
                    labels=["M", "$h$",
                            "$\Omega_{b} h^2$", "$\Omega_{c} h^2$"],
                    quantiles=(0.16, 0.5, 0.84), show_titles=True,
                    title_kwargs={"fontsize": 12})
fig.suptitle('walkers: %s steps: %s' % (nwalkers, steps))
# fig = corner.corner(flat_samples)

fig.savefig(out_name + ".png")
