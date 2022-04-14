import emcee
import numpy as np


def get_samples(sampler):
    tau = sampler.get_autocorr_time()
    burn_in = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burn_in, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True, thin=thin)

    print("burn-in: {0}".format(burn_in))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))

    return samples


def main():
    fname = "results/chains_210926.h5"
    sampler = emcee.backends.HDFBackend(fname)

    get_samples(sampler)


if __name__ == "__main__":
    main()
