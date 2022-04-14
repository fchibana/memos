from datetime import datetime

import corner
import emcee
import numpy as np
import pathlib

import itm.cosmology
from itm.posterior_calculator import PosteriorCalculator
import itm.utils


def get_chains(model: itm.cosmology.Cosmology, experiments: list, results_dir=None):

    # Ensember configuration
    max_iter = 1000
    n_walkers = 32

    params_ic = model.get_initial_guess()
    n_params = len(params_ic)

    if results_dir is None:
        results_dir = pathlib.Path(
            model.get_name() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    if not results_dir.is_dir():
        results_dir.mkdir()

    chains_path = results_dir / "chains.h5"
    backend = emcee.backends.HDFBackend(chains_path)
    backend.reset(n_walkers, n_params)

    print("Configuration")
    print("  model: ", model.get_name())
    print("  initial guess: ", params_ic)
    print("  walkers: ", n_walkers)

    # set up probabilities
    prob = PosteriorCalculator(cosmology=model, experiments=experiments)

    # set up emcee sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, prob.ln_posterior, backend=backend
    )

    # set up walkers around initial conditions
    walkers_ic = [
        params_ic + 1e-4 * np.random.randn(n_params) for i in range(n_walkers)
    ]

    # We'll track how the average autocorrelation time estimate changes
    autocorr_index = 0
    autocorr = np.empty(max_iter)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(walkers_ic, iterations=max_iter, progress=True):
        # Only check convergence every 100 steps
        step = sampler.iteration
        if step % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        # print(tau)
        autocorr[autocorr_index] = np.mean(tau)
        autocorr_index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau


# def check_convergence():


def main():
    # run configuration
    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]
    cosmo = itm.cosmology.LCDM()
    # cosmo = itm.cosmology.WCDM()

    # paths
    # RESULTS_BASE_DIR = pathlib.Path("results")
    # run_name = cosmo.get_name() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    # results_dir = RESULTS_BASE_DIR / run_name

    get_chains(cosmo, experiments)


if __name__ == "__main__":
    main()
