import corner
from datetime import datetime
import pathlib

import emcee
import matplotlib.pyplot as plt
import numpy as np

from itm.cosmology import Cosmology
from itm.posterior_calculator import PosteriorCalculator


class Estimator:
    def __init__(self, model: Cosmology, experiments: list) -> None:
        self._model = model
        self._experiments = experiments
        self._chains = None
        self._samples = None
        self._nwalkers = None
        self._ndim = None
        self._results_dir = None

    def run(self, results_dir=None, nwalkers=32, max_iter=50000):
        self._nwalkers = nwalkers

        params_ic = self._model.get_initial_guess()
        self._ndim = len(params_ic)

        self._results_dir = (
            pathlib.Path(
                "results",
                self._model.get_name() + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            if results_dir is None
            else pathlib.Path(results_dir)
        )

        if not self._results_dir.is_dir():
            self._results_dir.mkdir()

        chains_path = self._results_dir / "chains.h5"
        backend = emcee.backends.HDFBackend(chains_path)
        backend.reset(self._nwalkers, self._ndim)

        print("Configuration")
        print("  model: ", self._model.get_name())
        print("  initial guess: ", params_ic)
        print("  walkers: ", self._nwalkers)

        # set up probabilities
        prob = PosteriorCalculator(cosmology=self._model, experiments=self._experiments)

        # set up emcee sampler
        sampler = emcee.EnsembleSampler(
            self._nwalkers, self._ndim, prob.ln_posterior, backend=backend
        )

        # set up walkers around initial conditions
        walkers_ic = [
            params_ic + 1e-4 * np.random.randn(self._ndim)
            for i in range(self._nwalkers)
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

        self._chains = sampler

        return sampler

    def load_chains(self, filename):
        self._chains = emcee.backends.HDFBackend(filename)
        self._nwalkers, self._ndim = self._chains.shape

        if self._ndim != len(self._model.get_initial_guess()):
            raise Exception("ndim for chains and model are different")

    def get_samples(self):
        tau = self._chains.get_autocorr_time()
        burn_in = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        self._samples = self._chains.get_chain(discard=burn_in, flat=True, thin=thin)
        log_prob_samples = self._chains.get_log_prob(
            discard=burn_in, flat=True, thin=thin
        )

        print(f"burn-in: {burn_in}")
        print(f"thin: {thin}")
        print(f"flat chain shape: {self._samples.shape}")
        print(f"flat log prob shape: {log_prob_samples.shape}")

        return self._samples

    def plot(self):

        if self._samples is None:
            print("No samples to plot")
            return

        fig = corner.corner(
            self._samples,
            # labels=["M", "$h$", "$\Omega_{b} h^2$", "$\Omega_{c} h^2$", "$w$"],
            quantiles=(0.16, 0.5, 0.84),
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        fig.suptitle(
            f"{self._model.get_name()} with {self._nwalkers} walkers and xx steps"
        )
        plt.show()
        # TODO: option to save plot
        return fig
