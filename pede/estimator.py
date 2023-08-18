import math
import pathlib
from datetime import datetime

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pede.cosmology import Cosmology
from pede.posterior_calculator import PosteriorCalculator


class Estimator:
    def __init__(self, model: Cosmology, experiments: list) -> None:
        self._model = model
        self._experiments = experiments

        self._results_dir = None
        self._nwalkers = None
        self._ndim = None

        self._chains = None
        self._samples = None
        self._best_fit = None

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

        # Now we'll sample for up to max_iter steps
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

    def load_chains(self, results_dir):
        self._results_dir = pathlib.Path(results_dir)
        fname = self._results_dir / "chains.h5"
        self._chains = emcee.backends.HDFBackend(fname)
        self._nwalkers, self._ndim = self._chains.shape

        if self._ndim != len(self._model.get_initial_guess()):
            raise Exception(
                "Oops, number of parameters in chains and model are different."
            )

    def get_samples(self):
        if self._chains is None:
            raise Exception("Oops, no chains to process")

        tau = self._chains.get_autocorr_time()
        burn_in = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        self._samples = self._chains.get_chain(discard=burn_in, flat=True, thin=thin)
        log_prob_samples = self._chains.get_log_prob(
            discard=burn_in, flat=True, thin=thin
        )

        print("\nProcessing chains:")
        print(f"  burn-in: {burn_in}")
        print(f"  thin: {thin}")
        print(f"  flat chain shape: {self._samples.shape}")
        print(f"  flat log prob shape: {log_prob_samples.shape}")

        return self._samples

    def plot(self, save=False):
        if self._samples is None:
            print("Oops, no samples to plot. Let me get that for ya'.")
            self.get_samples()

        fig = corner.corner(
            self._samples,
            label=self._model.get_params_names(),
            quantiles=(0.16, 0.5, 0.84),
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )
        fig.suptitle(
            f"{self._model.get_name()} with {self._nwalkers} walkers and xx steps"
        )
        plt.show()

        if save:
            fname = self._results_dir / "plot.png"
            fig.savefig(fname)

        return fig

    def get_best_fit(self, save=False):
        if self._samples is None:
            print("Oops, no samples loaded. Let me get that for ya'.")
            self.get_samples()

        one_sigma_up = np.quantile(self._samples, q=0.84, axis=0)
        mean = np.quantile(self._samples, q=0.5, axis=0)
        one_sigma_down = np.quantile(self._samples, q=0.15, axis=0)

        err_up = one_sigma_up - mean
        err_down = mean - one_sigma_down

        bf = pd.DataFrame(
            [mean, err_up, err_down],
            index=["best_fit", "err_up", "err_down"],
            columns=self._model.get_params_names(),
        )

        self._best_fit = bf.transpose()

        # print("\nBest-fit results:")
        # print(self._best_fit)

        if save:
            fname = self._results_dir / "best_fit.csv"
            self._best_fit.to_csv(fname)

        return self._best_fit

    def information_criterion(self, save=False):
        if self._best_fit is None:
            print("Oops, no best fit. Let be get that for ya'.")
            self.get_best_fit(save=save)

        prob = PosteriorCalculator(cosmology=self._model, experiments=self._experiments)
        n_data = prob.get_n_data()

        chi2 = -2.0 * prob._ln_likelihood(self._best_fit["best_fit"])
        red_chi2 = chi2 / (n_data - self._ndim)
        aic = chi2 + 2.0 * self._ndim
        bic = chi2 + self._ndim * math.log(n_data)

        info_crit = pd.DataFrame(
            {
                # "model": [self._model.get_name()],
                "n_parameters": [self._ndim],
                "n_data": [n_data],
                "chi2": [chi2],
                "reduced_chi2": [red_chi2],
                "aic": [aic],
                "bic": [bic],
            },
            index=[self._model.get_name()],
        )

        # print("\nInformation criteria results:")
        # print(info_crit)

        # save to disk
        if save:
            fname = self._results_dir / "info_crit.csv"
            info_crit.to_csv(fname, index=False)
        return info_crit

    def analysis(self, save=False):
        self.get_best_fit(save=save)
        self.information_criterion(save=save)
        self.plot(save=save)
