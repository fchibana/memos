import numpy as np

from itm.cosmology import Cosmology
from itm.data_loader import DataLoader
from itm.observables import Observables


class PosteriorCalculator:
    def __init__(self, cosmology: Cosmology, experiments: list) -> None:
        self._cosmology = cosmology
        self._experiments = experiments

        self._observables = Observables(self._cosmology)
        self._data = DataLoader(experiments)
        self._n_data = self._data.get_n_data()

    def ln_posterior(self, parameters):
        # print(type(parameters))     # ndarray
        # print(parameters.shape)     # (4,)
        # print(type(parameters[0]))  # np.float64

        # ln_priors = self._ln_prior(parameters)
        ln_priors = self._cosmology.get_prior(parameters)

        if np.isinf(ln_priors):
            return -np.inf

        if self._cosmology.get_name() == "itm":
            self._cosmology.update_and_solve(parameters)

        # make sure energy density is positive
        x = np.linspace(0, 20, 11)
        if np.any(self._cosmology.rho_de(x, parameters) < 0):
            # print("Negative rho_de. Skipping")
            return -np.inf
        if np.any(self._cosmology.rho_cdm(x, parameters) < 0):
            # print("Negative rho_cdm. Skipping")
            return -np.inf

        return ln_priors + self._ln_likelihood(parameters)

    def _ln_likelihood(self, parameters):
        # M, h, omega0_b, omega0_cdm = parameters
        # M = parameters[0]
        h = parameters[1]
        # omega0_b = parameters[2]
        # omega0_cdm = parameters[3]

        H0 = 100.0 * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2

        ln_likehood = 0

        if "local_hubble" in self._experiments:
            data = self._data.get_local_hubble()
            model = H0
            ln_likehood += self._ln_gauss(
                y_fit=model,
                y_target=data["y"],
                y_err=data["y_err"],
            )

        if "cosmic_chronometers" in self._experiments:
            data = self._data.get_cosmic_chronometers()
            model = self._cosmology.hubble(data["x"], parameters)
            ln_likehood += self._ln_gauss(
                y_fit=model,
                y_target=data["y"],
                y_err=data["y_err"],
            )

        if "jla" in self._experiments:
            data = self._data.get_jla()
            model = self._observables.distance_modulus(data["x"], parameters)
            ln_likehood += self._ln_multivariate_gauss(
                y_fit=model, y_target=data["y"], y_cov=data["cov"]
            )

        if "bao_compilation" in self._experiments:
            data = self._data.get_bao_compilation()
            model = self._observables.d_BAO(data["x"], parameters)
            ln_likehood += self._ln_gauss(
                y_fit=model,
                y_target=data["y"],
                y_err=data["y_err"],
            )

        if "bao_wigglez" in self._experiments:
            data = self._data.get_bao_wigglez()
            model = self._observables.d_bao_wigglez(data["x"], parameters)
            ln_likehood += self._ln_multivariate_gauss(
                y_fit=model,
                y_target=data["y"],
                y_cov=data["cov"],
            )

        return ln_likehood

    def _ln_gauss(self, y_fit, y_target, y_err):
        inv_sigma2 = 1.0 / y_err**2

        r = y_target - y_fit
        chi2 = r**2 * inv_sigma2 - np.log(inv_sigma2)

        return -0.5 * np.sum(chi2)

    def _ln_multivariate_gauss(self, y_fit, y_target, y_cov):
        inv_cov = np.linalg.inv(y_cov)
        det_inv_cov = np.linalg.det(inv_cov)

        r = y_target - y_fit
        chi2 = np.dot(r, np.dot(inv_cov, r))

        return -0.5 * (chi2 - np.log(det_inv_cov))

    def get_n_data(self):
        return self._n_data
