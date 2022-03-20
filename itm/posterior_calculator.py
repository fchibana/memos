import numpy as np

from itm.data_loader import DataLoader
from itm.observables import Observables


class PosteriorCalculator:

    def __init__(self, cosmology, experiments) -> None:
        self._cosmology = cosmology
        self._experiments = experiments

        self._observables = Observables(self._cosmology)
        self._data = DataLoader(experiments)
        self._parameters = {}

    def ln_posterior(self, parameters):
        ln_priors = self._ln_prior(parameters)

        if np.isinf(ln_priors):
            return -np.inf

        return ln_priors + self._ln_likelihood(parameters)

    def _ln_prior(self, parameters):
        M, h, omega0_b, omega0_cdm = parameters

        H0 = 100. * h
        Omega0_b = omega0_b/h**2
        Omega0_cdm = omega0_cdm/h**2

        prior_H0 = (60. < H0 < 80.)
        prior_Omega0_b = (0.01 < Omega0_b < 0.10)
        prior_Omega0_cdm = (0.10 < Omega0_cdm < 0.5)

        if prior_H0 and prior_Omega0_b and prior_Omega0_cdm:
            return 0
        return -np.inf

    def _ln_likelihood(self, parameters):
        M, h, omega0_b, omega0_cdm = parameters

        H0 = 100. * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2

        ln_likehood = 0

        if 'local_hubble' in self._experiments:
            model = H0
            ln_likehood += self._ln_gaussian(
                y_fit=model,
                y_target=self._data._local_hubble['y'],
                y_err=self._data._local_hubble['y_err'])

        if 'cosmic_chronometers' in self._experiments:
            model = self._cosmology.hubble(
                self._data._cosmic_chronometers['x'], parameters)
            ln_likehood += self._ln_gaussian(
                y_fit=model,
                y_target=self._data._cosmic_chronometers['y'],
                y_err=self._data._cosmic_chronometers['y_err'])

        if 'jla' in self._experiments:
            model = self._observables.distance_modulus(
                self._data._jla['x'], parameters)
            ln_likehood += self._ln_multival_gaussian(
                y_fit=model,
                y_target=self._data._jla['y'],
                y_cov=self._data._jla['cov'])

        if 'bao_compilation' in self._experiments:
            model = self._observables.d_BAO(
                self._data._bao_compilation['x'], parameters)
            ln_likehood += self._ln_gaussian(
                y_fit=model,
                y_target=self._data._bao_compilation['y'],
                y_err=self._data._bao_compilation['y_err'])

        if 'bao_wigglez' in self._experiments:
            model = self._observables.d_bao_wigglez(
                self._data._bao_wigglez['x'], parameters)
            ln_likehood += self._ln_multival_gaussian(
                y_fit=model,
                y_target=self._data._bao_wigglez['y'],
                y_cov=self._data._bao_wigglez['inv_cov'],
                is_inv_cov=True)

        return ln_likehood

    def _ln_gaussian(self, y_fit, y_target, y_err):
        inv_sigma2 = 1.0 / y_err**2

        r = y_target - y_fit
        chi2 = r**2 * inv_sigma2 - np.log(inv_sigma2)

        # -0.5 * (np.sum((y_target - y_fit)**2 * inv_sigma2
        # - np.log(inv_sigma2)))
        return -0.5 * np.sum(chi2)

    def _ln_multival_gaussian(self, y_fit, y_target, y_cov, is_inv_cov=False):
        # wigglez has inverse covariance matrix
        if (is_inv_cov):
            inv_cov = y_cov.reshape((y_target.shape[0], y_target.shape[0]))
        else:
            cov = y_cov.reshape((y_target.shape[0], y_target.shape[0]))
            inv_cov = np.linalg.inv(cov)
        det_inv_cov = np.linalg.det(inv_cov)

        r = y_target - y_fit
        chi2 = np.dot(r, np.dot(inv_cov, r))

        return -0.5 * (chi2 - np.log(det_inv_cov))
