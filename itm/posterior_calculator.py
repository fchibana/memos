import numpy as np
from math import pow, pi, sqrt

# from itm.lcdm import hubble
# from itm.lcdm import H
from itm.lcdm import LCDM
from itm.cosmo_functions import distance_modulus, cmb, d_BAO, wigglez, fap

class PosteriorCalculator:

    def __init__(self, experiments) -> None:
        self._experiments = experiments
        self._cosmology = LCDM()
        
        # TODO: use DataLoader class
        if 'local_hubble' in experiments:
            print("Loading local_hubble data")
            
            y, y_err = np.loadtxt("data/H0.txt", comments='#', unpack=True)
            self._local_hubble = {'y': y, 
                                  'y_err': y_err}
        
        if 'cosmic_chronometers' in experiments:
            print("Loading cosmic_chronometers data")
            
            x, y, y_err = np.loadtxt("data/hubble.txt", unpack=True)
            self._cosmic_chronometers = {'x': x,
                                         'y': y,
                                         'y_err': y_err}
            
        # TODO: pass cosmology class
            
        
    
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

        if 60. < H0 < 80. and 0.01 < Omega0_b < 0.10 and 0.10 < Omega0_cdm < 0.5:
            return 0
        return -np.inf
    
    def _ln_likelihood(self, parameters):
        M, h, omega0_b, omega0_cdm = parameters

        H0 = 100. * h
        Omega0_b = omega0_b/h**2
        Omega0_cdm = omega0_cdm/h**2

        ln_likehood = 0

        if 'local_hubble' in self._experiments:
            model = H0
            ln_likehood += self._ln_gaussian(y_fit=model,
                                             y_target=self._local_hubble['y'],
                                             y_err=self._local_hubble['y_err'])
        
        if 'cosmic_chronometers' in self._experiments:
            model = self._cosmology.hubble(self._cosmic_chronometers['x'], parameters)
            ln_likehood += self._ln_gaussian(y_fit=model,
                                             y_target=self._cosmic_chronometers['y'],
                                             y_err=self._cosmic_chronometers['y_err'])
        
        return ln_likehood
    
    def _ln_gaussian(self, y_fit, y_target, y_err):
        inv_sigma2 = 1.0 / y_err**2
        return -0.5 * (np.sum((y_target - y_fit)**2 * inv_sigma2 - np.log(inv_sigma2)))
