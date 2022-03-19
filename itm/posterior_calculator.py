import numpy as np
from math import pow, pi, sqrt

from itm.lcdm import H
from itm.cosmo_functions import distance_modulus, cmb, d_BAO, wigglez, fap

class PosteriorCalculator:
    def __init__(self) -> None:
        pass
        # load data here? create a data loader class?
    
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

        use_H0 = 1
        use_cosmic_clock = 1
        use_jla = 1
        use_cmb = 1
        use_bao = 1
        use_fap = 1

        lnlikehood = 0

        # = H0 ====================================================================
        if use_H0 == 0:
            x = 0.
            y = 0.
            yerr = 0.
            y, yerr = np.loadtxt("data/H0.txt", comments='#', unpack=True)

            model = H0
            inv_sigma2 = 1.0/yerr**2

            lnlikehood += -0.5 * \
                (np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        # = cosmic clocks =========================================================
        if use_cosmic_clock == 0:
            x = 0.
            y = 0.
            yerr = 0.
            x, y, yerr = np.loadtxt("data/hubble.txt", unpack=True)

            model = H(x, parameters)

            inv_sigma2 = 1.0/yerr**2

            lnlikehood += -0.5 * \
                (np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
        
        return lnlikehood