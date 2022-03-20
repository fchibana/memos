from abc import ABCMeta, abstractmethod
import numpy as np

from itm import constants


class Cosmology(metaclass=ABCMeta):

    @abstractmethod
    def rho_cdm(self, x, params):
        pass

    @abstractmethod
    def rho_de(self, x, params):
        pass

    def rho_radiation(self, x, params):
        # M, h, omega0_b, omega0_cdm = params
        # M = params[0]
        h = params[1]
        # omega0_b = params[2]
        # omega0_cdm = params[3]

        H0 = 100. * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2
        Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        Omega0_g = constants.radiation_density(h)
        return Omega0_g * H0**2 * np.power(1. + x, 4.)
        
    def rho_baryons(self, x, params):  
        h = params[1]
        omega0_b = params[2]
        # omega0_cdm = params[3]

        H0 = 100. * h
        Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm
        
        return Omega0_b * H0**2 * np.power(1. + x, 3.)

    def hubble(self, x, params):
        # M, h, omega0_b, omega0_cdm = params
        # M = params[0]
        # h = params[1]
        # omega0_b = params[2]
        # omega0_cdm = params[3]

        # H0 = 100. * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        rho_tot = 0
        rho_tot += self.rho_radiation(x, params)
        rho_tot += self.rho_baryons(x, params)
        rho_tot += self.rho_cdm(x, params)
        rho_tot += self.rho_de(x, params)

        return np.sqrt(rho_tot)
    

class LCDM(Cosmology):
    def rho_cdm(self, x, params):
        # M = params[0]
        h = params[1]
        # omega0_b = params[2]
        omega0_cdm = params[3]

        H0 = 100. * h
        # Omega0_b = omega0_b/h**2
        Omega0_cdm = omega0_cdm/h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm
        
        return Omega0_cdm * H0**2 * np.power(1 + x, 3.)
    
    def rho_de(self, x, params):
        # M = params[0]
        h = params[1]
        omega0_b = params[2]
        omega0_cdm = params[3]

        H0 = 100. * h
        Omega0_b = omega0_b/h**2
        Omega0_cdm = omega0_cdm/h**2
        Omega0_g = constants.radiation_density(h)
        Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm
        
        return Omega0_de * H0**2
    