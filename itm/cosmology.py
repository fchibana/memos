from abc import ABCMeta, abstractmethod
import numpy as np

from itm import constants


class Cosmology(metaclass=ABCMeta):
    @property
    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def rho_cdm(self, x, parameters):
        pass

    @abstractmethod
    def rho_de(self, x, parameters):
        pass

    def rho_radiation(self, x, parameters):
        # M, h, omega0_b, omega0_cdm = parameters
        # M = parameters[0]
        h = parameters[1]
        # omega0_b = parameters[2]
        # omega0_cdm = parameters[3]

        H0 = 100.0 * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2
        Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        Omega0_g = constants.radiation_density(h)
        return Omega0_g * H0**2 * np.power(1.0 + x, 4.0)

    def rho_baryons(self, x, parameters):
        h = parameters[1]
        omega0_b = parameters[2]
        # omega0_cdm = parameters[3]

        H0 = 100.0 * h
        Omega0_b = omega0_b / h**2
        # Omega0_cdm = omega0_cdm/h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        return Omega0_b * H0**2 * np.power(1.0 + x, 3.0)

    def hubble(self, x, parameters):
        # M, h, omega0_b, omega0_cdm = parameters
        # M = parameters[0]
        # h = parameters[1]
        # omega0_b = parameters[2]
        # omega0_cdm = parameters[3]

        # H0 = 100. * h
        # Omega0_b = omega0_b/h**2
        # Omega0_cdm = omega0_cdm/h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        rho_tot = 0
        rho_tot += self.rho_radiation(x, parameters)
        rho_tot += self.rho_baryons(x, parameters)
        rho_tot += self.rho_cdm(x, parameters)
        rho_tot += self.rho_de(x, parameters)

        return np.sqrt(rho_tot)


class LCDM(Cosmology):
    _name = "lcdm"

    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return self._name

    def rho_cdm(self, x, parameters):
        # M = parameters[0]
        h = parameters[1]
        # omega0_b = parameters[2]
        omega0_cdm = parameters[3]

        H0 = 100.0 * h
        # Omega0_b = omega0_b/h**2
        Omega0_cdm = omega0_cdm / h**2
        # Omega0_g = constants.radiation_density(h)
        # Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

        return Omega0_cdm * H0**2 * np.power(1 + x, 3.0)

    def rho_de(self, x, parameters):
        # M = parameters[0]
        h = parameters[1]
        omega0_b = parameters[2]
        omega0_cdm = parameters[3]

        H0 = 100.0 * h
        Omega0_b = omega0_b / h**2
        Omega0_cdm = omega0_cdm / h**2
        Omega0_g = constants.radiation_density(h)
        Omega0_de = 1.0 - Omega0_g - Omega0_b - Omega0_cdm

        return Omega0_de * H0**2
