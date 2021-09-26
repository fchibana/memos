import numpy as np

from itm import constants


def H(z, params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2
    Omega0_g = constants.radiation_density(h)
    Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

    rho_tot = 0

    # radiation:
    rho_tot += Omega0_g*H0**2 * np.power(1.+z, 4.)

    # baryons:
    rho_tot += Omega0_b*H0**2 * np.power(1.+z, 3.)

    # cdm:
    rho_tot += Omega0_cdm*H0**2 * np.power(1+z, 3.)

    # scf:
    rho_tot += Omega0_de*H0**2

    return np.sqrt(rho_tot)
