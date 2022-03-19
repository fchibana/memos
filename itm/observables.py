from scipy.integrate import quad
import numpy as np

from itm import constants
from itm.lcdm import LCDM

class Observables():
  def __init__(self, cosmology: LCDM):
    # print("Constructing Observables")
    # self._m_jla = 24.96
    # self._h = 0.69
    # self._omega_b = 0.022
    # self._omega_cdm = 0.12
    self._cosmology = cosmology

  # def update_cosmological_parameter(self, params):
  #   self._m_jla = params['m_jla']
  #   self._h = params['h']
  #   self._omega_b = params['omega_b']
  #   self._omega_cdm = params['omega_cdm']
    
  #   self._hubble0 = 100. * self._h
  
  # def _inv_hubble(self, z, params):
  #   return self._hubble0 / H(z, params)
  
  def _inv_E(self, x, params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    return H0/self._cosmology.hubble(x, params)

  def _comoving_distance(self, x, params):
    assert x.ndim == 1, f"input must be vector. got: {x.ndim}"
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    c = constants.C * pow(10., -3)
    
    dc_i = []
    for x_i in x:
      dc_i.append(quad(self._inv_E, 0, x_i, args=(params), limit=150)[0])

    # in Mpc
    return c * np.asarray(dc_i) / H0
  
  def _angular_diameter_distance(self, x, params):
    
    # in Mpc
    return self._comoving_distance(x, params) / (1. + x)

  def _luminosity_distance(self, x, params):
      # in Mpc
      return (1. + x) * self._comoving_distance(x, params)

  def distance_modulus(self, x, params):
      M, h, omega0_b, omega0_cdm = params

      return 5.0 * np.log10(self._luminosity_distance(x, params)) + M
    