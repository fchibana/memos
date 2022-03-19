from scipy.integrate import quad
import math
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
    
  def d_BAO(self, x, params):
    """BAO distance ratio

    Args:
        x (_type_): _description_
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    M, h, omega0_b, omega0_cdm = params

    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    omega0_m = (Omega0_b + Omega0_cdm)*h**2

    # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
    # TODO: where did this come from??
    rs = (44.5 * math.log(9.83 / omega0_m)) / math.sqrt(1. + 10 * pow(omega0_b, (3. / 4.)))
    
    c = constants.C * pow(10., -3)  # km/s
    hubble = self._cosmology.hubble(x, params)  # km/s/Mpc
    dc2 = self._comoving_distance(x, params)**2
    
    # dilation scale
    dv = np.power((c * x / hubble) * dc2, 1./3.) 
    
    # There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units  
    d_bao = dv / rs
        
    return d_bao
    