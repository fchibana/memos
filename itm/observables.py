class Observables():
  def __init__(self):
    print("Constructing Observables")
    self._m_jla = 24.96
    self._h = 0.69
    self._omega_b = 0.022
    self._omega_cdm = 0.12

  def update_cosmological_parameter(self, params):
    self._m_jla = params['m_jla']
    self._h = params['h']
    self._omega_b = params['omega_b']
    self._omega_cdm = params['omega_cdm']
    
    self._hubble0 = 100. * self._h
  
  def _inv_hubble(self, z, params):
    return self._hubble0 / H(z, params)
    