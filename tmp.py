# from itm.lcdm import LCDM
from itm.cosmology import LCDM
from itm.observables import Observables
import numpy as np

observable_test = Observables(cosmology=LCDM())
parameters = [24.96, 0.69, 0.022, 0.12]

M = parameters[0]
h = parameters[1]
omega0_b = parameters[2]
omega0_cdm = parameters[3]

# params = {}
# params['M'] = M
# params['H0'] = 100. * h
# params['Omega0_b'] = omega0_b / h**2
# params['Omega0_cdm'] = omega0_cdm / h**2

z = np.linspace(0.1, 3, 10)
print(observable_test._inv_E(z, parameters))

print(observable_test._comoving_distance(z, parameters))
