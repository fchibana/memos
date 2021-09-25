import numpy as np
from math import pow, pi, sqrt

# physical constants------------------------------------------------------
c = 2.99792458e5       						# km/s
T_cmb = 2.7255								# K 
_c_ = 2.99792458e8       					# m/s
_G_ = 6.67428e-11        					# Newton constant in m^3/Kg/s^2
_PI_ = 3.1415926535897932384626433832795
_Mpc_over_m_ = 3.085677581282e22        	# conversion factor from meters to megaparsecs
# parameters entering in Stefan-Boltzmann constant sigma_B 
_k_B_ = 1.3806504e-23
_h_P_ = 6.62606896e-34
# Stefan-Boltzmann constant in W/m^2/K^4 = Kg/K^4/s^3
sigma_B = 2. * pow(_PI_,5) * pow(_k_B_,4) / 15. / pow(_h_P_,3) / pow(_c_,2)  
#-----------------------------------------------------------------------------

def H(z, params):
	M, h, omega0_b, omega0_cdm = params

	H0 = 100. *h
	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2

	Omega0_g = (4.*sigma_B/_c_*pow(T_cmb,4.)) / (3.*_c_*_c_*1.e10*h*h/_Mpc_over_m_/_Mpc_over_m_/8./_PI_/_G_)
	Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm
	

	rho_tot = 0

	#radiation:
	rho_tot += Omega0_g*H0**2 *np.power(1.+z,4.)
	
	# baryons:
	rho_tot += Omega0_b*H0**2 *np.power(1.+z,3.)

	# cdm:
	rho_tot += Omega0_cdm*H0**2 *np.power(1+z,3.)
	
	# scf:
	rho_tot += Omega0_de*H0**2
	
	return np.sqrt(rho_tot)