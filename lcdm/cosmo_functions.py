import numpy as np
from scipy.integrate import quad
from math import sqrt, pow, log
from lcdm import H

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

def inv_E(x,params):
	M, h, omega0_b, omega0_cdm = params

	H0 = 100. *h
	
	return H0/H(x,params)

def comoving_distance(x, params):
	M, h, omega0_b, omega0_cdm = params

	H0 = 100. *h

	# in Mpc
	return c *quad(inv_E, 0, x, args = (params), limit = 150)[0]/H0

def angular_diameter_distance(x, params):
	# in Mpc
	return comoving_distance(x, params)/(1. + x)

def luminosity_distance(x, params):
	# in Mpc
	return (1. + x) *comoving_distance(x, params)

#To be used with JLA
def distance_modulus(x, params):
	M, h, omega0_b, omega0_cdm = params

	return 5.0*np.log10(luminosity_distance(x, params)) + M 

# Shift parameter and omega0_b
def cmb(params):
	M, h, omega0_b, omega0_cdm = params

	H0 = 100. *h
	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2

	Omega0_g = (4.*sigma_B/_c_*pow(T_cmb,4.)) / (3.*_c_*_c_*1.e10*h*h/_Mpc_over_m_/_Mpc_over_m_/8./_PI_/_G_)

	omega0_m = (Omega0_b + Omega0_cdm)*(H0/100.)**2
	
	# Redshift at the last scattering surface (Hu & Sugyama)
	g1 = 0.0783*pow(omega0_b, -0.238)/(1. + 39.5 *pow(omega0_b, 0.763))	
	g2 = 0.560/(1. + 21.1*pow(omega0_b, 1.81))
	z_star = 1048.*(1. + 0.00124*pow(omega0_b, -0.738))*(1. + g1*pow(omega0_m, g2))    
	
	# Sound horizon at the last scattering surface in Mpc
	r_s = 144.6188494
	
	# in Mpc
	com_dist = comoving_distance(z_star, params)
	
	# since H0 is in km/s/Mpc, we must divide by c (in km/s), so R has no units
	R = com_dist*sqrt((Omega0_b + Omega0_cdm)*H0**2)/c
	l_A = _PI_*com_dist/r_s
	
	return R, l_A, omega0_b

# BAO distance ratio
def d_BAO(z, params):
	M, h, omega0_b, omega0_cdm = params

	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2

	omega0_m = (Omega0_b + Omega0_cdm)*h**2

	# Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
	r_d = (44.5 *log(9.83 /omega0_m))/sqrt(1. + 10 *pow(omega0_b, (3. /4.)))
	
	# There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units
	return pow(comoving_distance(z, params)**2 *c *z/H(z, params), 1./3.) /r_d 

def wigglez(z, params):
	M, h, omega0_b, omega0_cdm = params

	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2

	omega0_m = (Omega0_b + Omega0_cdm)*h**2

	# Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
	r_d = (44.5 *log(9.83 /omega0_m))/sqrt(1. + 10 *pow(omega0_b, (3. /4.)))

	# Fiducial sound horizon at the drag epoch in Mpc used by the WiggleZ (1401.0358)
	r_fid = 152.3

	# There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units
	D_V = pow(comoving_distance(z, params)**2 *c *z/H(z, params), 1./3.) 
	
	return D_V*(r_fid/r_d) 

# Alcock & Paczynski
def fap(x, params):
	M, h, omega0_b, omega0_cdm = params
	
	H0 = 100. *h
	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2

	omega0_m = (Omega0_b + Omega0_cdm)*h**2

	# Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
	r_d = (44.5 *log(9.83 /omega0_m))/sqrt(1. + 10 *pow(omega0_b, (3. /4.)))

	# in km/s
	return H(x, params)*r_d/1000. , angular_diameter_distance(x, params)/r_d