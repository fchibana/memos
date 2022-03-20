import numpy as np
import pylab as pl
from math import pow, pi, sqrt

from lcdm import H
from cosmo_functions import distance_modulus, cmb, d_BAO, wigglez, fap

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
omega0_g = (4.*sigma_B/_c_*pow(T_cmb,4.)) / (3.*_c_*_c_*1.e10/_Mpc_over_m_/_Mpc_over_m_/8./_PI_/_G_)    
#-----------------------------------------------------------------------------

def chi_squared(params):

	M, h, omega0_b, omega0_cdm = params

	H0 = 100. *h
	Omega0_g = omega0_g/h**2
	Omega0_b = omega0_b/h**2
	Omega0_cdm = omega0_cdm/h**2
	
	use_H0 = 1
	use_cosmic_clock = 1
	use_jla = 1
	use_bao = 1
	use_fap = 1
	use_cmb = 1

	
	sum_chi2 = 0.

	#= H0 ====================================================================
	if use_H0 == 1:
		x = 0.
		y = 0.
		yerr = 0.
		y, yerr = np.loadtxt( "../../data/H0.txt", comments='#', unpack=True )

		model = H0
		inv_sigma2 = 1.0/yerr**2

		chi2 = (y-model)**2*inv_sigma2
		sum_chi2 =+ sum_chi2 + chi2

		print 'H0'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2
		
	#= cosmic clocks =========================================================
	if use_cosmic_clock == 1:
		x = 0.
		y = 0.
		yerr = 0.
		x, y, yerr = np.loadtxt("../../data/hubble.txt", unpack=True)

		model = H(x, params)
		inv_sigma2 = 1.0/yerr**2

		chi2 = np.sum( (y-model)**2*inv_sigma2 )
		sum_chi2 =+ sum_chi2 + chi2

		print 'CC'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

	#= jla ====================================================================
	if use_jla == 1:
		x = 0.
		y = 0.
		x, y = np.loadtxt( "../../data/jla_mub.txt", comments='#', unpack=True )

		# import the flattened covariance values
		aux = np.loadtxt("../../data/jla_mub_covmatrix.dat")
		# dimension of the matrix
		ndim = int(sqrt(len(aux)))
		# initilize the matrix
		cov_mat = np.zeros((ndim,ndim))

		# convert flat values to matrix form
		i = 0
		for j in range(ndim):
			for k in range(ndim):
				cov_mat[j][k] = aux[i]
				i = i+1
		inv_cov_mat = np.linalg.inv(cov_mat)
		inv_det_C = np.linalg.det(inv_cov_mat)

		model = []
		for i in range(len(x)):
			model.append(distance_modulus(x[i], params))

		r = y - model

		chi2 = np.dot(r, np.dot(inv_cov_mat,r))
		sum_chi2 =+ sum_chi2 + chi2

		print 'jla'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

	#= cmb ====================================================================
	if use_cmb == 1:
		x = 0.
		y = 0.
		yerr = 0.
		y, yerr = np.loadtxt( "../../data/cmb.txt", comments='#', unpack=True )

		# import the flattened covariance values
		aux = np.loadtxt("../../data/cmb_covmat.txt")
		# dimension of the matrix
		ndim = int(sqrt(len(aux)))
		# initilize the matrix
		cov_mat = np.zeros((ndim,ndim))

		# convert flat values to matrix form
		i = 0
		for j in range(ndim):
			for k in range(ndim):
				cov_mat[j][k] = aux[i]
				i = i+1
		inv_cov_mat = np.linalg.inv(cov_mat)
		inv_det_C = np.linalg.det(inv_cov_mat)
		
		model = cmb(params)
		
		r = y - model

		chi2 = np.dot(r, np.dot(inv_cov_mat,r))
		sum_chi2 =+ sum_chi2 + chi2

		print 'CMB'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

		#= bao ====================================================================
	if use_bao == 1:
		x = 0.
		y = 0.
		yerr = 0.
		x, y, yerr = np.loadtxt( "../../data/bao.txt", comments='#', unpack=True )

		model = []
		for i in range(len(x)):
			model.append(d_BAO(x[i], params))

		inv_sigma2 = 1.0/yerr**2
			
		chi2 = np.sum( (y-model)**2*inv_sigma2 )
		sum_chi2 =+ sum_chi2 + chi2

		print 'BAO'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2
				
		# WiggleZ-------------------------------
		x = 0.
		y = 0.
		yerr = 0.
		x, y, yerr = np.loadtxt( "../../data/wigglez.dat", comments='#', unpack=True )

		# import the flattened covariance values
		aux = np.loadtxt("../../data/wigglez_invcovmat.dat")
		# dimension of the matrix
		ndim = int(sqrt(len(aux)))
		# initilize the matrix
		inv_cov_mat = np.zeros((ndim,ndim))

		# convert flat values to matrix form
		i = 0
		for j in range(ndim):
			for k in range(ndim):
				inv_cov_mat[j][k] = aux[i]
				i = i+1
		inv_det_C = np.linalg.det(inv_cov_mat)

		model = []
		for i in range(len(x)):
			model.append(wigglez(x[i], params))

		r = y - model

		chi2 = np.dot(r, np.dot(inv_cov_mat,r))
		sum_chi2 =+ sum_chi2 + chi2

		print 'WiggleZ'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

	#= fap ====================================================================
	if use_fap == 1:
		# LOW Z
		x = 0.
		y = 0.
		yerr = 0.
		x, y, yerr = np.loadtxt( "../../data/lss_lowz.txt", comments='#', unpack=True )

		# import the flattened covariance values
		aux = np.loadtxt("../../data/lss_lowz_covmat.txt")
		# dimension of the matrix
		ndim = int(sqrt(len(aux)))
		# initilize the matrix
		cov_mat = np.zeros((ndim,ndim))

		# convert flat values to matrix form
		i = 0
		for j in range(ndim):
			for k in range(ndim):
				cov_mat[j][k] = aux[i]
				i = i+1
		inv_cov_mat = np.linalg.inv(cov_mat)
		inv_det_C = np.linalg.det(inv_cov_mat)

		model  = fap(x[0], params)
		r = y - model

		chi2 = np.dot(r, np.dot(inv_cov_mat,r))
		sum_chi2 =+ sum_chi2 + chi2

		print 'FAP-lowz'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

		# CMASS -----------------------------
		x = 0.
		y = 0.
		yerr = 0.
		x, y, yerr = np.loadtxt( "../../data/lss_cmass.txt", comments='#', unpack=True )

		# import the flattened covariance values
		aux = np.loadtxt("../../data/lss_cmass_covmat.txt")
		# dimension of the matrix
		ndim = int(sqrt(len(aux)))
		# initilize the matrix
		cov_mat = np.zeros((ndim,ndim))

		# convert flat values to matrix form
		i = 0
		for j in range(ndim):
			for k in range(ndim):
				cov_mat[j][k] = aux[i]
				i = i+1
		inv_cov_mat = np.linalg.inv(cov_mat)
		inv_det_C = np.linalg.det(inv_cov_mat)

		yerr_test = np.sqrt(cov_mat.diagonal())
		
		# print yerr, yerr_test

		model  = fap(x[0], params)
		r = y - model

		chi2 = np.dot(r, np.dot(inv_cov_mat,r))
		sum_chi2 =+ sum_chi2 + chi2

		print 'FAP-cmass'
		print 'chi2 = ', chi2
		print 'sum_chi2 = ', sum_chi2

		
	#-------------------------------------------------------------------------
		
	return sum_chi2



file_name = 'cc+sne+bao+lss+cmb'

# best-fit parameters (from getdist) (M, Omega_b, Omega_cmd, phi0, A, w0, beta)
M 			= 2.4953000E+01
h 			= 6.8061000E-01
omega0_b 	= 2.2221000E-02
omega0_cdm 	= 1.2055000E-01
# phi0		= 3.6719000E-02
# w0 			= -0.98181
# beta 		= 0.13196

params = [M, h, omega0_b, omega0_cdm]

# number of free parameters
k = 4.

# number of data points
n = 74.

chi2 = chi_squared(params) 
red_chi2 = chi2/(n-k)
aic = chi2 + 2.*k
bic = chi2 + k*np.log(n)

results = np.array([chi2, red_chi2,k, aic, bic])

header_string = 'chi2,reduced chi2, k, aic, bic'
np.savetxt("results/"+file_name+".modelselec", results, fmt='%1.6e', header=header_string)