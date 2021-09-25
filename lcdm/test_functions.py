import numpy as np
# from scipy.integrate import quad
from math import sqrt, pow 
import matplotlib.pyplot as plt

from lcdm import H
# from cosmo_functions import distance_modulus, cmb, d_BAO, wigglez, fap


# physical constants------------------------------------------------------
c = 2.99792458e5       						# km/s
T_cmb = 2.7255								# K 
_c_ = 2.99792458e8       					# m/s
_G_ = 6.67428e-11        					# Newton constant in m^3/Kg/s^2
_PI_ = 3.1415926535897932384626433832795
_Mpc_over_m_ = 3.085677581282e22         	# conversion factor from meters to megaparsecs
# parameters entering in Stefan-Boltzmann constant sigma_B 
_k_B_ = 1.3806504e-23
_h_P_ = 6.62606896e-34
# Stefan-Boltzmann constant in W/m^2/K^4 = Kg/K^4/s^3
sigma_B = 2. * pow(_PI_,5) * pow(_k_B_,4) / 15. / pow(_h_P_,3) / pow(_c_,2)  
#-----------------------------------------------------------------------------

# comological parameters -----------------------------------------------------
M = 25.					# JLA normalization 
h = 0.7302 
omega0_b = 0.022
omega0_cdm = 0.048

H0 = 100. *h
Omega0_b = omega0_b/h**2
Omega0_cdm = omega0_cdm/h**2

Omega0_g = (4.*sigma_B/_c_*pow(T_cmb,4.)) / (3.*_c_*_c_*1.e10*h*h/_Mpc_over_m_/_Mpc_over_m_/8./_PI_/_G_)
Omega0_de = 1. - Omega0_g - Omega0_b - Omega0_cdm

params = [M, h, omega0_b, omega0_cdm]

#------------------------------------------------------------------------------

test_hubble = 1
test_jla = 0
test_cmb = 0
test_bao = 0
test_wigglez = 0
test_fap = 0

# tests -----------------------------------------------------------------------

if test_hubble == 1:
    # TODO(me): add project root directory variable
	x, y, yerr = np.loadtxt("/Users/fabio/code/fchibana/memos/data/hubble.txt", unpack=True)
	model = H(x, params)

	plt.errorbar( x, y, yerr = yerr, fmt = ".k", label = "Data points")
	plt.plot(x,model, '-', label = "wCDM")
	plt.xlabel("$z$")
	plt.ylabel("$H(z)$ $[Mpc^{-2}]$")
	plt.legend(loc = 'upper left', prop={'size':11})
	plt.grid(True)
	plt.show()

# if test_jla == 1:
# 	x = 0.
# 	y = 0.
# 	x, y = np.loadtxt( "../../data/jla_mub.txt", comments='#', unpack=True )

# 	# import the flattened covariance values
# 	aux = np.loadtxt("../../data/jla_mub_covmatrix.dat")
# 	# dimension of the matrix
# 	ndim = int(sqrt(len(aux)))
# 	# initilize the matrix
# 	cov_mat = np.zeros((ndim,ndim))

# 	# convert flat values to matrix form
# 	i = 0
# 	for j in range(ndim):
# 		for k in range(ndim):
# 			cov_mat[j][k] = aux[i]
# 			i = i+1
# 	inv_cov_mat = np.linalg.inv(cov_mat)
# 	inv_det_C = np.linalg.det(inv_cov_mat)

# 	yerr = np.sqrt(cov_mat.diagonal())

# 	model = []
# 	for i in range(len(x)):
# 		model.append(distance_modulus(x[i], params))

# 	pl.errorbar( x, y, yerr = yerr, fmt = ".k", label = "Data points")	
# 	pl.plot(x,model, '-', label = "Model I")
# 	pl.xscale('log')
# 	pl.xlabel("$z$")
# 	pl.ylabel(r"$\mu(z)$ $[Mpc^{-2}]$")
# 	pl.legend(loc = 'upper left', prop={'size':11})
# 	pl.grid(True)
# 	pl.show()

# if test_cmb == 1:
# 	x = 0.
# 	y = 0.
# 	yerr = 0.
# 	y, yerr = np.loadtxt( "../../data/cmb.txt", comments='#', unpack=True )

# 	model = cmb(params)
	
# 	print "\nTest CMB:"
# 	print "Data    Model   Diff"
# 	for i in range(len(y)):
# 		print '%.3f   %.3f   %.3f' %(y[i], model[i], abs((y[i]-model[i])/model[i]))

# if test_bao == 1:
# 	x = 0.
# 	y = 0.
# 	yerr = 0.
# 	x, y, yerr = np.loadtxt( "../../data/bao.txt", comments='#', unpack=True )

# 	model = []
# 	for i in range(len(x)):
# 		model.append(d_BAO(x[i], params))

# 	print "\nTest BAO:"
# 	print "Data     Model    Diff"
# 	for i in range(len(x)):
# 		print '%.3f   %.3f   %.3f' %(y[i], model[i], abs((y[i]-model[i])/model[i]))

# if test_wigglez == 1:
# 	x = 0.
# 	y = 0.
# 	yerr = 0.
# 	x, y, yerr = np.loadtxt( "../../data/wigglez.dat", comments='#', unpack=True )

# 	# import the flattened covariance values
# 	aux = np.loadtxt("../../data/wigglez_invcovmat.dat")
# 	# dimension of the matrix
# 	ndim = int(sqrt(len(aux)))
# 	# initilize the matrix
# 	inv_cov_mat = np.zeros((ndim,ndim))

# 	# convert flat values to matrix form
# 	i = 0
# 	for j in range(ndim):
# 		for k in range(ndim):
# 			inv_cov_mat[j][k] = aux[i]
# 			i = i+1
# 	inv_det_C = np.linalg.det(inv_cov_mat)

# 	model = []
# 	for i in range(len(x)):
# 		model.append(wigglez(x[i], params))

# 	print "\nTest BAO (WiggleZ):"
# 	print "Data     Model    Diff"
# 	for i in range(len(x)):
# 		print '%.3f   %.3f   %.3f' %(y[i], model[i], abs((y[i]-model[i])/model[i]))

# if test_fap == 1:
# 	# LOW Z
# 	x = 0.
# 	y = 0.
# 	yerr = 0.
# 	x, y, yerr = np.loadtxt( "../../data/lss_lowz.txt", comments='#', unpack=True )

# 	# import the flattened covariance values
# 	aux = np.loadtxt("../../data/lss_lowz_covmat.txt")
# 	# dimension of the matrix
# 	ndim = int(sqrt(len(aux)))
# 	# initilize the matrix
# 	cov_mat = np.zeros((ndim,ndim))

# 	# convert flat values to matrix form
# 	i = 0
# 	for j in range(ndim):
# 		for k in range(ndim):
# 			cov_mat[j][k] = aux[i]
# 			i = i+1
# 	inv_cov_mat = np.linalg.inv(cov_mat)
# 	inv_det_C = np.linalg.det(inv_cov_mat)

# 	yerr_test = np.sqrt(cov_mat.diagonal())
	
# 	# print yerr, yerr_test

# 	model  = fap(x[0], params)

# 	print "\nTest Alcock-Paczynski (LOWZ):"
# 	print "Data   Model  Diff"
# 	for i in range(len(x)):
# 		print '%.1f   %.1f   %.3f' %(y[i], model[i], abs((y[i]-model[i])/model[i]))

# 	# CMASS		
# 	x = 0.
# 	y = 0.
# 	yerr = 0.
# 	x, y, yerr = np.loadtxt( "../../data/lss_cmass.txt", comments='#', unpack=True )

# 	# import the flattened covariance values
# 	aux = np.loadtxt("../../data/lss_cmass_covmat.txt")
# 	# dimension of the matrix
# 	ndim = int(sqrt(len(aux)))
# 	# initilize the matrix
# 	cov_mat = np.zeros((ndim,ndim))

# 	# convert flat values to matrix form
# 	i = 0
# 	for j in range(ndim):
# 		for k in range(ndim):
# 			cov_mat[j][k] = aux[i]
# 			i = i+1
# 	inv_cov_mat = np.linalg.inv(cov_mat)
# 	inv_det_C = np.linalg.det(inv_cov_mat)

# 	yerr_test = np.sqrt(cov_mat.diagonal())
	
# 	# print yerr, yerr_test

# 	model  = fap(x[0], params)

# 	print "\nTest Alcock-Paczynski (cmass):"
# 	print "Data   Model  Diff"
# 	for i in range(len(x)):
# 		print '%.1f   %.1f   %.3f' %(y[i], model[i], abs((y[i]-model[i])/model[i]))