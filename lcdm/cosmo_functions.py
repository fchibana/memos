import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from math import sqrt, pow, log
from lcdm import H

import constants


def inv_E(x, params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h

    return H0/H(x, params)


def comoving_distance(x, params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    c = constants.C * pow(10., -3)

    # in Mpc
    return c * quad(inv_E, 0, x, args=(params), limit=150)[0]/H0


def angular_diameter_distance(x, params):
    # in Mpc
    return comoving_distance(x, params)/(1. + x)


def luminosity_distance(x, params):
    # in Mpc
    return (1. + x) * comoving_distance(x, params)

# To be used with JLA


def distance_modulus(x, params):
    M, h, omega0_b, omega0_cdm = params

    return 5.0*np.log10(luminosity_distance(x, params)) + M


def cmb(params):
    M, h, omega0_b, omega0_cdm = params
    H0 = 100. * h
    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2
    
    omega0_m = (Omega0_b + Omega0_cdm)*(H0/100.)**2

	# Redshift at the last scattering surface (Hu & Sugyama)
    g1 = 0.0783*pow(omega0_b, -0.238)/(1. + 39.5 * pow(omega0_b, 0.763))
    g2 = 0.560/(1. + 21.1*pow(omega0_b, 1.81))
    z_star = 1048.*(1. + 0.00124*pow(omega0_b, -0.738)) * \
			(1. + g1*pow(omega0_m, g2))

    # Sound horizon at the last scattering surface in Mpc
    r_s = 144.6188494

    # in Mpc
    com_dist = comoving_distance(z_star, params)

    c = constants.C * pow(10., -3)
    # since H0 is in km/s/Mpc, we must divide by c (in km/s), so R has no units
    R = com_dist*sqrt((Omega0_b + Omega0_cdm)*H0**2)/c
    l_A = constants.PI*com_dist/r_s

    return R, l_A, omega0_b


def d_BAO(z, params):
	# BAO distance ratio
 	
    M, h, omega0_b, omega0_cdm = params

    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    omega0_m = (Omega0_b + Omega0_cdm)*h**2

    # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
    r_d = (44.5 * log(9.83 / omega0_m)) / sqrt(1. + 10 * pow(omega0_b, (3. / 4.)))
    c = constants.C * pow(10., -3)
    
    # There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units
    return pow(comoving_distance(z, params)**2 * c * z/H(z, params), 1./3.) / r_d


def wigglez(z, params):
    M, h, omega0_b, omega0_cdm = params

    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    omega0_m = (Omega0_b + Omega0_cdm)*h**2

    # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
    r_d = (44.5 * log(9.83 / omega0_m)) / \
        sqrt(1. + 10 * pow(omega0_b, (3. / 4.)))

    # Fiducial sound horizon at the drag epoch in Mpc used by the WiggleZ (1401.0358)
    r_fid = 152.3
    c = constants.C * pow(10., -3)
    # There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units
    D_V = pow(comoving_distance(z, params)**2 * c * z/H(z, params), 1./3.)

    return D_V*(r_fid/r_d)

# Alcock & Paczynski


def fap(x, params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    omega0_m = (Omega0_b + Omega0_cdm)*h**2

    # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
    r_d = (44.5 * log(9.83 / omega0_m)) / \
        sqrt(1. + 10 * pow(omega0_b, (3. / 4.)))

    # in km/s
    return H(x, params)*r_d/1000., angular_diameter_distance(x, params)/r_d


def test_cc(params):
	x, y, yerr = np.loadtxt(constants.ROOT_DIR + "/data/hubble.txt", unpack=True)
	model = H(x, params)

	plt.errorbar( x, y, yerr = yerr, fmt = ".k", label = "Data points")
	plt.plot(x,model, '-', label = "wCDM")
	plt.xlabel("$z$")
	plt.ylabel("$H(z)$ $[Mpc^{-2}]$")
	plt.legend(loc = 'upper left', prop={'size':11})
	plt.grid(True)
	plt.show()

def test_jla(params):
    x = 0.
    y = 0.
    x, y = np.loadtxt(constants.ROOT_DIR + "/data/jla_mub.txt",
                      comments='#', unpack=True)

    # import the flattened covariance values
    aux = np.loadtxt(constants.ROOT_DIR + "/data/jla_mub_covmatrix.dat")
    # dimension of the matrix
    ndim = int(sqrt(len(aux)))
    # initilize the matrix
    cov_mat = np.zeros((ndim, ndim))

    # convert flat values to matrix form
    i = 0
    for j in range(ndim):
        for k in range(ndim):
            cov_mat[j][k] = aux[i]
            i = i+1
    inv_cov_mat = np.linalg.inv(cov_mat)
    inv_det_C = np.linalg.det(inv_cov_mat)

    yerr = np.sqrt(cov_mat.diagonal())

    model = []
    for i in range(len(x)):
        model.append(distance_modulus(x[i], params))

    plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Data points")
    plt.plot(x, model, '-', label="Model I")
    plt.xscale('log')
    plt.xlabel("$z$")
    plt.ylabel(r"$\mu(z)$ $[Mpc^{-2}]$")
    plt.legend(loc='upper left', prop={'size': 11})
    plt.grid(True)
    plt.show()


def test_cmb(params):
	y, _ = np.loadtxt(constants.ROOT_DIR + "/data/cmb.txt", 
                         comments='#', unpack=True )

	model = cmb(params)
	
	print("\nTest CMB:")
	print("Data    Model   Diff")
	for i in range(len(y)):
		msg = "{0:.3f}    {1:.3f}    {2:.3f}".format(y[i], model[i], abs((y[i]-model[i])/model[i]))
		print(msg)
  

def test_bao(params):

	x, y, _ = np.loadtxt(constants.ROOT_DIR +  "/data/bao.txt", comments='#', unpack=True )

	model = []
	for i in range(len(x)):
		model.append(d_BAO(x[i], params))

	print("\nTest BAO:")
	print("Data    Model   Diff")
	for i in range(len(y)):
		msg = "{0:.3f}    {1:.3f}    {2:.3f}".format(y[i], model[i], abs((y[i]-model[i])/model[i]))
		print(msg)


def test_wigglez(params):
	x = 0.
	y = 0.
	yerr = 0.
	x, y, yerr = np.loadtxt(constants.ROOT_DIR +  "/data/wigglez.dat", comments='#', unpack=True )

	# import the flattened covariance values
	aux = np.loadtxt(constants.ROOT_DIR + "/data/wigglez_invcovmat.dat")
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

	print("\nTest BAO (WiggleZ):")
	print("Data    Model   Diff")
	for i in range(len(y)):
		msg = "{0:.3f}    {1:.3f}    {2:.3f}".format(y[i], model[i], abs((y[i]-model[i])/model[i]))
		print(msg)


def test_fap(params): 
	# LOW Z
	x, y, yerr = np.loadtxt(constants.ROOT_DIR +  "/data/lss_lowz.txt", comments='#', unpack=True )

	# import the flattened covariance values
	aux = np.loadtxt(constants.ROOT_DIR + "/data/lss_lowz_covmat.txt")
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

	print("\nTest Alcock-Paczynski (LOWZ):")
	print("Data    Model   Diff")
	for i in range(len(y)):
		msg = "{0:.3f}    {1:.3f}    {2:.3f}".format(y[i], model[i], abs((y[i]-model[i])/model[i]))
		print(msg)

	# CMASS		
	x, y, yerr = np.loadtxt(constants.ROOT_DIR +  "/data/lss_cmass.txt", comments='#', unpack=True )

	# import the flattened covariance values
	aux = np.loadtxt(constants.ROOT_DIR +  "/data/lss_cmass_covmat.txt")
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

	print("\nTest Alcock-Paczynski (CMASS):")
	print("Data    Model   Diff")
	for i in range(len(y)):
		msg = "{0:.3f}    {1:.3f}    {2:.3f}".format(y[i], model[i], abs((y[i]-model[i])/model[i]))
		print(msg)

if __name__ == "__main__":
    
    M = 25.					# JLA normalization 
    h = 0.7302 
    omega0_b = 0.022
    omega0_cdm = 0.048
    params = [M, h, omega0_b, omega0_cdm]
    
    test_cc(params)
    test_jla(params)
    test_cmb(params)
    test_bao(params)
    test_wigglez(params)
    test_fap(params)