import numpy as np
# import pylab as pl
from math import pow, pi, sqrt

from deprecated.lcdm import H
from deprecated.cosmo_functions import distance_modulus, cmb, d_BAO, wigglez, fap


def lnprior(params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    if 60. < H0 < 80. and 0.01 < Omega0_b < 0.10 and 0.10 < Omega0_cdm < 0.5:
        return 0
    return -np.inf


def lnlike(params):
    M, h, omega0_b, omega0_cdm = params

    H0 = 100. * h
    Omega0_b = omega0_b/h**2
    Omega0_cdm = omega0_cdm/h**2

    use_H0 = 1
    use_cosmic_clock = 1
    use_jla = 1
    use_cmb = 1
    use_bao = 1
    use_fap = 1

    lnlikehood = 0

    # = H0 ====================================================================
    if use_H0 == 1:
        x = 0.
        y = 0.
        yerr = 0.
        y, yerr = np.loadtxt("data/H0.txt", comments='#', unpack=True)

        model = H0
        inv_sigma2 = 1.0/yerr**2

        lnlikehood += -0.5 * \
            (np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    # = cosmic clocks =========================================================
    if use_cosmic_clock == 1:
        x = 0.
        y = 0.
        yerr = 0.
        x, y, yerr = np.loadtxt("data/hubble.txt", unpack=True)

        model = H(x, params)

        inv_sigma2 = 1.0/yerr**2

        lnlikehood += -0.5 * \
            (np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    # = jla ====================================================================
    if use_jla == 1:
        x = 0.
        y = 0.
        x, y = np.loadtxt("data/jla_mub.txt", comments='#', unpack=True)

        # import the flattened covariance values
        aux = np.loadtxt("data/jla_mub_covmatrix.dat")
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

        model = []
        for i in range(len(x)):
            model.append(distance_modulus(x[i], params))

        r = y - model

        chi2 = np.dot(r, np.dot(inv_cov_mat, r))

        lnlikehood += -0.5 * (chi2 - np.log(inv_det_C))

    # = cmb ====================================================================
    if use_cmb == 1:
        x = 0.
        y = 0.
        yerr = 0.
        y, yerr = np.loadtxt("data/cmb.txt", comments='#', unpack=True)

        # import the flattened covariance values
        aux = np.loadtxt("data/cmb_covmat.txt")
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

        model = cmb(params)

        r = y - model

        chi2 = np.dot(r, np.dot(inv_cov_mat, r))

        lnlikehood += -0.5 * (chi2 - np.log(inv_det_C))

    # = bao ====================================================================
    if use_bao == 1:
        x = 0.
        y = 0.
        yerr = 0.
        x, y, yerr = np.loadtxt("data/bao.txt", comments='#', unpack=True)

        model = []
        for i in range(len(x)):
            model.append(d_BAO(x[i], params))

        inv_sigma2 = 1.0/yerr**2

        lnlikehood += -0.5 * \
            (np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        # WiggleZ
        x = 0.
        y = 0.
        yerr = 0.
        x, y, yerr = np.loadtxt("data/wigglez.dat", comments='#', unpack=True)

        # import the flattened covariance values
        aux = np.loadtxt("data/wigglez_invcovmat.dat")
        # dimension of the matrix
        ndim = int(sqrt(len(aux)))
        # initilize the matrix
        inv_cov_mat = np.zeros((ndim, ndim))

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

        chi2 = np.dot(r, np.dot(inv_cov_mat, r))

        lnlikehood += -0.5 * (chi2 - np.log(inv_det_C))

    # = fap ====================================================================
    if use_fap == 1:
        # LOW Z
        x = 0.
        y = 0.
        yerr = 0.
        x, y, yerr = np.loadtxt("data/lss_lowz.txt", comments='#', unpack=True)

        # import the flattened covariance values
        aux = np.loadtxt("data/lss_lowz_covmat.txt")
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

        model = fap(x[0], params)
        r = y - model

        chi2 = np.dot(r, np.dot(inv_cov_mat, r))

        lnlikehood += -0.5 * (chi2 - np.log(inv_det_C))

        # CMASS -----------------------------
        x = 0.
        y = 0.
        yerr = 0.
        x, y, yerr = np.loadtxt("data/lss_cmass.txt",
                                comments='#', unpack=True)

        # import the flattened covariance values
        aux = np.loadtxt("data/lss_cmass_covmat.txt")
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

        yerr_test = np.sqrt(cov_mat.diagonal())

        # print yerr, yerr_test

        model = fap(x[0], params)
        r = y - model

        chi2 = np.dot(r, np.dot(inv_cov_mat, r))

        lnlikehood += -0.5 * (chi2 - np.log(inv_det_C))

    # -------------------------------------------------------------------------

    return lnlikehood


def lnprob(params):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params)
