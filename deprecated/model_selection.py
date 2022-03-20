import numpy as np
import os

from likelihood import lnlike

file_name = 'lcdm'

# best-fit parameters (from getdist) (M, H0, Omega_b, Omega_cmd, w)
params = [2.4953000E+01, 6.8061000E-01, 2.2221000E-02, 1.2055000E-01]

# number of free parameters
k = 4

# number of data points
n = 74.

chi2 = -2.*lnlike(params) 
red_chi2 = chi2/(n-k)
aic = chi2 + 2.*k
bic = chi2 + k*np.log(n)

results = np.array([chi2, red_chi2,k, aic, bic])

header_string = 'chi2,reduced chi2, k, aic, bic'
np.savetxt("results/"+file_name+".modelselec", results, fmt='%1.6e', header=header_string)