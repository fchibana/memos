import numpy as np
from math import pow, pi, sqrt
from scipy.integrate import ode
import pylab as pl

# physical constants------------------------------------------------------
c = 2.99792458e5  # km/s
T_cmb = 2.7255  # K
_c_ = 2.99792458e8  # m/s
_G_ = 6.67428e-11  # Newton constant in m^3/Kg/s^2
_PI_ = 3.1415926535897932384626433832795
_Mpc_over_m_ = 3.085677581282e22  # conversion factor from meters to megaparsecs
# parameters entering in Stefan-Boltzmann constant sigma_B
_k_B_ = 1.3806504e-23
_h_P_ = 6.62606896e-34
# Stefan-Boltzmann constant in W/m^2/K^4 = Kg/K^4/s^3
sigma_B = 2.0 * pow(_PI_, 5) * pow(_k_B_, 4) / 15.0 / pow(_h_P_, 3) / pow(_c_, 2)
omega0_g = (4.0 * sigma_B / _c_ * pow(T_cmb, 4.0)) / (
    3.0 * _c_ * _c_ * 1.0e10 / _Mpc_over_m_ / _Mpc_over_m_ / 8.0 / _PI_ / _G_
)
# -----------------------------------------------------------------------------

# derivative of phi: dphi/dz = -sqrt(1+w)/[(1+z).H]
# derivative of w: dw/dz = -2.*sqrt(1.+w)w(3H.sqrt(1+w)+dlnV/dphi + sqrt( -w/(1+w) )f(Q)/V ) /[(1+z).H]

#####
##### Based on /Users/fabio/code/fchibana/tachyons/archive/models/tachyon/general/tachyon_full_v7.py
##### 



def g(t, x, params):
    M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

    H0 = 100.0 * h

    phi = x[0]
    phi_dot = x[1]
    rho_cdm = x[2]

    Hubble = H_local(t, phi, phi_dot, rho_cdm, params)
    difflnV = dlnV(t, phi, phi_dot, params)
    U = V(t, phi, phi_dot, params)
    coupling = 3.0 * beta * H0 * rho_cdm * phi_dot

    if phi_dot**2 > 1.0:
        return

    x0_out = -phi_dot / ((1.0 + t) * Hubble)
    x1_out = (
        (1.0 - phi_dot**2)
        / (((1.0 + t) * Hubble))
        * (
            3.0 * Hubble * phi_dot
            + difflnV
            + coupling * sqrt(1.0 - phi_dot**2) / (U * phi_dot)
        )
    )
    x2_out = (3.0 * Hubble * rho_cdm - coupling) / ((1.0 + t) * Hubble)

    return [x0_out, x1_out, x2_out]


def H_local(z, phi, phi_dot, rho_cdm, params):
    M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

    H0 = 100.0 * h
    Omega0_g = omega0_g / h**2
    Omega0_b = omega0_b / h**2
    Omega0_cdm = omega0_cdm / h**2

    rho_tot = 0

    # radiation:
    rho_tot += Omega0_g * H0**2 * np.power(1 + z, 4.0)

    # baryons:
    rho_tot += Omega0_b * H0**2 * np.power(1 + z, 3.0)

    # cdm:
    rho_tot += rho_cdm

    # scf:
    rho_tot += V(z, phi, phi_dot, params) / np.sqrt(1 - phi_dot**2)

    return np.sqrt(rho_tot)


def H_dot(z, phi, w, rho_cdm, params):
    M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

    H0 = 100.0 * h
    Omega0_g = omega0_g / h**2
    Omega0_b = omega0_b / h**2
    Omega0_cdm = omega0_cdm / h**2

    rho_tot = 0
    p_tot = 0

    # radiation:
    rho_tot += Omega0_g * H0**2 * np.power(1 + z, 4.0)
    p_tot += (1.0 / 3.0) * Omega0_g * H0**2 * np.power(1 + z, 4.0)

    # baryons:
    rho_tot += Omega0_b * H0**2 * np.power(1 + z, 3.0)
    p_tot += 0.0

    # cdm:
    rho_tot += rho_cdm
    p_tot += 0.0

    # scf:
    rho_tot += V(z, phi, w, params) / np.sqrt(-w)
    p_tot += -V(z, phi, w, params) * np.sqrt(-w)

    return -1.5 * (rho_tot + p_tot)


def V(z, phi, phi_dot, params):
    M, omega0_b, omega0_cdm, phi0, h, w0, beta = params
    omega0_fld = omega0_g + omega0_b + omega0_cdm
    A = 100.0 * phi0 * sqrt((h**2 - omega0_fld) * sqrt(-w0))
    return np.power(A / phi, 2)


def dlnV(z, phi, phi_dot, params):
    return -2.0 / phi


def H_LCDM(z, params):
    M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

    H0 = 100.0 * h
    Omega0_g = omega0_g / h**2
    Omega0_b = omega0_b / h**2
    Omega0_cdm = omega0_cdm / h**2

    return H0 * np.sqrt(
        Omega0_g * (1.0 + z) ** 4
        + Omega0_b * (1.0 + z) ** 3
        + Omega0_cdm * (1.0 + z) ** 3
        + Omega0_scf
    )


# def dwdz(z, phi, w, rho_cdm, params):
# 	Hubble = H_local(z, phi, w, rho_cdm, params)
# 	difflnV = dlnV(z, phi, w, params)
# 	U = V(z, phi, w, params)
# 	coupling = beta*H0 *rho_cdm *np.sqrt(-w)/U


# 	return  -(2.*w*np.sqrt(1.+w)/((1.+z)*Hubble)) * ( 3.*Hubble*np.sqrt(1.+w) + difflnV + coupling )

# def dphidz(z, phi, w, rho_cdm, params):
# 	Hubble = H_local(z, phi, w, rho_cdm, params)

# 	return  -np.sqrt(1.+w)/((1.+z)*Hubble)

# def q(z, phi, w, rho_cdm, params):
# 	return -1. - H_dot(z, phi, w, rho_cdm, params)/H_local(z, phi, w, rho_cdm, params)**2


# ============================================================================
# what to show
print_verbose = 1
plot_hubble_local = 1
plot_hubble = 1
plot_rho = 1
plot_Omegas = 1
plot_scf = 1
plot_eos = 1
plot_r = 1
plot_lambda = 1
plot_dwdz = 0
plot_dphidz = 0
plot_q = 0

# comological parameters -----------------------------------------------------
M = 2.4937000e01
omega0_b = 2.2789000e-02
omega0_cdm = 1.1728000e-01
phi0H0 = 2.5046000e00
h = 6.8209000e-01
w0 = -9.8181000e-01
beta = 1.3196000e-01
# beta = 10.

H0 = 100.0 * h
phi0 = phi0H0 / H0
Omega0_g = omega0_g / h**2
Omega0_b = omega0_b / h**2
Omega0_cdm = omega0_cdm / h**2
Omega0_scf = 1.0 - Omega0_g - Omega0_b - Omega0_cdm

params = M, omega0_b, omega0_cdm, phi0, h, w0, beta

# ------------------------------------------------------------------------------

z_ini = 0.0
phi_ini = phi0
phi_dot_ini = sqrt(1 + w0)
rho_cdm_ini = Omega0_cdm * H0**2

init = phi_ini, phi_dot_ini, rho_cdm_ini

z_max = 1600

# ode solver
# backend = 'vode'
backend = "dopri5"
solver = ode(g).set_integrator(backend, nsteps=1)
solver.set_initial_value(init, z_ini).set_f_params(params)
sol = []
while solver.t < z_max:
    solver.integrate(z_max, step=True)
    sol.append([solver.t, solver.y[0], solver.y[1], solver.y[2]])
sol = np.array(sol)
z = sol[:, 0]
phi = sol[:, 1]
phi_dot = sol[:, 2]
rho_cdm = sol[:, 3]


# ------------------------------------------------------------------------------
# analitical rho_cdm
rho_cdm_anal = Omega0_cdm * H0**2 * np.power(1.0 + z, 3)
rho_cdm_anal_coupled = rho_cdm_anal * np.exp(3.0 * beta * H0 * (phi - phi0))
w_anal = w0 / ((1.0 + z) ** 6 * (1 + w0) - w0)
gamma = 1.0 + 1.0 / w0
# rho_scf_anal = Omega0_scf*H0**2*np.sqrt( (1.- gamma*np.power(1+z,6))/(1-gamma) )
rho_scf_anal = Omega0_scf * H0**2 * np.sqrt(-w0 + (w0 + 1.0) * np.power(1.0 + z, 6))
rho_test = sqrt(1 + w0) * Omega0_scf * H0**2 * np.power(1 + z, 3)

w = np.power(phi_dot, 2) - 1

# other bg quantities
H_final = H_local(z, phi, phi_dot, rho_cdm, params)
rho_g = Omega0_g * H0**2 * np.power(1.0 + z, 4.0)
Omega_g = rho_g / H_final**2
rho_b = Omega0_b * H0**2 * np.power(1.0 + z, 3.0)
Omega_b = rho_b / H_final**2
# rho_cdm =
Omega_cdm = rho_cdm / H_final**2
Omega_m = Omega_cdm + Omega_b
rho_m = rho_cdm + rho_b
rho_scf = V(z, phi, phi_dot, params) / np.sqrt(-w)
Omega_scf = rho_scf / H_final**2
r = rho_cdm / rho_scf
lambda_scf = -beta * (1.0 + Omega_scf * w) / (Omega_scf * w)
Omega_scf_check = 1.0 - Omega_g - Omega_b - Omega_cdm
Omega_scf_r = (1.0 - Omega0_g - Omega0_b) / (1.0 + r)
Omega_cdm_r = (1.0 - Omega0_g - Omega0_b) * r / (1.0 + r)

# deceleration = q(z,phi,w,rho_cdm,params)
# z_test = np.linspace(0.6,0.7,1000)
# deceleration_test = np.interp(z_test, z, deceleration)
# z_deceleration = z_test[np.argmin(np.abs(deceleration_test))]

# # # # plots ########################################################

# if plot_hubble_local == 1:
#     pl.figure(1)
#     x, y, yerr = np.loadtxt("../../data/hubble.txt", unpack=True)
#     pl.errorbar(x, y, yerr=yerr, fmt=".k", label="Data points")
#     pl.plot(z, H_LCDM(z, params), "-", label="$\Lambda$CDM")
#     pl.plot(z, H_final, ".-", label="ode")
#     pl.xlabel("$z$")
#     pl.ylabel("$H(z)$ $[Mpc^{-2}]$")
#     pl.axis([0, 2, 0, 250])
#     pl.legend(loc="upper left", prop={"size": 11})
#     pl.grid(True)

if plot_hubble == 1:
    pl.figure(2)
    pl.loglog(z, H_LCDM(z, params), "-", label="$\Lambda$CDM")
    pl.loglog(z, H_final, ".-", label="ode")
    pl.xlabel("$z$")
    pl.ylabel("$H(z)$ $[Mpc^{-2}]$")
    pl.legend(loc="lower left", prop={"size": 11})
    pl.grid(True)

if plot_rho == 1:
    pl.figure(3)
    # pl.loglog(z, rho_g, '-y', label = "g")
    # pl.loglog(z, rho_b, '-b', label = "b")
    pl.loglog(z, rho_cdm, "-g", label="cdm")
    # pl.loglog(z, rho_cdm_anal, '-m', label = "cdm_expected")
    pl.loglog(z, rho_cdm_anal_coupled, "-y", label="cdm_expected_coupled")
    # pl.loglog(z, rho_scf, '-r', label = "$scf$")
    # pl.loglog(z, rho_scf_anal, '-k', label = "$scf anal$")
    # pl.loglog(z, rho_test, '-b', label = "$scf test$")
    # pl.xlabel("$z$")
    pl.ylabel(r"$\rho$")
    pl.legend(loc="upper left", prop={"size": 11})
    pl.grid(True)

if plot_Omegas == 1:
    pl.figure(4)
    pl.loglog(z, Omega_g, "-y", label="Radiation")
    pl.loglog(z, Omega_b, "-b", label="Baryons")
    pl.loglog(z, Omega_cdm, "-g", label="CDM")
    pl.loglog(z, Omega_scf, "-r", label="Tachyon")
    # pl.loglog(z, Omega_scf_check, '-k', label = "Tachyon Check")
    # pl.loglog(z, Omega_cdm_r, '-y', label = r"$\Omega_m$ Check")
    # pl.loglog(z, Omega_scf_r, '-b', label = r"$\Omega_\phi$ Check")
    pl.xlabel("$z$", fontsize=20)
    pl.ylabel("$\Omega$", fontsize=20)
    pl.axis([0, z_max, 0, 1.1])
    pl.legend(loc="lower right", prop={"size": 12})
    pl.grid(True)

if plot_scf == 1:
    pl.figure(5)
    pl.loglog(z, phi, "-b", label="$\phi$")
    pl.xlabel("$z$")
    pl.legend(loc="upper left", prop={"size": 11})
    pl.grid(True)

if plot_eos == 1:
    pl.figure(6)
    pl.plot(z, w, "-g", label="Numerical")
    pl.plot(z, w_anal, ".b", label="Analytical approximation")
    pl.xscale("log")
    pl.xlabel("$z$", fontsize=20)
    pl.ylabel("$w(z)$", fontsize=20)
    pl.legend(loc="lower right", prop={"size": 11})
    pl.axis([0, z_max, -1, 0.1])
    pl.grid(True)

if plot_r == 1:
    pl.figure(7)
    pl.plot(z, r, "-g", label=r"$\rho_{cdm}/\rho{\phi}$")
    pl.xscale("log")
    pl.xlabel("$z$")
    pl.legend(loc="upper left", prop={"size": 11})
    pl.grid(True)

if plot_lambda == 1:
    pl.figure(8)
    pl.loglog(z, lambda_scf, "-", label=r"$\lambda_{scf}$")
    # pl.xscale('log')
    pl.xlabel("$z$")
    pl.legend(loc="upper left", prop={"size": 11})
    pl.grid(True)

# if plot_dwdz == 1:
#     pl.figure(9)
#     pl.plot(z, dwdz(z, phi, w, rho_cdm, params), "-", label=r"$dw/dz$")
#     pl.xscale("log")
#     pl.xlabel("$z$")
#     pl.legend(loc="upper left", prop={"size": 11})
#     pl.grid(True)

# if plot_dphidz == 1:
#     pl.figure(10)
#     pl.plot(z, dphidz(z, phi, w, rho_cdm, params), "-", label=r"$dphi/dz$")
#     pl.xscale("log")
#     pl.xlabel("$z$")
#     pl.legend(loc="upper left", prop={"size": 11})
#     pl.grid(True)

# if plot_q == 1:
#     pl.figure(10)
#     pl.plot(z, deceleration, "-b")
#     pl.plot(
#         (0, z_max),
#         (0, 0),
#         ":k",
#     )
    # pl.plot((0,z_deceleration), (0,0), ':k', )
    # pl.plot((z_deceleration,z_deceleration), (-0.5,0), ':k', label ="$z_{decel} = %.2f$" %z_deceleration )
    pl.xlabel("$z$", fontsize=20)
    pl.ylabel("$q$", fontsize=20)
    pl.legend(loc="lower right", prop={"size": 15})
    # pl.axis([0, 1, -0.43, 0.2])
    pl.grid(False)


# pl.figure(11)
# pl.plot(z,3.*H_final*np.sqrt(1 + w), '-b')
# # pl.plot((0,z_max), (0,0), ':k', )
# # pl.plot((0,z_deceleration), (0,0), ':k', )
# # pl.plot((z_deceleration,z_deceleration), (-0.5,0), ':k', label ="$z_{decel} = %.2f$" %z_deceleration )
# pl.xlabel("$z$",  fontsize=20)
# pl.ylabel("$q$",  fontsize=20)
# pl.legend(loc = 'lower right', prop={'size':15})
# # pl.axis([0, 1, -0.43, 0.2])
# pl.grid(False)
pl.show()
