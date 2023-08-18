import matplotlib.pyplot as plt
import numpy as np
from math import pow, sqrt
from scipy.integrate import ode

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


class DimensionlessITM:
    def __init__(self, parameters) -> None:
        self.M = parameters[0]
        self.h = parameters[1]
        self.omega0_b = parameters[2]
        self.omega0_cdm = parameters[3]
        self.w0 = parameters[4]
        self.beta = parameters[5]
        self.phi0 = parameters[6]

        # H0 = 100.0 * h
        # self.Omega0_g = omega0_g / h**2
        self.Omega0_g = 0.0
        self.Omega0_b = self.omega0_b / self.h**2
        self.Omega0_cdm = self.omega0_cdm / self.h**2
        self.Omega0_scf = 1.0 - self.Omega0_g - self.Omega0_b - self.Omega0_cdm

        self.n = 2.0

    def dimensionless_hubble_from_energy_density(self, z, phi, dphi, rho_cdm):
        """E = H/H0"""

        rho_g = self.Omega0_g * np.power(1.0 + z, 4.0)
        rho_b = self.Omega0_b * np.power(1.0 + z, 3.0)
        rho_scf = self.dimensionless_rho_scf(phi, dphi)

        return np.sqrt(rho_g + rho_b + rho_cdm + rho_scf)

    def dimensionless_rho_scf(self, phi, dphi):
        """phi is the dimensionless version!"""
        potential = self.dimensionless_scf_potential(phi, dphi)
        return potential / np.sqrt(1.0 - dphi**2)

    def dimensionless_scf_potential(self, phi, dphi):
        v0 = self.Omega0_scf * np.sqrt(1.0 - dphi**2)
        return v0 * np.power(self.phi0 / phi, self.n)

    def dln_scf_potential(self, phi):
        return -self.n / phi

    def dimensionless_coupling(self, z, phi, dphi, rho_cdm):
        hubble = self.dimensionless_hubble_from_energy_density(z, phi, dphi, rho_cdm)
        return self.beta * dphi * rho_cdm / hubble

    def get_ode(self, z, x):
        phi = x[0]
        dphi = x[1]
        rho_cdm = x[2]

        hubble = self.dimensionless_hubble_from_energy_density(z, phi, dphi, rho_cdm)
        pot = self.dimensionless_scf_potential(phi, dphi)
        dpot = self.dln_scf_potential(phi)
        coup = self.dimensionless_coupling(z, phi, dphi, rho_cdm)

        # if phi_dot**2 > 1.0:
        #     return

        dphi_dz = -dphi / (1.0 + z) / hubble

        ddphi_dz = 3.0 * dphi
        ddphi_dz += dpot / hubble
        ddphi_dz -= np.sqrt(1.0 - dphi**2) * coup / dphi / pot
        ddphi_dz *= (1.0 - dphi**2) / (1.0 + z)

        drho_cdm_dz = (3.0 * rho_cdm - coup) / (1.0 + z)

        return [dphi_dz, ddphi_dz, drho_cdm_dz]

    def solve(self, z_max):
        # initial conditions (move to constructor?)
        z_ini = 0.0
        phi_ini = self.phi0
        phi_dot_ini = sqrt(1 + self.w0)
        rho_cdm_ini = self.Omega0_cdm

        ic = phi_ini, phi_dot_ini, rho_cdm_ini

        # ode solver
        backend = "vode"
        # backend = "dopri5"
        # solver = ode(self.get_ode).set_integrator(backend, nsteps=1)
        solver = ode(self.get_ode).set_integrator(backend)
        solver.set_initial_value(ic, z_ini).set_f_params()
        sol = []
        flag = 0
        while solver.t < z_max and flag == 0:
            solver.integrate(z_max, step=True)
            # if solver.y[1] <= -1.:
            #     flag = 1
            # else:
            sol.append([solver.t, solver.y[0], solver.y[1], solver.y[2]])

        sol = np.array(sol)
        z = sol[:, 0]
        phi = sol[:, 1]
        dphi = sol[:, 2]
        rho_cdm = sol[:, 3]

        self.result = {"z": z, "phi": phi, "dphi": dphi, "rho_cdm": rho_cdm}

        return self.result

    def plot_solution(self):
        plt.plot(self.result["z"], self.result["phi"], label="phi")
        plt.plot(self.result["z"], self.result["dphi"], label="dphi")
        plt.plot(self.result["z"], self.result["rho_cdm"], label="rho_cdm")
        plt.xlabel("$z$")
        # pl.ylabel("$H(z)$ $[Mpc^{-2}]$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.show()

    def hubble_interpolation_table():
        pass


def test_rho_cdm_uncoupled():
    M = 25.0
    omega0_b = 0.02262
    omega0_cdm = 0.12
    h = 6.9213000e-01
    phi0 = 0.05
    w0 = -0.99
    beta = 0.0

    z_max = 0.5
    params = [M, h, 0.0, omega0_b + omega0_cdm, w0, beta, phi0]
    itm = DimensionlessITM(params)
    solution = itm.solve(z_max=z_max)

    z_sol = solution["z"]
    rho_cdm_sol = solution["rho_cdm"]

    # analytical solution for uncoupled model
    rho_cdm_analytic = itm.Omega0_cdm * (1.0 + z_sol) ** 3

    error = (rho_cdm_sol - rho_cdm_analytic) ** 2
    error = error.mean()

    print("Error: ", error)

    assert error < 0.0001


def main():
    M = 25.0
    omega0_b = 0.02262
    omega0_cdm = 0.12
    h = 6.9213000e-01
    phi0 = 0.05
    w0 = -0.99
    beta = 0.0

    params = [M, h, omega0_b, omega0_cdm, w0, beta, phi0]
    z_max = 1.0

    itm = DimensionlessITM(params)
    itm.solve(z_max=z_max)
    itm.plot_solution()

    test_rho_cdm_uncoupled()


if __name__ == "__main__":
    main()
