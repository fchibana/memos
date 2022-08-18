from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode

import itm.constants


class ITMSolver:
    def __init__(self, parameters: list) -> None:

        # M = parameters[0]
        self.h = parameters[1]
        self.omega0_b = parameters[2]
        self.omega0_cdm = parameters[3]
        self.w0 = parameters[4]
        self.beta = parameters[5]
        self.phi0 = parameters[6]

        self.H0 = 100.0 * self.h
        self.omega0_g = itm.constants.get_omega0_g()
        self.Omega0_g = itm.constants.radiation_density(self.h)
        self.Omega0_b = self.omega0_b / self.h**2
        self.Omega0_cdm = self.omega0_cdm / self.h**2

        # TODO: what to do with this?
        omega0_fld = self.omega0_g + self.omega0_b + self.omega0_cdm
        self.A = 100.0 * self.phi0 * sqrt((self.h**2 - omega0_fld) * sqrt(-self.w0))

        self._solution = None

    def solve(self, z_max):
        # initial conditions
        z_ini = 0.0
        phi_ini = self.phi0
        dphi_ini = sqrt(1 + self.w0)
        rho_cdm_ini = self.Omega0_cdm * self.H0**2
        init = phi_ini, dphi_ini, rho_cdm_ini

        # ode solver
        backend = "vode"
        # backend = "dopri5"
        # solver = ode(g).set_integrator(backend, nsteps=1)
        solver = ode(self._get_ode).set_integrator(backend)
        solver.set_initial_value(init, z_ini).set_f_params()

        sol = []
        while solver.t < z_max:
            solver.integrate(z_max, step=True)
            # if solver.y[1] <= -1.:
            # 	flag = 1
            # else:
            sol.append([solver.t, solver.y[0], solver.y[1], solver.y[2]])
        sol = np.array(sol)
        z = sol[:, 0]
        phi = sol[:, 1]
        phi_dot = sol[:, 2]
        rho_cdm = sol[:, 3]
        self._solution = {"z": z, "phi": phi, "dphi": phi_dot, "rho_cdm": rho_cdm}
        return self._solution

    def rho_cdm_analytical(self, z, phi):
        rho_bare = self.Omega0_cdm * self.H0**2 * np.power(1 + z, 3.0)
        coupling = np.exp(self.beta * self.H0 * (phi - self.phi0))
        return rho_bare * coupling

    def get_rho_cdm_at_z(self, z):
        return np.interp(z, self._solution["z"], self._solution["rho_cdm"])

    def get_rho_cdm_analytical_at_z(self, z):
        rho_cdm = self.rho_cdm_analytical(self._solution["z"], self._solution["phi"])
        return np.interp(z, self._solution["z"], rho_cdm)

    def get_rho_scf_at_z(self, z):
        rho_scf = self._get_scf_energy_density(
            self._solution["z"],
            self._solution["phi"],
            self._solution["dphi"])
        return np.interp(z, self._solution["z"], rho_scf)

    def plot_solution(self, result):
        plt.plot(result["z"], result["phi"], label="phi")
        plt.plot(result["z"], result["dphi"], label="dphi")
        # plt.plot(result["z"], result["rho_cdm"], label="rho_cdm")
        plt.xlabel("$z$")
        # pl.ylabel("$H(z)$ $[Mpc^{-2}]$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.show()

    #
    # private
    #

    def _get_ode(self, t, x):
        phi = x[0]
        phi_dot = x[1]
        rho_cdm = x[2]

        Hubble = self._get_hubble(t, phi, phi_dot, rho_cdm)
        difflnV = self._get_scf_dln_potential(t, phi, phi_dot)
        U = self._get_scf_potential(t, phi, phi_dot)
        coupling = 3.0 * self.beta * self.H0 * rho_cdm * phi_dot

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

    def _get_hubble(self, z, phi, phi_dot, rho_cdm):

        rho_tot = 0

        # radiation:
        # rho_tot += self.Omega0_g * self.H0**2 * np.power(1 + z, 4.0)
        rho_tot += self.Omega0_g * self.H0**2 * (1.0 + z)**4.0

        # baryons:
        # rho_tot += self.Omega0_b * self.H0**2 * np.power(1 + z, 3.0)
        rho_tot += self.Omega0_b * self.H0**2 * (1.0 + z)**3.0

        # cdm:
        rho_tot += rho_cdm

        # scf:
        rho_tot += self._get_scf_energy_density(z, phi, phi_dot)

        # return np.sqrt(rho_tot)
        return sqrt(rho_tot)

    def _get_scf_potential(self, z, phi, phi_dot):
        # omega0_fld = self.omega0_g + self.omega0_b + self.omega0_cdm
        # A = 100.0 * self.phi0 * sqrt((self.h**2 - omega0_fld) * sqrt(-self.w0))
        # return np.power(A / phi, 2)
        return (self.A / phi) ** 2

    def _get_scf_dln_potential(self, z, phi, phi_dot):
        return -2.0 / phi

    def _get_scf_energy_density(self, z, phi, dphi):
        if isinstance(z, np.ndarray):
            return self._get_scf_potential(z, phi, dphi) / np.sqrt(1.0 - dphi**2)    
        return self._get_scf_potential(z, phi, dphi) / sqrt(1.0 - dphi**2)
