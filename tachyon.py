import matplotlib.pyplot as plt
import numpy as np

from math import pow, pi, sqrt
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

# derivative of phi: dphi/dz = -sqrt(1+w)/[(1+z).H]
# derivative of w: dw/dz = -2.*sqrt(1.+w)w(3H.sqrt(1+w)+dlnV/dphi + sqrt( -w/(1+w) )f(Q)/V ) /[(1+z).H]


class ITM:
    def __init__(self) -> None:
        pass

    def g(self, t, x, params):
        M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

        H0 = 100.0 * h

        phi = x[0]
        phi_dot = x[1]
        rho_cdm = x[2]

        Hubble = self.H_local(t, phi, phi_dot, rho_cdm, params)
        difflnV = self.dlnV(t, phi, phi_dot, params)
        U = self.V(t, phi, phi_dot, params)
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

    def H_local(self, z, phi, phi_dot, rho_cdm, params):
        M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

        H0 = 100.0 * h
        Omega0_g = omega0_g / h**2
        Omega0_b = omega0_b / h**2
        # Omega0_cdm = omega0_cdm / h**2

        rho_tot = 0

        # radiation:
        rho_tot += Omega0_g * H0**2 * np.power(1 + z, 4.0)

        # baryons:
        rho_tot += Omega0_b * H0**2 * np.power(1 + z, 3.0)

        # cdm:
        rho_tot += rho_cdm

        # scf:
        rho_tot += self. V(z, phi, phi_dot, params) / np.sqrt(1 - phi_dot**2)

        return np.sqrt(rho_tot)

    def V(self, z, phi, phi_dot, params):
        M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

        omega0_fld = omega0_g + omega0_b + omega0_cdm
        A = 100.0 * phi0 * sqrt((h**2 - omega0_fld) * sqrt(-w0))
        return np.power(A / phi, 2)

    def dlnV(self, z, phi, phi_dot, params):
        return -2.0 / phi

    def H(self, z_max, params):
        """Computes H(z) up to a given redshift z_max. This is used to interpolate H(z) in the future."""
        M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

        H0 = 100.0 * h
        # Omega0_g = omega0_g / h**2
        # Omega0_b = omega0_b / h**2
        Omega0_cdm = omega0_cdm / h**2

        z_ini = 0.0
        phi_ini = phi0
        phi_dot_ini = sqrt(1 + w0)
        rho_cdm_ini = Omega0_cdm * H0**2

        init = phi_ini, phi_dot_ini, rho_cdm_ini

        # ode solver
        # backend = 'vode'
        backend = "dopri5"
        solver = ode(g).set_integrator(backend, nsteps=1)
        solver.set_initial_value(init, z_ini).set_f_params(params)
        sol = []
        flag = 0
        while solver.t < z_max and flag == 0:
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

        return flag, z, self.H_local(z, phi, phi_dot, rho_cdm, params)

    def solve(self, z_max, params):
        M, omega0_b, omega0_cdm, phi0, h, w0, beta = params

        H0 = 100.0 * h
        # Omega0_g = omega0_g / h**2
        # Omega0_b = omega0_b / h**2
        Omega0_cdm = omega0_cdm / h**2

        z_ini = 0.0
        phi_ini = phi0
        phi_dot_ini = sqrt(1 + w0)
        rho_cdm_ini = Omega0_cdm * H0**2

        init = phi_ini, phi_dot_ini, rho_cdm_ini

        # ode solver
        backend = 'vode'
        # backend = "dopri5"
        # solver = ode(g).set_integrator(backend, nsteps=1)
        solver = ode(self.g).set_integrator(backend)
        solver.set_initial_value(init, z_ini).set_f_params(params)
        sol = []
        flag = 0
        while solver.t < z_max and flag == 0:
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
        result = {"z": z, "phi": phi, "dphi": phi_dot, "rho_cdm": rho_cdm}
        return result

    def plot_solution(self, result):
        plt.plot(result["z"], result["phi"], label="phi")
        plt.plot(result["z"], result["dphi"], label="dphi")
        # plt.plot(result["z"], result["rho_cdm"], label="rho_cdm")
        plt.xlabel("$z$")
        # pl.ylabel("$H(z)$ $[Mpc^{-2}]$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.show()


def main():

    M = 25.0
    omega0_b = 0.02262
    omega0_cdm = 0.12
    h = 6.9213000e-01
    phi0 = 0.05
    w0 = -0.99
    beta = 0.0

    params = [M, omega0_b, omega0_cdm, phi0, h, w0, beta]
    z_max = 10.0

    itm = ITM()

    solution = itm.solve(z_max, params)

    itm.plot_solution(solution)


if __name__ == "__main__":
    main()
