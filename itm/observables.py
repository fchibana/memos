from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
import numpy as np

from itm import constants
from itm.data_loader import DataLoader
from itm.cosmology import Cosmology


class Observables:
    def __init__(self, cosmology: Cosmology):

        self._cosmology = cosmology
        # print("Constructing Observables")
        # self._m_jla = 24.96
        # self._h = 0.69
        # self._omega_b = 0.022
        # self._omega_cdm = 0.12

    # def update_cosmological_parameter(self, params):
    #   self._m_jla = params['m_jla']
    #   self._h = params['h']
    #   self._omega_b = params['omega_b']
    #   self._omega_cdm = params['omega_cdm']

    #   self._hubble0 = 100. * self._h

    def _inv_E(self, x, params):
        # M, h, omega0_b, omega0_cdm = params
        # M = params[0]
        h = params[1]
        # omega0_b = params[2]
        # omega0_cdm = params[3]

        H0 = 100.0 * h
        return H0 / self._cosmology.hubble(x, params)

    def _comoving_distance(self, x, params):
        assert x.ndim == 1, f"input must be vector. got: {x.ndim}"
        # M, h, omega0_b, omega0_cdm = params
        # M = params[0]
        h = params[1]
        # omega0_b = params[2]
        # omega0_cdm = params[3]

        H0 = 100.0 * h
        c = constants.C * pow(10.0, -3)

        dc_i = []
        for x_i in x:
            dc_i.append(quad(self._inv_E, 0, x_i, args=(params), limit=150)[0])

        # in Mpc
        return c * np.asarray(dc_i) / H0

    def _angular_diameter_distance(self, x, params):

        # in Mpc
        return self._comoving_distance(x, params) / (1.0 + x)

    def _luminosity_distance(self, x, params):
        # in Mpc
        return (1.0 + x) * self._comoving_distance(x, params)

    def distance_modulus(self, x, params):
        # M, h, omega0_b, omega0_cdm = params
        M = params[0]
        # h = params[1]
        # omega0_b = params[2]
        # omega0_cdm = params[3]

        return 5.0 * np.log10(self._luminosity_distance(x, params)) + M

    def _sound_horizon(self, params):
        """Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)

        Args:
            params (_type_): _description_
        """
        # M, h, omega0_b, omega0_cdm = params
        # M = params[0]
        h = params[1]
        omega0_b = params[2]
        omega0_cdm = params[3]

        Omega0_b = omega0_b / h**2
        Omega0_cdm = omega0_cdm / h**2

        omega0_m = (Omega0_b + Omega0_cdm) * h**2

        # TODO: where did this come from??
        # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
        r_d = (44.5 * math.log(9.83 / omega0_m)) / math.sqrt(
            1.0 + 10 * pow(omega0_b, (3.0 / 4.0))
        )

        return r_d

    def d_BAO(self, x, params):
        """BAO distance ratio

        Args:
            x (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Sound horizon at the drag epoch in Mpc (Eisenstein & Hu)
        rs = self._sound_horizon(params)

        c = constants.C * pow(10.0, -3)  # km/s
        hubble = self._cosmology.hubble(x, params)  # km/s/Mpc
        dc2 = self._comoving_distance(x, params) ** 2
        dv = np.power((c * x / hubble) * dc2, 1.0 / 3.0)  # dilation scale
        rs = self._sound_horizon(params)  # sound horizon

        # There's a c factor in D_H, so D_V is in Mpc and d_BAO has no units
        d_bao = dv / rs

        return d_bao

    def d_bao_wigglez(self, x, params):
        # Fiducial sound horizon at the drag epoch in Mpc
        # used by the WiggleZ (1401.0358)
        r_fid = 152.3

        d_bao = self.d_BAO(x, params)
        d_bao_wigglez = r_fid * d_bao

        return d_bao_wigglez


class Visualizer(Observables):
    def __init__(self, cosmology: Cosmology, experiments: list) -> None:
        super().__init__(cosmology)
        self._cosmology = cosmology
        self._experiments = experiments
        self._data = DataLoader(experiments)

    def show_local_hubble(self, parameters):
        # dataset = "local_hubble"
        data = self._data.get_local_hubble()
        model = self._cosmology.hubble(x=0, parameters=parameters)
        print("Local Hubble (H0)")
        print("  H0_fit: ", model)
        print("  H0_data(err): {} ({})".format(data["y"], data["y_err"]))
        print("  Diff: {}".format(model - data["y"]))

    def show_cosmic_chronometers(self, parameters):
        dataset = "cosmic_chronometers"
        data = self._data.get_cosmic_chronometers()
        model = self._cosmology.hubble(data["x"], parameters)

        plt.errorbar(
            data["x"], data["y"], yerr=data["y_err"], fmt=".k", label="data points"
        )
        plt.plot(data["x"], model, "-", label=self._cosmology.get_name())
        plt.xlabel("$z$")
        plt.ylabel("$H(z)$ $[Mpc^{-2}]$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.title(dataset)
        plt.show()

    def show_jla(self, parameters):
        dataset = "jla"
        data = self._data.get_jla()
        model = self.distance_modulus(data["x"], parameters)
        plt.errorbar(
            data["x"], data["y"], yerr=data["y_err"], fmt=".k", label="data points"
        )
        plt.plot(data["x"], model, "-", label=self._cosmology.get_name())
        plt.xscale("log")
        plt.xlabel("$z$")
        plt.ylabel(r"$\mu(z)$ $[Mpc^{-2}]$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.title(dataset)
        plt.show()

    def show_bao_compilation(self, parameters):
        dataset = "bao_compilation"
        data = self._data.get_bao_compilation()
        model = self.d_BAO(data["x"], parameters)
        plt.errorbar(
            data["x"], data["y"], yerr=data["y_err"], fmt=".k", label="data points"
        )
        plt.plot(data["x"], model, "-", label=self._cosmology.get_name())
        # plt.xscale("log")
        plt.xlabel("$z$")
        plt.ylabel(r"$D_V / r_s$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.title(dataset)
        plt.show()

    def show_bao_wigglez(self, parameters):
        dataset = "bao_wigglez"
        data = self._data.get_bao_wigglez()
        model = self.d_bao_wigglez(data["x"], parameters)
        plt.errorbar(
            data["x"], data["y"], yerr=data["y_err"], fmt=".k", label="data points"
        )
        plt.plot(data["x"], model, "-", label=self._cosmology.get_name())
        # plt.xscale("log")
        plt.xlabel("$z$")
        plt.ylabel(r"$D_V$")
        plt.legend(loc="upper left", prop={"size": 11})
        plt.grid(True)
        plt.title(dataset)
        plt.show()
