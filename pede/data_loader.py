import numpy as np


class DataLoader:
    def __init__(self, experiments):
        self._experiments = experiments
        self._n_data = 0

        if "local_hubble" in experiments:
            print("Loading local_hubble data")

            y, y_err = np.loadtxt("data/local_hubble.txt", comments="#", unpack=True)
            self._local_hubble = {"y": y, "y_err": y_err}
            self._n_data += 1

        if "cosmic_chronometers" in experiments:
            print("Loading cosmic_chronometers data")

            x, y, y_err = np.loadtxt("data/cosmic_chronometers.txt", unpack=True)
            self._cosmic_chronometers = {"x": x, "y": y, "y_err": y_err}
            self._n_data += y.shape[0]

        if "jla" in experiments:
            print("Loading jla data")

            x, y = np.loadtxt("data/jla_mub.txt", comments="#", unpack=True)
            flatten_cov = np.loadtxt("data/jla_mub_covmatrix.dat")

            cov = flatten_cov.reshape((x.shape[0], x.shape[0]))
            y_err = np.sqrt(cov.diagonal())

            self._jla = {"x": x, "y": y, "y_err": y_err, "cov": cov}
            self._n_data += y.shape[0]

        if "bao_compilation" in experiments:
            print("Loading bao_compilation data")

            x, y, y_err = np.loadtxt(
                "data/bao_compilation.txt", comments="#", unpack=True
            )
            self._bao_compilation = {"x": x, "y": y, "y_err": y_err}
            self._n_data += y.shape[0]

        if "bao_wigglez" in experiments:
            print("Loading bao_wigglez data")

            x, y, y_err = np.loadtxt("data/bao_wigglez.dat", comments="#", unpack=True)
            flatten_inv_cov = np.loadtxt("data/bao_wigglez_invcovmat.dat")

            inv_cov = flatten_inv_cov.reshape((x.shape[0], x.shape[0]))
            cov = np.linalg.inv(inv_cov)

            self._bao_wigglez = {"x": x, "y": y, "y_err": y_err, "cov": cov}
            self._n_data += y.shape[0]

    def get_n_data(self):
        return self._n_data

    def get_local_hubble(self) -> dict:
        dataset = "local_hubble"
        assert self._is_loaded(dataset), (
            dataset + " data not loaded. Update experiments list."
        )
        return self._local_hubble

    def get_cosmic_chronometers(self) -> dict:
        dataset = "cosmic_chronometers"
        assert self._is_loaded(dataset), (
            dataset + " data not loaded. Update experiments list."
        )
        return self._cosmic_chronometers

    def get_jla(self) -> dict:
        dataset = "jla"
        assert self._is_loaded(dataset), (
            dataset + " data not loaded. Update experiments list."
        )
        return self._jla

    def get_bao_compilation(self) -> dict:
        dataset = "bao_compilation"
        assert self._is_loaded(dataset), (
            dataset + " data not loaded. Update experiments list."
        )
        return self._bao_compilation

    def get_bao_wigglez(self) -> dict:
        dataset = "bao_wigglez"
        assert self._is_loaded(dataset), (
            dataset + " data not loaded. Update experiments list."
        )
        return self._bao_wigglez

    def _is_loaded(self, dataset: str):
        return dataset in self._experiments
