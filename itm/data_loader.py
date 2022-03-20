import numpy as np


class DataLoader:
    def __init__(self, experiments):
        if 'local_hubble' in experiments:
            print("Loading local_hubble data")

            y, y_err = np.loadtxt("data/H0.txt", comments='#', unpack=True)
            self._local_hubble = {'y': y,
                                  'y_err': y_err}

        if 'cosmic_chronometers' in experiments:
            print("Loading cosmic_chronometers data")

            x, y, y_err = np.loadtxt("data/hubble.txt", unpack=True)
            self._cosmic_chronometers = {'x': x,
                                         'y': y,
                                         'y_err': y_err}

        if 'jla' in experiments:
            print("Loading jla data")

            x, y = np.loadtxt("data/jla_mub.txt", comments='#', unpack=True)
            flatten_cov = np.loadtxt("data/jla_mub_covmatrix.dat")
            self._jla = {'x': x,
                         'y': y,
                         'cov': flatten_cov}

        if 'bao_compilation' in experiments:
            print("Loading bao_compilation data")
            x, y, y_err = np.loadtxt("data/bao.txt", comments='#', unpack=True)
            self._bao_compilation = {'x': x,
                                     'y': y,
                                     'y_err': y_err}

        if 'bao_wigglez' in experiments:
            print("Loading bao_wigglez data")
            x, y, y_err = np.loadtxt(
                "data/wigglez.dat", comments='#', unpack=True)
            flatten_inv_cov = np.loadtxt("data/wigglez_invcovmat.dat")
            self._bao_wigglez = {'x': x,
                                 'y': y,
                                 'inv_cov': flatten_inv_cov}
