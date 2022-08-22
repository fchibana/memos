import pede.cosmology
from pede.estimator import Estimator
import pede.utils


def main():
    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]
    cosmo = pede.cosmology.LCDM()
    # cosmo = pede.cosmology.WCDM()
    # cosmo = pede.cosmology.IDE1()
    # cosmo = pede.cosmology.IDE2()
    # cosmo = pede.cosmology.ITM()

    estimator = Estimator(model=cosmo, experiments=experiments)
    estimator.run(nwalkers=16)


if __name__ == "__main__":
    main()
