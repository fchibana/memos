import itm.cosmology
from itm.estimator import Estimator
import itm.utils


def main():
    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]
    # cosmo = itm.cosmology.LCDM()
    # cosmo = itm.cosmology.WCDM()
    # cosmo = itm.cosmology.IDE1()
    # cosmo = itm.cosmology.IDE2()
    cosmo = itm.cosmology.ITM()

    estimator = Estimator(model=cosmo, experiments=experiments)
    estimator.run(nwalkers=16)


if __name__ == "__main__":
    main()
