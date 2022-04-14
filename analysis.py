import itm.cosmology
from itm.estimator import Estimator


def main():
    fname = "results/chains_210926.h5"
    cosmo = itm.cosmology.LCDM()
    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]

    estimator = Estimator(cosmo, experiments)
    estimator.load_chains(fname)
    estimator.get_samples()
    estimator.plot()


if __name__ == "__main__":
    main()
