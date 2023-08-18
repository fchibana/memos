import pede.cosmology
from pede.estimator import Estimator


def main():
    fname = "results/lcdm_20230817_133704"
    cosmo = pede.cosmology.LCDM()
    # cosmo = pede.cosmology.IDE1()
    # cosmo = pede.cosmology.IDE2()
    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]

    estimator = Estimator(cosmo, experiments)
    estimator.load_chains(fname)
    # estimator.get_samples()

    bf = estimator.get_best_fit(save=True)
    print("\nBest-fit results:")
    print(bf)

    info_crit = estimator.information_criterion(save=True)
    print("\nInformation criteria results:")
    print(info_crit)

    estimator.plot(save=True)

    print("\nDONE!\n")


if __name__ == "__main__":
    main()
