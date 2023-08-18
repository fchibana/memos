import argparse

import pede.cosmology
from pede.estimator import Estimator


# parse command line arguments
def parse_args():
    parse = argparse.ArgumentParser()

    # add argument for model. possible values: LCDM, WCDM, IDE1, IDE2, ITM.
    # default is LCDM.
    parse.add_argument(
        "-m",
        "--model",
        type=str,
        default="LCDM",
        help="model to be used for the estimation",
    )


def main():
    # TODO: add command line arguments
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
    estimator.run(nwalkers=16, max_iter=50000)


if __name__ == "__main__":
    main()
