import argparse

import pede.cosmology
from pede.estimator import Estimator


# parse command line arguments
def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help=(
            "model to be used for the estimation. "
            "possible values: "
            + pede.cosmology.ModelNames.LCDM
            + ", "
            + pede.cosmology.ModelNames.WCDM
            + ", "
            + pede.cosmology.ModelNames.IDE1
            + ", "
            + pede.cosmology.ModelNames.IDE2
            + ". "
        ),
    )

    parse.add_argument(
        "-n",
        "--nwalkers",
        type=int,
        default=16,
        help="number of walkers to be used for the estimation. default is 16.",
    )

    parse.add_argument(
        "-i",
        "--max_iter",
        type=int,
        default=50000,
        help=(
            "maximum number of iterations to be used for the estimation. "
            "default is 50000."
        ),
    )

    return vars(parse.parse_args())


def main():
    args = parse_args()

    if args["model"] == pede.cosmology.ModelNames.LCDM:
        cosmo = pede.cosmology.LCDM()
    elif args["model"] == pede.cosmology.ModelNames.WCDM:
        cosmo = pede.cosmology.WCDM()
    elif args["model"] == pede.cosmology.ModelNames.IDE1:
        cosmo = pede.cosmology.IDE1()
    elif args["model"] == pede.cosmology.ModelNames.IDE2:
        cosmo = pede.cosmology.IDE2()
    # elif args["model"] == pede.cosmology.ModelNames.ITM:
    #     cosmo = pede.cosmology.ITM()
    else:
        raise ValueError(f"Invalid model: {args['model']}.")

    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]

    estimator = Estimator(model=cosmo, experiments=experiments)
    estimator.run(nwalkers=args["nwalkers"], max_iter=args["max_iter"])
    print("Done")


if __name__ == "__main__":
    main()
