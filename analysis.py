import argparse
import os

import yaml

import pede.cosmology
from pede.estimator import Estimator


# parse command line arguments
def parse_args():
    parse = argparse.ArgumentParser()

    # add argument for results directory. required.
    parse.add_argument(
        "-r",
        "--results_dir",
        type=str,
        required=True,
        help="directory where the results are stored.",
    )

    return vars(parse.parse_args())


def load_yaml_params(results_dir):
    params_path = os.path.join(results_dir, "params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)
    return params


def main():
    args = parse_args()
    params = load_yaml_params(args["results_dir"])

    if params["model"] == pede.cosmology.ModelNames.LCDM:
        cosmo = pede.cosmology.LCDM()
    elif params["model"] == pede.cosmology.ModelNames.WCDM:
        cosmo = pede.cosmology.IDE1()
    elif params["model"] == pede.cosmology.ModelNames.IDE1:
        cosmo = pede.cosmology.IDE1()
    elif params["model"] == pede.cosmology.ModelNames.IDE2:
        cosmo = pede.cosmology.IDE2()
    # elif params["model"] == pede.cosmology.ModelNames.ITM:
    #     cosmo = pede.cosmology.ITM()
    else:
        raise ValueError(f"Invalid model: {params['model']}.")

    experiments = [
        "local_hubble",
        "cosmic_chronometers",
        "jla",
        "bao_compilation",
        "bao_wigglez",
    ]

    estimator = Estimator(cosmo, experiments)
    estimator.load_chains(args["results_dir"])
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
