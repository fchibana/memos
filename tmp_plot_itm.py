import matplotlib.pyplot as plt
import numpy as np

from pede.itm_solver import ITMSolver


def relative_rms_error(x_expected: np.ndarray, x_predicted: np.ndarray):
    """Root mean square error between two vectors"""

    msg = f"got arrays of different shape {x_expected.shape} and {x_predicted.shape}"
    assert x_expected.shape == x_predicted.shape, msg

    relative_error = (x_expected - x_predicted) / x_expected
    mse = (np.square(relative_error)).mean()
    return np.sqrt(mse)


def test_rho_cdm_coupled():
    params = [
        25.0,  # M
        6.9213000e-01,  # h
        0.02262,  # omega0_b
        0.12,  # omega0_cdm
        -0.99,  # w0
        -0.1,  # beta
        0.05,  # phi0
    ]

    itm = ITMSolver(params)
    solution = itm.solve(z_max=5.0)

    z = solution["z"]
    phi = solution["phi"]
    rho_cdm_numerical = solution["rho_cdm"]
    rho_cdm_analytical = itm.rho_cdm_analytical(z, phi)

    plt.plot(z, rho_cdm_analytical)
    plt.plot(z, rho_cdm_numerical)
    # plt.xscale("log")
    plt.yscale("log")
    plt.show()

    diff = rho_cdm_analytical - rho_cdm_numerical
    diff = diff / rho_cdm_analytical
    plt.plot(z, diff)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.show()


def main():
    # params = [
    #     25.0,  # M
    #     6.9213000e-01,  # h
    #     0.02262,  # omega0_b
    #     0.12,  # omega0_cdm
    #     -0.99,  # w0
    #     0.0,  # beta
    #     0.05,  # phi0
    # ]

    # itm = ITMSolver(params)
    # solution = itm.solve(z_max=5.0)

    # z = solution["z"]
    # rho_cdm_numerical = solution["rho_cdm"]
    # rho_cdm_analytical = itm.rho_cdm_analytical(z)
    # rmse = rms_error(rho_cdm_numerical, rho_cdm_analytical)

    # plt.plot(z, rho_cdm_analytical)
    # plt.plot(z, rho_cdm_numerical)
    # # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()

    # diff = rho_cdm_analytical - rho_cdm_numerical
    # diff = diff / rho_cdm_analytical
    # plt.plot(z, diff)
    # plt.xscale("log")
    # # plt.yscale("log")
    # plt.show()

    # # error_threshold = 1e-5
    # # msg = f"Error too large {rmse} (thresh: {error_threshold})"
    # # assert rmse < error_threshold, msg

    test_rho_cdm_coupled()


if __name__ == "__main__":
    main()
