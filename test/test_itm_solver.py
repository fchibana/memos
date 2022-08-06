import numpy as np

from itm.itm_solver import ITMSolver


def rms_error(x: np.ndarray, y: np.ndarray):
    """Root mean square error between two vectors"""

    msg = f"got arrays of different shape {x.shape} and {y.shape}"
    assert x.shape == y.shape, msg

    mse = (np.square(x - y)).mean()
    return np.sqrt(mse)


def relative_rms_error(x_expected: np.ndarray, x_predicted: np.ndarray):
    """Root mean square RELATIVE error between two vectors"""

    msg = f"got arrays of different shape {x_expected.shape} and {x_predicted.shape}"
    assert x_expected.shape == x_predicted.shape, msg

    relative_error = (x_expected - x_predicted) / x_expected
    mse = (np.square(relative_error)).mean()
    return np.sqrt(mse)


class TestITMSolver:

    def test_solution_dictionary_has_keys(self):
        params = [
            25.0,  # M
            6.9213000e-01,  # h
            0.02262,  # omega0_b
            0.12,  # omega0_cdm
            -0.99,  # w0
            0.0,  # beta
            0.05,  # phi0
        ]
        itm = ITMSolver(params)
        solution = itm.solve(z_max=1.0)

        test_keys = ["z", "phi", "dphi", "rho_cdm"]

        for k in test_keys:
            assert k in solution

    def test_rho_cdm_uncoupled(self):
        params = [
            25.0,  # M
            6.9213000e-01,  # h
            0.02262,  # omega0_b
            0.12,  # omega0_cdm
            -0.99,  # w0
            0.0,  # beta
            0.05,  # phi0
        ]

        itm = ITMSolver(params)
        solution = itm.solve(z_max=5.0)

        z = solution["z"]
        phi = solution["phi"]
        rho_cdm_numerical = solution["rho_cdm"]
        rho_cdm_analytical = itm.rho_cdm_analytical(z, phi)
        rmse = rms_error(rho_cdm_numerical, rho_cdm_analytical)
        rel_rmse = relative_rms_error(rho_cdm_analytical, rho_cdm_numerical)

        error_threshold = 1e-5
        # msg = f"Error too large {rmse} (thresh: {error_threshold})"
        assert rmse < 1.0
        assert rel_rmse < error_threshold

        print(f"rel_error {rel_rmse} for beta = {params[5]}")

    def test_rho_cdm_coupled(self):
        params = [
            25.0,  # M
            6.9213000e-01,  # h
            0.02262,  # omega0_b
            0.12,  # omega0_cdm
            -0.99,  # w0
            0.01,  # beta
            0.05,  # phi0
        ]

        itm = ITMSolver(params)
        solution = itm.solve(z_max=5.0)

        z = solution["z"]
        phi = solution["phi"]
        rho_cdm_numerical = solution["rho_cdm"]
        rho_cdm_analytical = itm.rho_cdm_analytical(z, phi)

        # rmse = rms_error(rho_cdm_numerical, rho_cdm_analytical)
        # assert rmse < 1.0

        rel_rmse = relative_rms_error(rho_cdm_analytical, rho_cdm_numerical)
        assert rel_rmse < 0.1  # 10% for now

        print(f"rel_error {rel_rmse} for beta = {params[5]}")
