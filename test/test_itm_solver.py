from itm.itm_solver import ITMSolver


def main():
    # "M": parameters[0],
    # "h": parameters[1],
    # "omega0_b": parameters[2],
    # "omega0_cdm": parameters[3],
    # "w0": parameters[4],
    # "beta": parameters[5],
    # "phi0": parameters[6],

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

    z_max = 10.0
    solution = itm.solve(z_max)

    itm.plot_solution(solution)


if __name__ == "__main__":
    main()