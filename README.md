# PEDE: Parameter Estimation for Dark Energy

PEDE is a Python tool designed to facilitate the estimation of best-fit parameters for various dark energy models using [observational data from diverse sources](./data/README.md).
This package employs the maximum likelihood method and Markov Chain Monte Carlo (MCMC) techniques to achieve accurate parameter estimation.
Additionally, it offers a script for analyzing MCMC chains, extracting optimal fit values, and computing essential information criteria such as the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

Currently the supported DE models are: $\Lambda$CDM, $w$CDM, IDE1, and IDE2 TODO: add refs

## Installation/Dependencies

~~To install DarkEnergyEstimator, you can use pip:~~


- [EMCEE](https://emcee.readthedocs.io/en/stable/)
- [corner.py](https://corner.readthedocs.io/en/latest/index.html)
- NumPy, Matplotlib, Scipy
- pandas

conda create -p ./conda-env -c conda-forge emcee matplotlib numpy scipy pandas

h5py
yaml
tqdm: conda install -c conda-forge tqdm

`python3 -m pip install numpy matplotlib scipy pandas emcee corner h5py tqdm yaml`

TODO: add venv and conda instructions

## Usage

TODO: Check Usage section

### ~~Parameter Estimation~~ Running the chains

To estimate the best-fit parameters for dark energy models using the maximum likelihood method and MCMC, you can utilize the `estimate_parameters` function:

```python
from darkenergyestimator import estimate_parameters

# Load your observational data
data = load_observation_data()

# Specify the dark energy model (e.g., 'LCDM', 'wCDM', etc.)
model = 'LCDM'

# Perform parameter estimation
best_fit_params, chain = estimate_parameters(data, model)
```

### MCMC Chain Analysis

The package provides a convenient script for analyzing MCMC chains and computing information criteria:

```bash
darkenergy_mcmc_analysis --chain-file mcmc_chain.txt --output-file analysis_results.txt
```

- `chain-file`: Path to the MCMC chain file (format: plain text or CSV).
- `output-file`: Path to the output file where analysis results will be saved.


TODO: add example output

## References

PEDE was originally developed during my master's course at the Institute of Physics, University of Sao Paulo (IFUSP).
Please check the thesis below for further details and references

> Castro, F. C. (2017). [Tachyon Scalar Field Cosmology](https://teses.usp.br/teses/disponiveis/43/43134/tde-17052017-063702/en.php). Master's Dissertation, Instituto de Física, University of São Paulo, São Paulo. doi:10.11606/D.43.2017.tde-17052017-063702.


## License

This project is licensed under the [MIT License](LICENSE).

