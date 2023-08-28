# PEDE: Parameter Estimation for Dark Energy

PEDE is a Python tool designed to facilitate the estimation of best-fit parameters for various dark energy models using observational data from diverse sources.

This package employs the maximum likelihood method and Markov Chain Monte Carlo (MCMC) techniques to achieve accurate parameter estimation.
Additionally, it offers a script for analyzing MCMC chains, extracting optimal fit values, and computing essential information criteria such as the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

The supported DE models are the standard $\Lambda$CDM, $w$CDM with an arbitrary equation of state constant, and two interacting dark energy models (IDE1, and IDE2).

## Dependencies

- [EMCEE](https://emcee.readthedocs.io/en/stable/) and [corner.py](https://corner.readthedocs.io/en/latest/index.html)
- NumPy, Matplotlib, Scipy, pandas, h5py, yaml, tqdm

Using venv:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install numpy matplotlib scipy pandas emcee corner h5py tqdm yaml
```

## Usage

To estimate the best-fit parameters for dark energy models using the maximum likelihood method and MCMC, you first need to get some samples and then analyse the chains.

### Drawing samples

```bash
python3 run.py -m <model>
```

Possible models are: `lcdm`, `wcdm`, `ide1`, and `ide2`.
Use the flag `-h` to inspect all possible arguments

```bash
$ python3 run.py -h
usage: run.py [-h] -m MODEL [-n NWALKERS] [-i MAX_ITER]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        model to be used for the estimation. possible values: lcdm, wcdm, ide1, ide2.
  -n NWALKERS, --nwalkers NWALKERS
                        number of walkers to be used for the estimation. default is 16.
  -i MAX_ITER, --max_iter MAX_ITER
                        maximum number of iterations to be used for the estimation. default is 50000.
```

Upon executing the `run.py` script, a new directory  `results/<model>_<timestamp>`  is created.
It contains two files, `params.yaml` with details about the sampling, and `chains.h5` with the actual samples.

### Analysing the chains

The package provides a convenient script for analysing MCMC chains and computing information criteria:

```bash
python3 analysis.py -r results/<model>_<timestamp>
```

This script computes and plots the best-fit values for the respective model (`best_fit.csv`, `plot.png`),
as well as the information criteria results (`info_crit.csv`).

## References

PEDE was originally developed during my master's course at the Institute of Physics, University of Sao Paulo (IFUSP).
Please check the thesis below for further details and references:

> Castro, F. C. (2017). [Tachyon Scalar Field Cosmology](https://teses.usp.br/teses/disponiveis/43/43134/tde-17052017-063702/en.php). Master's Dissertation, Instituto de Física, University of São Paulo, São Paulo. doi:10.11606/D.43.2017.tde-17052017-063702.

## License

This project is licensed under the [MIT License](LICENSE).
