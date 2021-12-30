# DP-Means++

### Authors: Rylan Schaeffer, Ila Rani Fiete

-----

## Setup

After cloning the repository, create a virtual environment for Python 3:

`python3 -m venv dpmeanspp_venv`

Then activate the virtual environment:

`source dpmeanspp_venv/bin/activate`

Ensure pip is up to date:

`pip install --upgrade pip`

Then install the required packages:

`pip install -r requirements.txt`

We did not test Python2, but Python2 may work.


## Running

Each experiment has its own directory, each containing a `main.py` that creates a `plots`
subdirectory (e.g. `exp_00_ibp_prior/plots`) and then reproduces the plots in the paper. Each 
`main.py` should be run from the repository directory e.g.:

`python3 exp_00_ibp_prior/main.py`

## TODO

- MNIST or Omniglot
  - Implement MNIST comparison of classes based on inferred features
  - PCA MNIST
  - Implement Omniglot
- Landmark Dataset
  - linear gaussian -> NMF
- Derive
  - Factor Analysis
  - Non-negative Matrix Factorization
- Implement
  - Factor Analysis
  - Non-negative Matrix Factorization
- Implement IBP baselines
  - One of Paisley & Carin's papers
  - Particle Filtering (Wood and Griffiths)
- Refactor code into python package

## Contact

Questions? Comments? Interested in collaborating? Open an issue or 
email Rylan Schaeffer (rylanschaeffer@gmail.com) and cc Yilun Du
(yilundu@gmail.com) and Ila Fiete (fiete@mit.edu).
