# Poisson matrix factorization with Poisson-randomized gamma priors

This implements MCMC for Poisson matrix factorization (PMF) with independent Poisson-randomized gamma (PRG) priors.

This model is based on the one presented by Stein-O’Brien et al. (2019) [1] for factorizing gene-cell matrices of RNA sequence data. The code is specifically written for that application.

The MCMC algorithm is based on the one presented by Schein et al. (2019) [2] and the code is adapted from the open-source [code for that paper](https://github.com/aschein/prgds).

[1] Stein-O’Brien, Genevieve L., et al. "Decomposing cell identity for transfer learning across cellular measurements, platforms, tissues, and species." Cell systems 8.5 (2019): 395-411. https://www.cell.com/cell-systems/pdfExtended/S2405-4712(19)30146-2

[2] Schein, Aaron, et al. "Poisson-Randomized Gamma Dynamical Systems." Advances in Neural Information Processing Systems. 2019. http://papers.nips.cc/paper/8366-poisson-randomized-gamma-dynamical-systems

MIT License

Copyright (c) 2020 Aaron Schein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What's included in src:

* [apf.pyx](src/apf/base/apf.pyx): Allocation-based Poisson factorization (APF). This is the base class for Poisson tensor decomposition models with non-negative priors.
* [bessel.pyx](src/apf/base/bessel.pyx): Sampling algorithms for the Bessel distribution.
* [sbch.pyx](src/apf/base/sbch.pyx): Sampling algorithms for the size-biased confluent hypergeometric (SCH) distribution.
* [prgpmf.pyx](src/apf/models/prgpmf.pyx): Poisson matrix factorization (PMF) with Poisson-randomized gamma (PRG) priors.

## Dependencies:
* [cython](https://cython.org/)
* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [pandas](https://pandas.pydata.org/)
* [path](https://anaconda.org/anaconda/path.py)
* [scikit-learn](https://scikit-learn.org/stable/)
* [tensorly](http://tensorly.org/stable/index.html)

## Compiling and running in Docker
All of the code is provided within a Docker container. To build the code, cd into the home directory and run:
```
docker build -t prgpmf .
```
There is an example script provided on how to call the model from Python [example_prgpmf.py](src/scripts/example_prgpmf.py) which can be run from the command line within the Docker container with:
```
docker run --rm  prgpmf python /work/src/scripts/example_prgpmf.py
```
