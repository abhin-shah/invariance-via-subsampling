# Source code for "Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge"

Reference: Abhin Shah, Karthikeyan Shanmugam, Kartik Ahuja,
"Finding Valid Adjustments under Non-ignorability with Minimal DAG Knowledge," 
The 25th International Conference on Artificial Intelligence and Statistics (AISTATS), 2022

Contact: abhin@mit.edu

Arxiv: [https://arxiv.org/pdf/2106.11560.pdf](https://arxiv.org/pdf/2106.11560.pdf)

### Dependencies:

In order to successfully execute the code, the following libraries must be installed:

1. Python --- causallib, sklearn, multiprocessing, contextlib, scipy, functools, pandas, numpy, itertools, random, argparse, time, matplotlib, pickle, pyreadr, rpy2, torch

2. R --- RCIT

### Command inputs:

-   nr: number of repetitions (default = 100)
-   no: number of observations (default = 50000)
-   use_t_in_e: indicator for whether t should be used to generate e (default = 1)
-   ne: number of environments (default = 3)
-   number_IRM_iterations - number of iterations of IRM (default = 15000)
-   nrd - number of features for sparse subset search (default = 5)

### Reproducing the figures and tables:

1. To reproduce Figure 3a and Figure 10a, run the following three commands:
```shell
$ mkdir synthetic_theory
$ python3 -W ignore synthetic_theory.py --nr 100
$ python3 plot_synthetic_theory.py --nr 100
```
2. To reproduce Figure 3b and Figure 10b, run the following three commands:
```shell
$ mkdir synthetic_algorithms
$ python3 -W ignore synthetic_algorithms.py --nr 100
$ python3 plot_synthetic_algorithms.py --nr 100
```
3. To reproduce Figure 3c, run the following three commands:
```shell
$ mkdir synthetic_high_dimension
$ python3 -W ignore synthetic_high_dimension.py --nr 100
$ python3 plot_synthetic_high_dimension.py --nr 100
```
4. To reproduce Table 1, run the following two commands:
```shell
$ mkdir syn-entner 
$ python3 -W ignore syn-entner --nr 100
```
5. To reproduce Table 2, run the following two commands:
```shell
$ mkdir syn-cheng 
$ python3 -W ignore syn-cheng --nr 100
```
6. To reproduce Figure 4, Figure 12a and Figure 12b, run the following three commands:
```shell
$ mkdir ihdp
$ python3 -W ignore ihdp.py --nr 100
$ python3 plot_ihdp.py --nr 100
```
7. To reproduce Figure 5, run the following three commands:
```shell
$ mkdir cattaneo
$ python3 -W ignore cattaneo.py --nr 100
$ python3 plot_cattaneo.py --nr 100
```
8. To reproduce Figure 11a and Figure 11c, run the following three commands:
```shell
$ mkdir synthetic_theory
$ python3 -W ignore synthetic_theory.py --nr 100 --use_t_in_e 0
$ python3 plot_synthetic_theory.py --nr 100 --use_t_in_e 0
```
9. To reproduce Figure 11b and Figure 11d, run the following three commands:
```shell
$ mkdir synthetic_algorithms
$ python3 -W ignore synthetic_algorithms.py --nr 100 --use_t_in_e 0
$ python3 plot_ synthetic_algorithms.py --nr 100 --use_t_in_e 0
```
