# warehouse_allocation


![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
[![Actions Status](https://github.com/andresliszt/warehouse-allocation/workflows/Testing/badge.svg)](https://github.com/andresliszt/warehouse-allocation/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Solver for the warehouse allocation presented in the paper *Two-stage storage assignment to minimize travel time and congestion for warehouse order picking operations* by authors Gyu Lee and Sung Hoon Chung that can found [here](https://www.researchgate.net/publication/336713740_Two-stage_Storage_Assignment_to_Minimize_Travel_Time_and_Congestion_for_Warehouse_Order_Picking_Operations)

The definition and solution of the problem is implemented using [pymoo](https://pymoo.org/) in its version 0.5.0 and installation can be found in the [installation guide](https://pymoo.org/installation.html) that mostly of the time is direct. This project also uses [ortools](https://developers.google.com/optimization) in its backing for the definition of the crossover operator.


## Requirements

This project needs:

-   Python 3.8.x or Python 3.9.x
-   [pymoo](https://pymoo.org/) (Check own ``pymoo`` requirements)
-   [ortools](https://developers.google.com/optimization)
-   [Poetry](https://python-poetry.org/) (For installation)
-   [Tox](https://tox.wiki/en/latest/) (Optional)

## Installation from source

To install the package you must clone the repository, create a virtual environment to install the dependencies and finally install with poetry

```sh
    git clone 
    cd warehouse-allocation
    python -m venv .venv
    poetry install
```

Or simply *no dev* dependencies

```sh
    poetry install --no-dev
```


## Mathematical Formulation

Let $Q$ de total number of skus and $C$ the total number of clusters. $D = (D_i)$ the *demand vector*, where the $i$ entry is the demand of the sku $i$ in the period. Let $Z = (Z_k)$ the *clusters capacities vector*, where the $k$ entry is the amount of skus that the cluster $k$ can allocate. Let $N = (N_{i, j})$ the *occurrence matrix*, where $N_{i,j}$ is the is the number of times that sku $i$ and sku $j$ appeared in the same order. We define the binary variables $x_{k, i}$ that is 1 if the sku $i$ is allocated in the cluster $k$. We define our functions to be optimizated


$$\begin{eqnarray}
    f_1  & = & \sum_{i = 1}^{Q-1}\sum_{j = i + 1}^{Q}\sum_{k = 1}^{C}N_{i,j}x_{k,i}x_{k,j} \label{eq1}\tag{1}\\
    f_2 & = &  W_{\text{max}}\label{eq2}\tag{2}\\
   \end{eqnarray}$$


Note that the function (2) apparently not doing anything, but it will have a regularizing role when we present the constraints of the problem. Note also that  (1) only has real dependencie on $x = (x_{k,i})$ and (2) on $W_{\text{max}}$. Our optimization problem is


$$\min{-f_1, f_2}\label{eq3}\tag{3}$$

with the constraints


$$\begin{eqnarray}
  \sum_{k = 1}^{C} x_{k,i} & = & 1 \hspace{0.5cm} \forall i = 1, ..., Q\label{eq4}\tag{4}\\
  \sum_{i = 1}^{Q} x_{k,i} & \leq & Z_{k}  \hspace{0.5cm} \forall k = 1, ..., C\label{eq5}\tag{5}\\
  \sum_{i = 1}^{Q} D_ix_{k,i} & \leq & W_{\text{max}}  \hspace{0.5cm} \forall k = 1, ..., C\label{eq6}\tag{6}\\
  x_{k,i} & \in & \lbrace 0,1 \rbrace \label{eq7}\tag{7}\\
\end{eqnarray}$$

Maximization of the funcion (1) tries to put in the same cluster the skus that appear constantly together in orders in the same cluster. The function (2) is linked with the constraint (6) trying to dissipate the demand between the clusters. The constraint (4) imposes that each sku will be allocated in only one cluster and (6) is the limit of skus allowed in a cluster. 


There are new constraints added in this work and the documentation will be able soon.

## Usage

This project **does not receive** required environment variables. However, there is a default variable setting
related to the control of the logs. All of these variables can be loaded into an environment variable file, called `.env` which must be in the root of the project. For example, if we want to save all loggers from an `info` level onwards to a file, we must **simultaneously** set `WAREHOUSE_ALLOCATION_LOG_LEVEL=20`, `WAREHOUSE_ALLOCATION_LOG_DESTINATION=FILE` and `WAREHOUSE_ALLOCATION_LOG_FILE_PATH=MyPathToLogFile`, where `MyPathToLogFile` is the `path` where the loggers will be stored.

El nivel de logger está dado por la clase

```python
# warehouse_allocation.settings

class LogLevel(IntEnum):
"""Logs levels."""
    CRITICAL  = logging.CRITICAL
    ERROR  = logging.ERROR
    WARNING  = logging.WARNING
    INFO  = logging.INFO
    DEBUG  = logging.DEBUG
    TRACE  =  1  + logging.NOTSET
    NOTSET  = logging.NOTSET
```

For example, if `WAREHOUSE_ALLOCATION_LOG_LEVEL=30`, then the loggers will be printed from *warning* up. Full values ​​are `logging.CRITICAL=50`, `logging.ERROR=40`, `logging.WARNING=30`, `logging.INFO=20`, `logging.DEBUG=10`, `logging.NOTSET=0 `.

The way to write the environment variables in the `.env` configuration file is in the form `key=value` with no spaces between the two values.

## Solvers

The correct use of solvers depends on the understanding of the *warehouse allocation problem*. The problem is solved using genetic algorithms for multi-objective optimization problems, specifically the algorithm [NSGAII](https://pymoo.org/algorithms/moo/nsga2.html) and derivatives of it such as [NSGAIII](https://pymoo.org/algorithms/moo/nsga3.html).

#### Original paper problem

```python
"""Small example with 3 skus, and 2 clusters"""

import numpy as np

from warehouse_allocation.algorithms import solve_chung_problem

D = np.array([10, 10, 10])  # Demand for the Skus
Z = np.array([2, 2])  # Clusters capacities
OCM = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 0]])  # Occurrence Matrix

if __name__ == "__main__":
    res = solve_chung_problem(
        algorithm_name="NSGA2",
        constraints=False,
        D=D,
        Z=Z,
        OCM=OCM,
        pop_size=5,
        n_offsprings=5,
        iterations = 20,
        verbose = True,
    )

```

Note that the problem defined above has only one Pareto optimal solution, which is to put sku 1 next to sku 2,
This means that when requesting `n_offsprings=2`, the desired quantity is not fulfilled, since the search for descendants is done
to improve the current population, and if the current population already set the optimal individual that was mentioned, there is nothing to do.
What ``pymoo`` does is raise a warning saying that the mating cannot form the desired number of descendants and sometimes cuts off the mating.
execution when it detects that the iterations no longer make sense. This happens because the example is a minimal example.


#### Problem imposing constraints on clusters

```python
"""Small example with 3 skus, and 2 clusters"""

import numpy as np

from warehouse_allocation.algorithms import solve_chung_problem

D = np.array([10, 10, 10])  # Demand for the Skus
Z = np.array([2, 2])  # Clusters capacities
OCM = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 0]])  # Occurrence Matrix
W = np.array([10, 2, 4]) # sku weights
WT = np.array([[0,5], [3, 10]]) # Range of weights allowed by clusters

if __name__ == "__main__":
    res = solve_chung_problem(
        algorithm_name="NSGA2",
        constraints=False,
        D=D,
        Z=Z,
        W = W,
        WT = WT,
        OCM=OCM,
        pop_size=2,
        n_offsprings=2,
        iterations = 20,
    )
```

This problem requires the weights of the skus ``W`` and the capabilities matrix ``WT``, where each row corresponds to the lower and upper limit in weights that the cluster admits.

#### Problem with division constraints

```python
"""Small example with 3 skus, and 2 clusters"""

import numpy as np

from warehouse_allocation.algorithms import solve_chung_problem

D  = np.array([10,10,10]) # Demand for the Skus
Z  = np.array([[2, 2],[2, 0]]) # Clusters capacities
OCM  = np.array([[0,10,0], [10,0,0], [0,0,0]]) # Occurrence Matrix
division_types = np.array([0, 0, 1]) # Type of division of the skus

if __name__ == "__main__":
    res = solve_chung_problem(
        algorithm_name="NSGA2",
        constraints=False,
        D=D,
        Z=Z,
        OCM=OCM,
        division_types = division_types,
        pop_size=2,
        n_offsprings=2,
        iterations = 20,
    )
```

Aisles can have full slots, half slots, third slots, etc. There are times when a sku must go an entire slot, in a half or in another type of division. The capability vector ``Z`` is converted to a matrix, where column ``j`` corresponds to the number of skus of split type ``j``. The ``division_types`` vector corresponds to the division classification for each of the skus starting from 0 and the unique values ​​(``np.unique(division_types)``) must be a perfect enumeration (``0``, ``1``, ...) . The number of unique records in ``division_types`` must match the number of columns in ``Z``.


#### Problem with division and weight constraints

```python
"""Small example with 3 skus, and 2 clusters"""

import numpy as np

from warehouse_allocation.algorithms import solve_chung_problem

D = np.array([10, 10, 10])  # Demand for the skus
Z = np.array([[2, 2], [2, 0]])  # Clusters capacities
OCM = np.array([[0, 10, 0], [10, 0, 0], [0, 0, 0]])  # Occurrence Matrix
W = np.array([10, 2, 4])  # Weights of the skus
WT = np.array([[0, 5], [3, 10]])  # Range of weights allowed by clusters
division_types = np.array([0, 0, 1])  # Type of division of the skus


if __name__ == "__main__":
    res = solve_chung_problem(
        algorithm_name="NSGA2",
        constraints=False,
        D=D,
        Z=Z,
        OCM=OCM,
        division_types=division_types,
        W=W,
        WT=WT,
    )
```

## Solver arguments

The ``solve_chung_problem`` solver has a number of default values, the most notable being ``mutation_prob`` which is a value between 0 and 1 that
controls the proportion of individuals that will mutate in each iteration, with a default value of 0.1. Similarly ``crossover_prob`` for crossover with
default value to 0.9. We also have ``aff_prob`` which is a proportion specific to the crossover operator that can be understood as
At values ​​close to 1 this operator will play more in favor of maximizing the affinity, and close to 0 in favor of uniformly distributing the demand across the clusters.


## Good practices

There are parameter settings that do not make sense, for example, when the capacity of the ``Z`` clusters is too large with respect to the number of skus
total, solutions that maximize affinity will tend to leave some empty cluster, being bad for the performance of the algorithms defined in ``warehouse_allocation``, this
because the crossover and mutation operators make sku changes between clusters, and if one is empty then that cluster will never enter the mating operators.
This package is designed to distribute the skus between clusters where ``Z`` is close enough to the number of skus, to think that the opposite does not even make mathematical sense.

The choice of the ``WT`` array must be done carefully, for example, in a problem with 3 clusters, if we set ``WT = np.array([[10,15], [5,10], [0 ,5]])`` then
the algorithm will *do nothing* in the sense that we are forcing to put all the skus between 10 and 15 kilos in the first cluster, those of 5 and 10 in the second and those of 0 and 5 in the third,
in this case the weight constraint is too *rigid*.


The choice of ``Z`` in the problem that considers divisions can lead to an inconsistency analogous to that described for ``WT``, for example in a problem with 4 clusters
if ``Z = np.array([[10,0], [10, 0], [0,10], [0,10]])``, we are saying that the skus whose ``division_type = 0`` can only be allocated in the first cluster and second cluster, and the skus whose ``division_type = 1`` only appear in the third and fourth, which makes it more convenient to separate the problem into two optimization problems.


## Autodocs (Sphinx)

This project contains self documentation built with [sphinx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html). In order to build the documentation it is first necessary to have [pandoc](https://pandoc.org/) installed on the system and have all the sphinx dependencies installed, which are listed in the ``extensions`` variable.
from the ``docs/conf.py`` file. If the project was installed in development mode (``poetry install``) then the extensions will be installed automatically (except pandoc). If you are on a Unix system, you must also include [make](https://www.gnu.org/software/make/) to compile the documentation.


```sh
# Linux
cd docs
sphinx-apidoc -o api ../warehouse_allocation
make html
```

```sh
# Windows
cd docs
sphinx-apidoc -o api ..\warehouse_allocation
make.bat html
```

The above commands will generate a ``_build`` folder with the project's html, in particular browsing with the ``index.html`` file will show the full documentation.


## TODO

-   English documentation.
-   Coverage 100%.
-   Write a better README
-   Write a better TODO