# # -*- coding: utf-8 -*-
"""Algoritmos para resolver el problema de Chung.

En este modulo están los algoritmos para resolver el problema
de Chung original, es decir, tanto la afinidad como el tráfico son
vistos como un objetivo a optimizar. Además, los métodos incluyen
(opcional) restricciones de pesos, lo que es un aditivo al problema
del paper. El método que resuelve el problema de chung esta dado por
:func:`~warehouse_allocation.models.algorithms.solve_chung_problem`.

"""

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.result import Result
from pymoo.core.sampling import Sampling
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from warehouse_allocation import logger
from warehouse_allocation.models import ChungProblem
from warehouse_allocation.models import ChungProblemNotConstraints
from warehouse_allocation.models import ChungProblemStrictlyWeighted
from warehouse_allocation.models import ChungProblemStrictlyWeightedNotConstraints
from warehouse_allocation.models import ChungProblemWithDivisions
from warehouse_allocation.models import ChungProblemWithDivisionsNotConstraints
from warehouse_allocation.models import ChungProblemWithDivisionsPlusWeight
from warehouse_allocation.models import (
    ChungProblemWithDivisionsPlusWeightNotConstraints,
)
from warehouse_allocation.operators import ChungAislePermutationMutation
from warehouse_allocation.operators import ChungAislePermutationMutationWeighted
from warehouse_allocation.operators import ChungAislePermutationWithDivisions
from warehouse_allocation.operators import (
    ChungAislePermutationWithDivisionsPlusWeight,
)
from warehouse_allocation.operators import ChungPartiallyMappedCrossover
from warehouse_allocation.operators import ChungPartiallyMappedCrossoverWeigthed
from warehouse_allocation.operators import ChungPartiallyMappedCrossoverWithDivisions
from warehouse_allocation.operators import NonDominatedSortingSampling
from warehouse_allocation.operators import NonDominatedSortingSamplingWeighted
from warehouse_allocation.operators import NonDominatedSortingSamplingWithDivisions
from warehouse_allocation.operators.duplicates import DropDuplicates

ALGORITHMS = {
    "NSGA2": NSGA2,
    "NSGA3": NSGA3,
    "RNSGA2": RNSGA2,
    "UNSGA3": UNSGA3,
}
"""Algoritmos genéticos disponibles en ``warehouse_allocation``"""


def select_operators(
    problem: Union[
        ChungProblem,
        ChungProblemStrictlyWeighted,
        ChungProblemWithDivisions,
        ChungProblemWithDivisionsNotConstraints,
    ]
) -> Tuple[Sampling, Mutation, Crossover]:
    """Selecciona operadores mating de acuerdo al problema.

    :param problem: Problema que se intenta resolver.

    :return: Una tupla con el operador de sampling,
        mutation y crossover que es compatible
        con los problemas multiobjetivo disponibles
        en :py mod:`~warehouse_allocation.models.chung`

    """

    if isinstance(problem, ChungProblemWithDivisionsPlusWeight):
        return (
            NonDominatedSortingSamplingWithDivisions,
            ChungAislePermutationWithDivisionsPlusWeight,
            ChungPartiallyMappedCrossoverWithDivisions,
        )

    if isinstance(problem, ChungProblemWithDivisions):
        return (
            NonDominatedSortingSamplingWithDivisions,
            ChungAislePermutationWithDivisions,
            ChungPartiallyMappedCrossoverWithDivisions,
        )

    if isinstance(problem, ChungProblemStrictlyWeighted):
        return (
            NonDominatedSortingSamplingWeighted,
            ChungAislePermutationMutationWeighted,
            ChungPartiallyMappedCrossoverWeigthed,
        )

    if isinstance(problem, ChungProblem):
        return (
            NonDominatedSortingSampling,
            ChungAislePermutationMutation,
            ChungPartiallyMappedCrossover,
        )

    raise NotImplementedError(
        f"Operadores de mating no definidos para problema {problem}"
    )


def init_problem(
    D: Union[List[float], np.ndarray],
    Z: Union[List[int], np.ndarray],
    OCM: np.ndarray,
    W: Union[List[float], np.ndarray] = None,
    WT: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
    division_types: Optional[Union[List[int], np.ndarray]] = None,
    clusters_penalizations: Optional[Union[List[float], np.ndarray]] = None,
    constraints: bool = False,
) -> Union[
    ChungProblem,
    ChungProblemStrictlyWeighted,
    ChungProblemWithDivisions,
    ChungProblemWithDivisionsNotConstraints,
    ChungProblemNotConstraints,
    ChungProblemStrictlyWeightedNotConstraints,
    ChungProblemWithDivisionsNotConstraints,
    ChungProblemWithDivisionsPlusWeightNotConstraints,
]:
    """Instanciación del problema de Chung.

    Requiere la información mínima para definir el problema
    de Chung, esto es, la demanda ``D``, la capacidad
    de los clusters ``Z``, y la matriz de afinidad
    ``OCM``.

    :param D: Vector de demandas de los skus.
    :param Z: Lista conteniendo la capacidad por tipos
        de skus. Siguiendo la notación de Chung,
        corresponden a los valores :math:`Z_k`.
    :param OCM: Matriz de ocurrencia en la cual cada entrada
        :math:`(j,j')` corresponde a la cantidad de veces
        que los skus :math:`j` y :math:`j'` han aparecido juntos
        en ordenes de pedidos. Siguiendo la notación
        de Chung, cada entrada corresponde a el
        valor :math:`N_{j,j'}`.
    :param W: Vector de pesos de los SKUs.
    :param WT: Tolerancia en pesos de los clusters.
        La convención es una lista de tuplas de la forma
        :math:`(w_{\\text{k,min}}, w_{\\text{k,max}})`, con
        :math:`k` moviendose en la indexación de los clusters.
    :param division_types: Es un arreglo de largo igual a la cantidad
        de skus. La entrada :math:`i` corresponde a la clasificación
        de división del sku :math:`i`.

    :raises TypeError: Si se entrega la tolerancia en pesos de
        los clusters, pero no el vector de pesos de los SKUs
        o viceversa.

    """

    if sum([WT is None, W is None]) == 1:
        raise TypeError("No puede setearse `WT` sin setear `W`")

    if W is None:
        if division_types is None:
            logger.info(
                "Problema a resolver: Problema de Chung original",
                constraints=constraints,
            )
            return (
                ChungProblem(
                    D=D,
                    Z=Z,
                    OCM=OCM,
                    clusters_penalizations=clusters_penalizations,
                )
                if constraints
                else ChungProblemNotConstraints(
                    D=D,
                    Z=Z,
                    OCM=OCM,
                    clusters_penalizations=clusters_penalizations,
                )
            )

        logger.info(
            "Problema a resolver: Problema de Chung original +  restricciones de división",
            constraints=constraints,
        )

        return (
            ChungProblemWithDivisions(
                D=D,
                Z=Z,
                OCM=OCM,
                division_types=division_types,
                clusters_penalizations=clusters_penalizations,
            )
            if constraints
            else ChungProblemWithDivisionsNotConstraints(
                D=D,
                Z=Z,
                OCM=OCM,
                division_types=division_types,
                clusters_penalizations=clusters_penalizations,
            )
        )

    if division_types is None:
        logger.info(
            "Problema a resolver: Problema de Chung original +  restricciones de peso sobre clusters",
            constraints=constraints,
        )
        return (
            ChungProblemStrictlyWeighted(
                D=D,
                Z=Z,
                OCM=OCM,
                W=W,
                WT=WT,
                clusters_penalizations=clusters_penalizations,
            )
            if constraints
            else ChungProblemStrictlyWeightedNotConstraints(
                D=D,
                Z=Z,
                OCM=OCM,
                W=W,
                WT=WT,
                clusters_penalizations=clusters_penalizations,
            )
        )

    logger.info(
        "Problema a resolver: Problema de Chung original + restricciones de división + restricciones de peso sobre clusters",
        constraints=constraints,
    )

    return (
        ChungProblemWithDivisionsPlusWeight(
            D=D,
            Z=Z,
            OCM=OCM,
            W=W,
            WT=WT,
            division_types=division_types,
            clusters_penalizations=clusters_penalizations,
        )
        if constraints
        else ChungProblemWithDivisionsPlusWeightNotConstraints(
            D=D,
            Z=Z,
            OCM=OCM,
            W=W,
            WT=WT,
            division_types=division_types,
            clusters_penalizations=clusters_penalizations,
        )
    )


def solve_chung_problem(
    *,
    algorithm_name: str,
    D: Union[List[float], np.ndarray],
    Z: Union[List[int], np.ndarray],
    OCM: np.ndarray,
    W: Union[List[float], np.ndarray, None] = None,
    WT: Union[List[Tuple[float, float]], np.ndarray, None] = None,
    division_types: Union[List[int], np.ndarray, None] = None,
    clusters_penalizations: Optional[Union[List[float], np.ndarray]] = None,
    sampling: Optional[np.ndarray] = None,
    constraints: bool = False,
    algorithm_callback: Optional[Callback] = None,
    mutation_prob: float = 0.1,
    crossover_prob: float = 0.9,
    crossover_aff_prob: float = 0.5,
    iterations: int = 300,
    verbose=True,
    save_history=False,
    **kwargs,
) -> Result:
    """Solver del problema de Chung en sus diferentes formulacionees.

    Requiere la información mínima para definir el problema
    de Chung, esto es, la demanda ``D``, la capacidad
    de los clusters ``Z``, y la matriz de afinidad
    ``OCM``.

    Además:
        - Si se entregan ``W`` y ``WT`` se resuelve el problema con restricciones
          de peso sobre los clusters.
        - Si se entregan ``division_types`` se resuelve el problema
          restricciones de rotación/división en los clusters.
          En este caso la cantidad de columnas de ``Z``
          debe coincidir con la de valores únicos de ``division_types``.
        - Si se entregan ``W``, ``WT`` y ``division_types`` se resuleve
          el problema que contiene todas las restricciones mencionadas
          anteriormente.

    :param algorithm_name: Nombre del algoritmo. Válidos
        ``NSGA2``, ``NSGA3``, ``RNSGA2``, ``UNSGA3``. Cabe
        destacar que algoritmos diferentes al ``NSGA2`` necesitan
        configuración adicional que debe ser seteada en los
        ``kwargs``. Para más información ver en ``pymoo`` los
        `algoritmos <https://pymoo.org/algorithms/index.html>`_.
    :param D: Vector de demandas de los skus.
    :param Z: Lista conteniendo la capacidad por tipos
        de skus. Siguiendo la notación de Chung,
        corresponden a los valores :math:`Z_k`.
    :param OCM: Matriz de ocurrencia en la cual cada entrada
        :math:`(j,j')` corresponde a la cantidad de veces
        que los skus :math:`j` y :math:`j'` han aparecido juntos
        en ordenes de pedidos. Siguiendo la notación
        de Chung, cada entrada corresponde a el
        valor :math:`N_{j,j'}`.
    :param W: Opcional. Vector de pesos de los SKUs.
    :param WT: Opcional. Tolerancia en pesos de los clusters.
        La convención es una lista de tuplas de la forma
        :math:`(w_{\\text{k,min}}, w_{\\text{k,max}})`, con
        :math:`k` moviendose en la indexación de los clusters.
    :param division_types: Es un arreglo de largo igual a la cantidad
        de skus. La entrada :math:`i` corresponde a la clasificación
        de división del sku :math:`i`.
    :param sampling: Si es provisto como ``np.ndarray``, se usará
        en vez de la técnica de sampleo aleatoria de ``warehouse_allocation``.
    :param constraints: Si es ``True``, en cada iteración ``pymoo`` calculará
        las constraints del problema para tomar decisiones de selección
        sobre los individuos. Solo tiene sentido en un ambiente
        de desarrollo o testeo debido a que los operadores
        mating que se usan en este método respetan todas las
        constraints, implicando un paso innecesario en la búsqueda
        de la curva de Pareto.
    :param algorithm_callback: Callback para algortimos. Ver
        :py:mod:`warehouse_allocation.operators.callback`.
    :param mutation_prob: Porcentaje de la población que se le aplicará
        mutación. Ver :py:mod:`warehouse_allocation.operators.mutation`.
    :param crossover_prob: Porcentaje de la población a los que se le
        aplicará crossover. Ver :py:mod:`warehouse_allocation.operators.crossover`
    :param crossover_aff_prob: Parámetro ``aff_prob`` en
        :class:`warehouse_allocation.operators.crossover.ChungPartiallyMappedCrossoverBase`.
    :param iterations: Cantidad de iteraciones.
    :param verbose: Si es ``True`` imprime métricas con valores por defecto
        controlados por ``pymoo``.
    :param save_history: Guarda la historia de los valores objetivos
        en cada iteraciones en ``problem``. Solo usar para una cantidad
        pequeña iteraciones para no romper la memoria RAM y con fines
        de testing/development.

    :param kwargs: Extra ``kwargs`` pasados a la clase
        algoritmo de ``pymoo``. Notar que la clase algoritmo
        instanciada será controlada por ``algorithm_name``.

    :return: `Resultado <https://pymoo.org/interface/result.html>`_.

    """

    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algoritmos válidos son {list(ALGORITHMS.keys())}")

    if (algorithm_name in ("NSGA3", "UNSGA3")) and "ref_dirs" not in kwargs:
        raise TypeError(
            f"Algoritmo {algorithm_name} requiere setear `ref_dirs` para su inicialización"
        )

    if algorithm_name == "RNSGA2" and "ref_points" not in kwargs:
        raise TypeError(
            f"Algoritmo {algorithm_name} requiere setear `ref_points` para su inicialización"
        )

    problem = init_problem(
        D=D,
        Z=Z,
        OCM=OCM,
        W=W,
        WT=WT,
        division_types=division_types,
        clusters_penalizations=clusters_penalizations,
        constraints=constraints,
    )

    sampling_class, mutation_class, crossover_class = select_operators(problem)

    kwargs.setdefault("eliminate_duplicates", DropDuplicates())

    algorithm = ALGORITHMS[algorithm_name](
        crossover=crossover_class(
            prob=crossover_prob, aff_prob=crossover_aff_prob
        ),
        mutation=mutation_class(prob=mutation_prob),
        sampling=sampling_class() if sampling is None else sampling,
        **kwargs,
    )

    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_iter", iterations),
        verbose=verbose,
        save_history=save_history,
        callback=algorithm_callback,
    )

    return result
