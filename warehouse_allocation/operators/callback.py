# -*- coding: utf-8 -*-
"""Callaback para algoritmos y problemas.

Los algoritmos que hereden de
:class:`pymoo.algorithms.genetic_algorithm.GeneticAlgorithm`
permiten usar un callaback que se activa en cada iteración,
y puede ser una clase que here de
:class:`pymoo.model.callback.Callback` o un ``callable``
que tome como parámetro único el algoritmo mismo.

Los problemas que hereden de
:class:`pymoo.model.problem.Problem` permiten usar un callback
que se activa igualmente en cada iterción, y debe ser un
``callable`` que reciba como parámetro único la matriz poblacional
de individuos ``X_pop``, es decir, cada fila es un individuo.

"""

import numpy as np
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.callback import Callback
from pymoo.operators.mutation.nom import NoMutation

from warehouse_allocation import logger


class ReportObjectiveBestValuesCallback(Callback):
    """Reporta los mejores valores de los objetivos en cada iteración.

    Este callback guarda los mejores valores para cada una de las
    funciones objetivos. Por ejemplo, si el algoritmo intenta resolver
    un problema bi-objetivo, se seteará un diccionario dado por el
    atriburo ``self.data``, con dos llaves, ``F_1`` y ``F_2``, donde
    cada value es una lista. Cada elemento en ``self.data["F_1"]``, es
    una tupla conteniendo el mejor valor de ``F_1`` para cada iteración,
    adicionando su valor de ``F_2``. Análogo para  ``self.data["F_2"]``.

    """

    def __init__(self, n_obj: int = 2) -> None:
        logger.info(
            f"Callback para {n_obj} objetivos. Si el algoritmo no tiene esa cantidad, habrá `IndexError`."
        )
        super().__init__()
        for n in range(1, n_obj + 1):
            self.data[f"F_{n}"] = np.empty((0, n_obj), float)

    def notify(self, algorithm: GeneticAlgorithm) -> None:

        F = algorithm.pop.get("F")

        for n in range(1, F.shape[1] + 1):
            self.data[f"F_{n}"] = np.append(
                self.data[f"F_{n}"], [F[F[:, n - 1].argmin()]], axis=0
            )


class StopMutationAfterNgenCallback:
    """Anula el operador mutation despúes de cierta generación.

    Este callback está pensado para hacer una búsqueda intensiva con el
    operador de mutation en las primeras generaciones, y luego cuando ya
    se tengan individuos de alto rank, se corta la mutación para dejar
    actuar netamente al operador crossover.

    """

    def __init__(self, after_gen: int) -> None:
        logger.info(
            f"Callback para anular la mutación después de la generación {after_gen}."
        )
        self.after_gen = after_gen

    def __call__(self, algorithm: GeneticAlgorithm) -> None:

        if algorithm.n_gen == self.after_gen:
            logger.info("Callback de anular mutación activado")
            algorithm.mating.mutation = NoMutation()
            # Apagamos el operador de mutation
            algorithm.callback = None
            # Apagamos el callback
