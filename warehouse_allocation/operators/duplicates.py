# -*- coding: utf-8 -*-
"""Eliminación de duplicados para iteraciones de algoritmo genético."""

import numpy as np
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.population import Population


class DropDuplicates(DuplicateElimination):
    """Elimina los duplicados en cada iteración del algoritmo genético.

    Funcionalidad de ``pymoo`` para asegurar que los individuos
    resultantes del sampling, crossover o  mutation no sean repetidos
    entre sí ni repetidos con la población actual. Considerar remoción
    de duplicados es importante para la performance de un algoritmo
    genético.

    """

    def _do(
        self, pop: Population, other: Population, is_duplicate: np.ndarray
    ) -> np.ndarray:

        # En particular a mi no me gusta la manera interna
        # que tiene pymoo de eliminar duplicados, encuentro
        # que la clase DuplicateElimination y sus métodos es ambigua
        # en pymoo es utilizada en dos partes: En la inicilización
        # y en el mating. En la inicialización se asegura
        # que los individuos provenientes del sampleo no vengan duplicados
        # Luego, en el mating finalizado la aplicación del crossover/mutation
        # Cuando `other is None` refiere a que está comparando
        # solo si hay duplicados en `pop`. Si `other is not None`
        # entonces la variable `other` corresponde a la población
        # actual (ambiguo no?) y `pop` son los candidatos a entrar
        # a la población actual, por ende `pop` es limpiado.
        # No me gusta tampoco la elección de los nombres, es más,
        # en la ejecuciíon del mating hace un llamado interno intercambiado
        # los roles de los nombres.

        pop = self.func(pop)
        # Funcion interna de pymoo que le pega al atributo pop.X
        # que retorna el numpy.ndarray
        if other is None:
            _, unique_indexes = np.unique(pop, return_index=True, axis=0)
            # Retornamos el índice de la primera ocurrencia única
            # en pop
            unique_indexes.sort()
            is_duplicate[
                ...,
                [idx for idx in range(len(pop)) if idx not in unique_indexes],
            ] = True
            return is_duplicate

        pop_conc = np.concatenate((self.func(other), pop), axis=0)
        # Concatenamos para tomar np.unique
        _, unique_indexes = np.unique(pop_conc, return_index=True, axis=0)
        unique_indexes = unique_indexes[unique_indexes >= len(other)] - len(
            other
        )
        # Acá se reescala a los índices de `pop`, en `other` siempre hay únicos
        # y no hay que limpiarlos
        unique_indexes.sort()
        is_duplicate[
            ..., [idx for idx in range(len(pop)) if idx not in unique_indexes]
        ] = True
        # is_duplicate es una lista booleana interna de pymoo que se inicializa en este
        # método con solo valores False, la idea es retornar True en el índice de un
        #  duplicado de `pop`
        return is_duplicate
