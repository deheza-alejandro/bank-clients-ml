"""Constructores de datos usados por los tests de evaluación."""

import numpy as np
import polars as pl


def build_balanced_target_train() -> pl.DataFrame:
    """Devuelve un entrenamiento con cinco ceros y cinco unos.

    ## Returns:
        DataFrame con un target binario balanceado.
    """
    return pl.DataFrame({"Target": [0] * 5 + [1] * 5})


def build_ordered_probabilities() -> np.ndarray:
    """Devuelve diez probabilidades crecientes y separables.

    ## Returns:
        Array con probabilidades ordenadas de 0.05 a 0.95.
    """
    return np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])


def build_train_with_ordered_probabilities() -> tuple[pl.DataFrame, np.ndarray]:
    """Devuelve un entrenamiento con probabilidades y target ordenados.

    ## Returns:
        Datos de entrenamiento y probabilidades crecientes separables.
    """
    return build_balanced_target_train(), build_ordered_probabilities()


def build_binary_test_with_probabilities() -> tuple[pl.DataFrame, np.ndarray]:
    """Devuelve un conjunto de prueba binario con sus probabilidades.

    ## Returns:
        DataFrame de prueba y probabilidades asociadas.
    """
    test = pl.DataFrame({"Target": [0, 0, 1, 1]})
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    return test, probabilities
