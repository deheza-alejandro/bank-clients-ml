"""Constructores de datos usados por los tests de entrenamiento."""

import polars as pl


def build_importances() -> pl.DataFrame:
    """Devuelve una tabla de importancias ordenada de mayor a menor.

    ## Returns:
        Tabla con tres variables y sus importancias.
    """
    return pl.DataFrame(
        {
            "Feature": ["Feature_A", "Feature_B", "Feature_C"],
            "Importance": [0.5, 0.3, 0.2],
        }
    )


def build_minimal_train() -> pl.DataFrame:
    """Devuelve un entrenamiento mínimo con target binario.

    ## Returns:
        DataFrame con una fila por clase del target.
    """
    return pl.DataFrame({"Target": [0, 1]})
