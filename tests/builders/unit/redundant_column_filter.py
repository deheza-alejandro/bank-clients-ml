"""Constructores de datos usados por los tests de filtrado de columnas."""

import polars as pl

from bank_clients_ml import config, redundant_column_filter


def build_train_data() -> pl.DataFrame:
    """Devuelve un conjunto de entrenamiento con columnas redundantes.

    ## Returns:
        DataFrame con una columna constante, una binaria desbalanceada, una
        binaria balanceada y dos variables numéricas perfectamente correlacionadas.
    """
    return pl.DataFrame(
        {
            "client_id": [f"client_{i}" for i in range(1, 11)],
            "Target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "constant": [7] * 10,
            "imbalanced": [0] * 9 + [1],
            "balanced": [0, 1] * 5,
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature_b": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                14.0,
                16.0,
                18.0,
                20.0,
            ],
        }
    )


def build_test_data() -> pl.DataFrame:
    """Devuelve un conjunto de prueba con el mismo esquema que el de entrenamiento.

    ## Returns:
        DataFrame de prueba con cuatro observaciones.
    """
    return pl.DataFrame(
        {
            "client_id": [f"test_{i}" for i in range(1, 5)],
            "Target": [0, 1, 0, 1],
            "constant": [7] * 4,
            "imbalanced": [0, 0, 0, 1],
            "balanced": [0, 1, 0, 1],
            "feature_a": [11.0, 12.0, 13.0, 14.0],
            "feature_b": [22.0, 24.0, 26.0, 28.0],
        }
    )


def build_binary_frame() -> pl.DataFrame:
    """Devuelve un DataFrame con target y feature binarios.

    ## Returns:
        DataFrame con una fila por valor binario.
    """
    return pl.DataFrame({"Target": [0, 1], "feature": [0, 1]})


def build_redundant_filter(
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve un filtro generado a partir de los datos de prueba.

    ## Args:
        train_data: Conjunto de entrenamiento con columnas redundantes.
        test_data: Conjunto de prueba con el mismo esquema.
        settings: Configuración global del proyecto.

    ## Returns:
        Filtro con las columnas redundantes ya detectadas.
    """
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train_data,
        test=test_data,
        imbalanced_binary_threshold=0.2,
        correlation_threshold=0.8,
        settings=settings,
    )
