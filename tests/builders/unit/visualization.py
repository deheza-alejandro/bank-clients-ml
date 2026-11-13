"""Constructores de datos y objetos usados por los tests de visualización."""

import pathlib

import numpy as np
import polars as pl
from matplotlib.figure import Figure

from bank_clients_ml import bivariate_table, evaluation, visualization


def build_capture_saver() -> tuple[
    list[tuple[Figure, pathlib.Path]], visualization.FigureSaver
]:
    """Devuelve una estrategia de guardado que captura figuras y rutas.

    ## Returns:
        Lista de capturas y la estrategia que las acumula.
    """
    saved: list[tuple[Figure, pathlib.Path]] = []

    def saver(fig: Figure, path: pathlib.Path) -> None:
        saved.append((fig, path))

    return saved, saver


def build_top_features_table() -> pl.DataFrame:
    """Devuelve una tabla de importancias ordenada de mayor a menor.

    ## Returns:
        Tabla con cinco variables y sus importancias.
    """
    return pl.DataFrame(
        {
            "Feature": ["Age", "Income", "Balance", "Tenure", "Score"],
            "Importance": [50.0, 30.0, 15.0, 4.0, 1.0],
        }
    )


def build_bivariate_age() -> bivariate_table.BivariateTable:
    """Devuelve una tabla bivariada de ejemplo para la edad.

    ## Returns:
        Tabla con tres bines y valores numéricos para formatear.
    """
    table = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": [18.456, 35.5, 60.0],
            "Max": [34.0, 59.5, 80.0],
            "Clients": pl.Series([10, 20, 30], dtype=pl.UInt32),
            "Target": pl.Series([1, 4, 9], dtype=pl.Int64),
            "Target_pct": [10.4, 19.6, 30.49],
        }
    )
    return bivariate_table.BivariateTable("age", "Age", table)


def build_evaluation_metrics() -> evaluation.EvaluationMetrics:
    """Devuelve métricas de evaluación con deciles de entrenamiento y prueba.

    ## Returns:
        Métricas con curva ROC y tablas de deciles de ejemplo.
    """
    train_deciles = pl.DataFrame(
        {
            "decile": ["1", "2"],
            "count": [10, 20],
            "target_1_count": [3, 7],
            "min_prob": [10.0, 30.0],
            "max_prob": [20.0, 40.0],
            "target_1_rate": [30.0, 35.0],
            "cum_gain": [30.0, 100.0],
            "lift": [1.234, 1.456],
            "ks": [10.0, 20.0],
        }
    )
    test_deciles = pl.DataFrame(
        {
            "decile": ["1", "2"],
            "count": [12, 18],
            "target_1_count": [2, 8],
            "min_prob": [15.0, 35.0],
            "max_prob": [25.0, 45.0],
            "target_1_rate": [16.666, 44.444],
            "cum_gain": [20.0, 100.0],
            "lift": [0.8, 1.2],
            "ks": [5.0, 25.0],
        }
    )
    return evaluation.EvaluationMetrics(
        roc_auc=0.876543,
        accuracy=0.765432,
        fpr=np.array([0.0, 0.2, 1.0]),
        tpr=np.array([0.0, 0.8, 1.0]),
        train_deciles=train_deciles,
        test_deciles=test_deciles,
    )
