"""Tests unitarios para el módulo de evaluación de modelos."""

import numpy as np
import polars as pl
import polars.testing
import pytest

import tests.builders.unit.evaluation
import tests.support.settings
from bank_clients_ml import config, evaluation

DECILE_LABELS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
DECILE_DTYPE = pl.Enum(DECILE_LABELS)


def _min_prob_for_decile(deciles: pl.DataFrame, decile: str) -> float:
    """Devuelve la probabilidad mínima registrada para un decil.

    ## Args:
        deciles: Tabla de métricas por decil.
        decile: Etiqueta del decil que se desea consultar.

    ## Returns:
        Probabilidad mínima del decil, expresada en porcentaje.
    """
    return deciles.filter(pl.col("decile") == decile).get_column("min_prob").item()


def test_should_compute_all_decile_metrics_ordered_by_descending_probability(
    settings: config.Settings,
) -> None:
    """Verifica todas las métricas por decil calculadas desde la API pública.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    ordered_probabilities = tests.builders.unit.evaluation.build_ordered_probabilities()
    train = pl.DataFrame({"Target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    metrics = evaluation.evaluate(
        train=train,
        test=train,
        probabilities_train=ordered_probabilities,
        probabilities_test=ordered_probabilities,
        settings=settings,
    )

    expected = pl.DataFrame(
        {
            "decile": pl.Series(
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                dtype=DECILE_DTYPE,
            ),
            "count": pl.Series([1] * 10, dtype=pl.UInt32),
            "target_1_count": pl.Series(
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=pl.UInt32
            ),
            "min_prob": [95.0, 85.0, 75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0],
            "max_prob": [95.0, 85.0, 75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0],
            "target_1_rate": [100.0] * 6 + [0.0] * 4,
            "cum_gain": [100.0 * value / 6 for value in range(1, 7)] + [100.0] * 4,
            "lift": [100.0 / 60.0] * 6 + [0.0] * 4,
            "ks": [100.0 * value / 6 for value in range(1, 7)]
            + [75.0, 50.0, 25.0, 0.0],
        }
    )

    polars.testing.assert_frame_equal(metrics.train_deciles, expected)


def test_should_assign_probability_equal_to_break_to_lower_decile(
    settings: config.Settings,
) -> None:
    """Verifica que los cortes de prueba sean cerrados por la derecha.

    Los cuantiles del entrenamiento producen los mismos cortes que la tabla
    explícita del caso original.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    probabilities_train = np.linspace(0.0, 1.0, 11)
    train = pl.DataFrame({"Target": [value % 2 for value in range(11)]})
    probabilities_test = np.array([0.1, 0.2, 0.3, 0.9])
    test = pl.DataFrame({"Target": [0, 0, 1, 1]})

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=settings,
    )

    assert metrics.test_deciles.get_column("decile").to_list() == [
        "2",
        "8",
        "9",
        "10",
    ]
    assert _min_prob_for_decile(metrics.test_deciles, "2") == pytest.approx(90.0)
    assert _min_prob_for_decile(metrics.test_deciles, "8") == pytest.approx(30.0)
    assert _min_prob_for_decile(metrics.test_deciles, "9") == pytest.approx(20.0)
    assert _min_prob_for_decile(metrics.test_deciles, "10") == pytest.approx(10.0)


def test_should_build_ten_equal_deciles_when_bins_are_not_provided(
    settings: config.Settings,
) -> None:
    """Verifica que se construyan diez deciles de igual tamaño por cuantiles.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    probabilities = np.array([0.005 + 0.01 * index for index in range(100)])
    train = pl.DataFrame({"Target": [index % 2 for index in range(100)]})

    metrics = evaluation.evaluate(
        train=train,
        test=train,
        probabilities_train=probabilities,
        probabilities_test=probabilities,
        settings=settings,
    )

    assert metrics.train_deciles.height == 10
    assert metrics.train_deciles.get_column("decile").to_list() == [
        str(index) for index in range(1, 11)
    ]
    assert metrics.train_deciles.get_column("count").to_list() == [10] * 10
    assert metrics.train_deciles.get_column("target_1_count").to_list() == [5] * 10


def test_should_raise_value_error_when_quantile_breaks_are_not_unique(
    settings: config.Settings,
) -> None:
    """Verifica que no se calculen deciles si los cortes se repiten.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    probabilities = np.array([0.1] * 10 + [0.9] * 10)
    train = pl.DataFrame({"Target": [0, 1] * 10})

    with pytest.raises(
        ValueError,
        match="got 3 unique values from 9 breaks",
    ):
        evaluation.evaluate(
            train=train,
            test=train,
            probabilities_train=probabilities,
            probabilities_test=probabilities,
            settings=settings,
        )


def test_should_compute_roc_auc_and_accuracy_from_test_probabilities(
    settings: config.Settings,
) -> None:
    """Verifica que las métricas globales se calculen sobre el conjunto de prueba.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    train, probabilities_train = (
        tests.builders.unit.evaluation.build_train_with_ordered_probabilities()
    )
    test, probabilities_test = (
        tests.builders.unit.evaluation.build_binary_test_with_probabilities()
    )

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=settings,
    )

    assert metrics.roc_auc == pytest.approx(0.75)
    assert metrics.accuracy == pytest.approx(0.75)


def test_should_treat_probability_equal_to_half_as_positive(
    settings: config.Settings,
) -> None:
    """Verifica que el umbral de clasificación incluya el valor 0.5.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    train, probabilities_train = (
        tests.builders.unit.evaluation.build_train_with_ordered_probabilities()
    )
    test = pl.DataFrame({"Target": [0, 1, 1]})
    probabilities_test = np.array([0.2, 0.5, 0.8])

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=settings,
    )

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_should_return_roc_curve_points_for_test_probabilities(
    settings: config.Settings,
) -> None:
    """Verifica que se devuelvan los puntos ROC calculados sobre la prueba.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    train, probabilities_train = (
        tests.builders.unit.evaluation.build_train_with_ordered_probabilities()
    )
    test, probabilities_test = (
        tests.builders.unit.evaluation.build_binary_test_with_probabilities()
    )

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=settings,
    )

    np.testing.assert_allclose(metrics.fpr, [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(metrics.tpr, [0.0, 0.5, 0.5, 1.0, 1.0])


def test_should_use_train_quantiles_to_bin_test_deciles(
    settings: config.Settings,
) -> None:
    """Verifica que los deciles de prueba usen los cortes del entrenamiento.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    train = tests.builders.unit.evaluation.build_balanced_target_train()
    probabilities_train = np.linspace(0.01, 0.10, 10)
    test = pl.DataFrame({"Target": [1, 0, 1]})
    probabilities_test = np.array([0.8, 0.85, 0.9])

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=settings,
    )

    assert metrics.train_deciles.height == 10
    assert metrics.test_deciles.height == 1
    row = metrics.test_deciles.row(0, named=True)
    assert row["decile"] == "1"
    assert row["count"] == 3
    assert row["target_1_count"] == 2
    assert row["min_prob"] == pytest.approx(80.0)
    assert row["max_prob"] == pytest.approx(90.0)


def test_should_use_target_column_name_from_settings() -> None:
    """Verifica que se respete el nombre de target configurado."""
    custom_settings = tests.support.settings.build_settings_for_tests(
        dataset=config.DatasetColumns(target="label")
    )
    train = pl.DataFrame({"label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})
    probabilities_train = np.linspace(0.05, 0.95, 10)
    test = pl.DataFrame({"label": [0, 1, 0, 1]})
    probabilities_test = np.array([0.1, 0.4, 0.35, 0.8])

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=custom_settings,
    )

    assert metrics.train_deciles.get_column("target_1_count").sum() == 5
    assert metrics.test_deciles.get_column("target_1_count").sum() == 2
