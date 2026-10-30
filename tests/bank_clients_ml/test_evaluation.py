"""Tests unitarios para el módulo de evaluación de modelos."""

import numpy as np
import polars as pl
import polars.testing
import pytest

from bank_clients_ml import config, evaluation

DECILE_LABELS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
DECILE_DTYPE = pl.Enum(DECILE_LABELS)


@pytest.fixture
def decile_bins() -> list[float]:
    """Devuelve los cortes que separan las diez categorías de decil.

    ## Returns:
        Lista de nueve cortes equiespaciados entre 0.1 y 0.9.
    """
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.fixture
def train_with_ordered_probabilities() -> tuple[pl.DataFrame, np.ndarray]:
    """Devuelve un entrenamiento con probabilidades y target ordenados.

    ## Returns:
        Datos de entrenamiento y probabilidades crecientes separables.
    """
    probabilities = np.array(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    )
    train = pl.DataFrame({"Target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})
    return train, probabilities


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
    decile_bins: list[float],
) -> None:
    """Verifica todas las métricas por decil con cortes explícitos.

    ## Args:
        settings: Configuración con el nombre de la columna target.
        decile_bins: Cortes que separan las diez categorías de decil.
    """
    probabilities = np.array(
        [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    )
    train = pl.DataFrame({"Target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    result = evaluation._compute_prediction_deciles(
        train, probabilities, settings, decile_bins
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

    polars.testing.assert_frame_equal(result, expected)


def test_should_assign_probability_equal_to_break_to_lower_decile(
    settings: config.Settings,
    decile_bins: list[float],
) -> None:
    """Verifica que los cortes sean cerrados por la derecha.

    ## Args:
        settings: Configuración con el nombre de la columna target.
        decile_bins: Cortes que separan las diez categorías de decil.
    """
    probabilities = np.array([0.1, 0.2, 0.3, 0.9])
    train = pl.DataFrame({"Target": [0, 0, 0, 1]})

    result = evaluation._compute_prediction_deciles(
        train, probabilities, settings, decile_bins
    )

    assert result.get_column("decile").to_list() == ["2", "8", "9", "10"]
    assert _min_prob_for_decile(result, "2") == pytest.approx(90.0)
    assert _min_prob_for_decile(result, "8") == pytest.approx(30.0)
    assert _min_prob_for_decile(result, "9") == pytest.approx(20.0)
    assert _min_prob_for_decile(result, "10") == pytest.approx(10.0)


def test_should_build_ten_equal_deciles_when_bins_are_not_provided(
    settings: config.Settings,
) -> None:
    """Verifica que se construyan diez deciles de igual tamaño por cuantiles.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    probabilities = np.array([0.005 + 0.01 * index for index in range(100)])
    train = pl.DataFrame({"Target": [index % 2 for index in range(100)]})

    result = evaluation._compute_prediction_deciles(train, probabilities, settings)

    assert result.height == 10
    assert result.get_column("decile").to_list() == [
        str(index) for index in range(1, 11)
    ]
    assert result.get_column("count").to_list() == [10] * 10
    assert result.get_column("target_1_count").to_list() == [5] * 10


def test_should_allow_duplicate_probabilities_when_bins_are_not_provided(
    settings: config.Settings,
) -> None:
    """Verifica que las probabilidades repetidas no impidan calcular deciles.

    ## Args:
        settings: Configuración con el nombre de la columna target.
    """
    probabilities = np.array([0.1] * 10 + [0.9] * 10)
    train = pl.DataFrame({"Target": [0, 1] * 10})

    result = evaluation._compute_prediction_deciles(train, probabilities, settings)

    assert result.height == 2
    assert result.get_column("count").to_list() == [10, 10]
    assert result.get_column("target_1_count").to_list() == [5, 5]


def test_should_compute_roc_auc_and_accuracy_from_test_probabilities(
    settings: config.Settings,
    train_with_ordered_probabilities: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """Verifica que las métricas globales se calculen sobre el conjunto de prueba.

    ## Args:
        settings: Configuración con el nombre de la columna target.
        train_with_ordered_probabilities: Entrenamiento y sus probabilidades.
    """
    train, probabilities_train = train_with_ordered_probabilities
    test = pl.DataFrame({"Target": [0, 0, 1, 1]})
    probabilities_test = np.array([0.1, 0.4, 0.35, 0.8])

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
    train_with_ordered_probabilities: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """Verifica que el umbral de clasificación incluya el valor 0.5.

    ## Args:
        settings: Configuración con el nombre de la columna target.
        train_with_ordered_probabilities: Entrenamiento y sus probabilidades.
    """
    train, probabilities_train = train_with_ordered_probabilities
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
    train_with_ordered_probabilities: tuple[pl.DataFrame, np.ndarray],
) -> None:
    """Verifica que se devuelvan los puntos ROC calculados sobre la prueba.

    ## Args:
        settings: Configuración con el nombre de la columna target.
        train_with_ordered_probabilities: Entrenamiento y sus probabilidades.
    """
    train, probabilities_train = train_with_ordered_probabilities
    test = pl.DataFrame({"Target": [0, 0, 1, 1]})
    probabilities_test = np.array([0.1, 0.4, 0.35, 0.8])

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
    train = pl.DataFrame({"Target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})
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
    custom_settings = config.Settings(dataset=config.DatasetColumns(target="label"))
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


@pytest.mark.parametrize(
    ("attribute", "source_attribute"),
    [
        pytest.param(
            "train_deciles_without_probs",
            "train_deciles",
            id="train",
        ),
        pytest.param(
            "test_deciles_without_probs",
            "test_deciles",
            id="test",
        ),
    ],
)
def test_should_drop_probability_columns_from_deciles(
    attribute: str,
    source_attribute: str,
) -> None:
    """Verifica que se omitan las probabilidades mínima y máxima.

    ## Args:
        attribute: Nombre de la propiedad de deciles que se desea evaluar.
        source_attribute: Nombre del atributo con la tabla de deciles original.
    """
    train_deciles = pl.DataFrame(
        {
            "decile": pl.Series(["1", "2"], dtype=DECILE_DTYPE),
            "count": pl.Series([10, 20], dtype=pl.UInt32),
            "target_1_count": pl.Series([3, 7], dtype=pl.UInt32),
            "min_prob": [10.0, 30.0],
            "max_prob": [20.0, 40.0],
            "target_1_rate": [30.0, 35.0],
            "cum_gain": [30.0, 100.0],
            "lift": [1.2, 1.4],
            "ks": [10.0, 20.0],
        }
    )
    test_deciles = train_deciles.reverse()
    metrics = evaluation.EvaluationMetrics(
        roc_auc=0.8,
        accuracy=0.7,
        fpr=np.array([0.0, 1.0]),
        tpr=np.array([0.0, 1.0]),
        train_deciles=train_deciles,
        test_deciles=test_deciles,
    )

    source = getattr(metrics, source_attribute)
    result = getattr(metrics, attribute)

    polars.testing.assert_frame_equal(result, source.drop("min_prob", "max_prob"))
    assert "min_prob" in source.columns
    assert "max_prob" in source.columns
