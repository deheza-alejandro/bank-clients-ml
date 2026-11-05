"""Tests unitarios para el filtrado de columnas redundantes."""

import types

import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock

from bank_clients_ml import config, redundant_column_filter


@pytest.fixture
def train_data() -> pl.DataFrame:
    """Devuelve un conjunto de entrenamiento con columnas redundantes.

    Returns:
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


@pytest.fixture
def test_data() -> pl.DataFrame:
    """Devuelve un conjunto de prueba con el mismo esquema que el de entrenamiento.

    Returns:
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


@pytest.fixture
def binary_frame() -> pl.DataFrame:
    """Devuelve un DataFrame con target y feature binarios.

    Returns:
        DataFrame con una fila por valor binario.
    """
    return pl.DataFrame({"Target": [0, 1], "feature": [0, 1]})


@pytest.fixture
def redundant_filter(
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve un filtro generado a partir de los datos de prueba.

    Args:
        train_data: Conjunto de entrenamiento con columnas redundantes.
        test_data: Conjunto de prueba con el mismo esquema.
        settings: Configuración global del proyecto.

    Returns:
        Filtro con las columnas redundantes ya detectadas.
    """
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train_data,
        test=test_data,
        imbalanced_binary_threshold=0.2,
        correlation_threshold=0.8,
        settings=settings,
    )


def _orthonormal_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve tres vectores centrados, ortogonales y de norma unitaria."""
    first = np.array([-3.0, -1.0, 1.0, 3.0])
    second = np.array([1.0, -3.0, 3.0, -1.0])
    third = np.array([1.0, -1.0, -1.0, 1.0])
    return (
        first / np.linalg.norm(first),
        second / np.linalg.norm(second),
        third / np.linalg.norm(third),
    )


def _frame_with_features(**features: np.ndarray) -> pl.DataFrame:
    """Construye un DataFrame con identificador, target y las features indicadas."""
    return pl.DataFrame(
        {
            "client_id": [f"client_{index}" for index in range(4)],
            "Target": [0, 0, 1, 1],
            **{name: values.tolist() for name, values in features.items()},
        }
    )


def _correlated_frame() -> pl.DataFrame:
    """Devuelve features con correlaciones conocidas entre sí."""
    first, second, third = _orthonormal_features()
    feature_a = first
    feature_b = 0.9 * first + np.sqrt(1.0 - 0.9**2) * second
    feature_c = -0.95 * first + np.sqrt(1.0 - 0.95**2) * third
    return _frame_with_features(
        feature_a=feature_a,
        feature_b=feature_b,
        feature_c=feature_c,
    )


def _negative_correlated_frame() -> pl.DataFrame:
    """Devuelve dos features con correlación negativa conocida."""
    first, second, _ = _orthonormal_features()
    return _frame_with_features(
        feature_a=first,
        feature_b=-0.95 * first + np.sqrt(1.0 - 0.95**2) * second,
    )


def _correlation_matrix(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    """Calcula la matriz de correlación de las columnas indicadas."""
    return frame.select(*columns).corr()


def test_should_remove_constant_columns_from_train_and_test(
    settings: config.Settings,
) -> None:
    """Verifica que las columnas constantes se eliminen de ambos conjuntos.

    Args:
        settings: Configuración global del proyecto.
    """
    train = pl.DataFrame(
        {
            "client_id": ["a", "b", "c", "d"],
            "Target": [0, 1, 0, 1],
            "constant_int": [7, 7, 7, 7],
            "constant_text": ["x", "x", "x", "x"],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    test = pl.DataFrame(
        {
            "client_id": ["e", "f"],
            "Target": [0, 1],
            "constant_int": [7, 7],
            "constant_text": ["x", "x"],
            "feature_a": [5.0, 6.0],
            "feature_b": [3.0, 2.0],
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        settings=settings,
    )

    expected_constants = ("constant_int", "constant_text")
    assert result.constant_cols == expected_constants
    polars.testing.assert_frame_equal(
        result.reduced_train,
        train.drop(expected_constants),
    )
    polars.testing.assert_frame_equal(
        result.correlated_test,
        test.drop(expected_constants),
    )


def test_should_remove_imbalanced_binary_columns_from_train_and_test(
    redundant_filter: redundant_column_filter.RedundantColumnFilter,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
) -> None:
    """Verifica que las columnas binarias desbalanceadas se eliminen de ambos conjuntos.

    Args:
        redundant_filter: Filtro generado a partir de los datos de prueba.
        train_data: Conjunto de entrenamiento original.
        test_data: Conjunto de prueba original.
    """
    assert redundant_filter.imbalanced_binary_columns == ("imbalanced",)
    polars.testing.assert_frame_equal(
        redundant_filter.correlated_train,
        train_data.drop(["constant", "imbalanced"]),
    )
    polars.testing.assert_frame_equal(
        redundant_filter.correlated_test,
        test_data.drop(["constant", "imbalanced"]),
    )


def test_should_remove_correlated_columns_only_from_uncorrelated_train(
    redundant_filter: redundant_column_filter.RedundantColumnFilter,
    train_data: pl.DataFrame,
) -> None:
    """Verifica que las columnas correlacionadas se descarten del entrenamiento.

    Args:
        redundant_filter: Filtro generado a partir de los datos de prueba.
        train_data: Conjunto de entrenamiento original.
    """
    assert redundant_filter.correlated_columns == ("feature_b",)
    polars.testing.assert_frame_equal(
        redundant_filter.uncorrelated_train,
        train_data.drop(["constant", "imbalanced", "feature_b"]),
    )
    assert redundant_filter.correlated_train.select(
        pl.exclude("client_id", "Target")
    ).columns == ["balanced", "feature_a", "feature_b"]


def test_should_use_global_settings_and_default_thresholds_when_not_provided(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    train_data: pl.DataFrame,
) -> None:
    """Verifica los valores por defecto al omitir configuración y umbrales.

    Args:
        mocker: Utilidad para sustituir la obtención de la configuración global.
        settings: Configuración global simulada.
        train_data: Conjunto de entrenamiento de prueba.
    """
    get_settings = mocker.patch.object(
        redundant_column_filter.config,
        "get_settings",
        return_value=settings,
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=train_data,
        test=train_data,
    )

    get_settings.assert_called_once_with()
    assert result.settings is settings
    assert result.imbalanced_binary_threshold == pytest.approx(0.1)
    assert result.correlation_threshold == pytest.approx(0.8)


def test_should_exclude_custom_id_and_target_from_correlation_analysis() -> None:
    """Verifica que se respeten los nombres personalizados de id y target."""
    custom_settings = config.Settings(
        dataset=config.DatasetColumns(id="customer", target="outcome")
    )
    train = pl.DataFrame(
        {
            "customer": [1, 2, 3, 4],
            "outcome": [0, 1, 0, 1],
            "constant": [5, 5, 5, 5],
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [2.0, 4.0, 6.0, 8.0],
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=train,
        settings=custom_settings,
    )

    assert result.correlated_columns == ("feat_b",)
    assert result.uncorrelated_train.columns == ["customer", "outcome", "feat_a"]


@pytest.mark.parametrize(
    ("zeros", "ones", "expected"),
    [
        pytest.param(1, 19, True, id="below-minority-threshold"),
        pytest.param(19, 1, True, id="above-majority-threshold"),
        pytest.param(2, 18, False, id="exactly-at-minority-threshold"),
        pytest.param(18, 2, False, id="exactly-at-majority-threshold"),
        pytest.param(10, 10, False, id="balanced"),
    ],
)
def test_should_detect_imbalanced_binary_columns_at_threshold_boundaries(
    settings: config.Settings,
    zeros: int,
    ones: int,
    expected: bool,
) -> None:
    """Verifica la proporción mínima exigida para una columna binaria.

    Args:
        settings: Configuración con el nombre de la columna target.
        zeros: Cantidad de ceros en la columna evaluada.
        ones: Cantidad de unos en la columna evaluada.
        expected: Indica si la columna debe considerarse desbalanceada.
    """
    data = pl.DataFrame(
        {
            "client_id": list(range(20)),
            "Target": [0] * 10 + [1] * 10,
            "feature": [0] * zeros + [1] * ones,
            "numeric": [float(index) for index in range(20)],
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        imbalanced_binary_threshold=0.1,
        settings=settings,
    )

    expected_columns = ("feature",) if expected else ()
    assert result.imbalanced_binary_columns == expected_columns


def test_should_not_classify_columns_with_more_than_two_values_as_binary(
    settings: config.Settings,
) -> None:
    """Verifica que las columnas con más de dos valores no se descarten."""
    data = pl.DataFrame(
        {
            "client_id": list(range(10)),
            "Target": [0, 1] * 5,
            "binary": [0] * 9 + [1],
            "balanced": [0, 1] * 5,
            "multi_value": list(range(10)),
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        imbalanced_binary_threshold=0.2,
        settings=settings,
    )

    assert result.imbalanced_binary_columns == ("binary",)


def test_should_exclude_target_from_imbalanced_binary_detection(
    settings: config.Settings,
) -> None:
    """Verifica que una target desbalanceada no se marque como columna redundante."""
    data = pl.DataFrame(
        {
            "client_id": list(range(20)),
            "Target": [0] * 19 + [1],
            "feature": [0, 1] * 10,
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        imbalanced_binary_threshold=0.1,
        settings=settings,
    )

    assert result.imbalanced_binary_columns == ()


def test_should_keep_first_feature_and_drop_later_correlated_columns(
    settings: config.Settings,
) -> None:
    """Verifica que se conserve la primera variable de un grupo correlacionado."""
    data = _correlated_frame()

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=0.8,
        settings=settings,
    )

    assert result.correlated_columns == ("feature_b", "feature_c")
    assert "feature_a" in result.uncorrelated_train.columns
    assert "feature_b" not in result.uncorrelated_train.columns
    assert "feature_c" not in result.uncorrelated_train.columns


def test_should_use_absolute_correlation_values_to_detect_redundancy(
    settings: config.Settings,
) -> None:
    """Verifica que las correlaciones negativas también indiquen redundancia."""
    data = _negative_correlated_frame()

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=0.8,
        settings=settings,
    )

    assert result.correlated_columns == ("feature_b",)


def test_should_keep_one_feature_when_all_are_correlated(
    settings: config.Settings,
) -> None:
    """Verifica que de un grupo correlacionado se conserven N-1 columnas."""
    data = _correlated_frame()

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=0.8,
        settings=settings,
    )

    assert result.correlated_columns == ("feature_b", "feature_c")
    assert result.uncorrelated_train.columns == ["client_id", "Target", "feature_a"]


def test_should_apply_correlation_threshold_strictly(
    settings: config.Settings,
) -> None:
    """Verifica que el umbral se aplique con comparación estricta."""
    data = _correlated_frame()
    correlations = _correlation_matrix(
        data,
        ("feature_a", "feature_b", "feature_c"),
    )
    correlation_ab = abs(correlations.get_column("feature_b").item(0))
    correlation_ac = abs(correlations.get_column("feature_c").item(0))

    at_ab = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=correlation_ab,
        settings=settings,
    )
    at_ac = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=correlation_ac,
        settings=settings,
    )

    assert at_ab.correlated_columns == ("feature_c",)
    assert at_ac.correlated_columns == ()


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(0.5, id="one-half"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.0, id="one"),
    ],
)
def test_should_raise_value_error_during_generate_for_invalid_imbalanced_threshold(
    settings: config.Settings,
    binary_frame: pl.DataFrame,
    threshold: float,
) -> None:
    """Verifica que la generación falle con un umbral binario inválido.

    Args:
        settings: Configuración global del proyecto.
        binary_frame: DataFrame con target y feature binarios.
        threshold: Umbral inválido para columnas binarias.
    """
    with pytest.raises(ValueError, match="Invalid threshold"):
        redundant_column_filter.RedundantColumnFilter.generate(
            train=binary_frame,
            test=binary_frame,
            imbalanced_binary_threshold=threshold,
            settings=settings,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.0, id="one"),
        pytest.param(-0.1, id="negative"),
    ],
)
def test_should_raise_value_error_during_generate_for_invalid_correlation_threshold(
    settings: config.Settings,
    threshold: float,
) -> None:
    """Verifica que la generación falle con un umbral de correlación inválido.

    Args:
        settings: Configuración global del proyecto.
        threshold: Umbral de correlación inválido.
    """
    df = pl.DataFrame(
        {
            "client_id": [1, 2, 3, 4],
            "Target": [0, 1, 0, 1],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [2.0, 4.0, 6.0, 8.0],
        }
    )

    with pytest.raises(ValueError, match="Invalid threshold"):
        redundant_column_filter.RedundantColumnFilter.generate(
            train=df,
            test=df,
            correlation_threshold=threshold,
            settings=settings,
        )


def test_should_count_values_of_imbalanced_binary_columns(
    redundant_filter: redundant_column_filter.RedundantColumnFilter,
) -> None:
    """Verifica la distribución de frecuencias de las columnas descartadas.

    Args:
        redundant_filter: Filtro con una columna binaria desbalanceada.
    """
    expected = pl.DataFrame(
        {
            "Column name": ["imbalanced", "imbalanced"],
            "value": ["0", "1"],
            "count": pl.Series([9, 1], dtype=pl.UInt32),
        }
    )

    result = redundant_filter.imbalanced_binary_counts

    polars.testing.assert_frame_equal(result, expected)


def test_should_filter_and_group_correlations_for_each_requested_column(
    settings: config.Settings,
) -> None:
    """Verifica el filtrado, orden y agrupación de las correlaciones públicas.

    Args:
        settings: Configuración global del proyecto.
    """
    data = _correlated_frame()
    column_filter = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=0.999,
        settings=settings,
    )

    result = column_filter.correlations_for_each(["feature_a", "feature_b"], 0.8)

    assert isinstance(result, types.MappingProxyType)
    assert tuple(result) == ("feature_a", "feature_b")
    assert result["feature_a"].columns == ["feature_name", "correlation"]
    assert result["feature_a"].get_column("feature_name").to_list() == [
        "feature_c",
        "feature_b",
    ]
    assert result["feature_b"].get_column("feature_name").to_list() == [
        "feature_a",
        "feature_c",
    ]
    assert "feature_a" not in result["feature_a"].get_column("feature_name").to_list()
    assert "feature_b" not in result["feature_b"].get_column("feature_name").to_list()
    for frame in result.values():
        absolute_values = frame.get_column("correlation").abs().to_list()
        assert absolute_values == sorted(absolute_values, reverse=True)
        assert (frame.get_column("correlation").abs() > 0.8).all()


def test_should_apply_correlation_threshold_strictly_in_correlations_for_each(
    settings: config.Settings,
) -> None:
    """Verifica la comparación estricta al filtrar correlaciones públicas.

    Args:
        settings: Configuración global del proyecto.
    """
    data = _correlated_frame()
    column_filter = redundant_column_filter.RedundantColumnFilter.generate(
        train=data,
        test=data,
        correlation_threshold=0.999,
        settings=settings,
    )
    correlations = _correlation_matrix(
        data,
        ("feature_a", "feature_b", "feature_c"),
    )
    correlation_ab = abs(correlations.get_column("feature_b").item(0))
    correlation_ac = abs(correlations.get_column("feature_c").item(0))

    at_ab = column_filter.correlations_for_each(["feature_a"], correlation_ab)
    at_ac = column_filter.correlations_for_each(["feature_a"], correlation_ac)

    assert at_ab["feature_a"].get_column("feature_name").to_list() == ["feature_c"]
    assert at_ac["feature_a"].is_empty()
