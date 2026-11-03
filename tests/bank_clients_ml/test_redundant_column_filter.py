"""Tests unitarios para el filtrado de columnas redundantes."""

import types

import polars as pl
import polars.testing
import pytest
import pytest_mock

from bank_clients_ml import config, redundant_column_filter, training


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
def correlation_matrix() -> pl.DataFrame:
    """Devuelve una matriz de correlación con pares de distinta intensidad.

    Returns:
        Matriz cuadrada con columnas `a`, `b`, `c` y `d`.
    """
    return pl.DataFrame(
        {
            "a": [1.0, 0.9, -0.5, 0.2],
            "b": [0.9, 1.0, -0.95, 0.1],
            "c": [-0.5, -0.95, 1.0, 0.0],
            "d": [0.2, 0.1, 0.0, 1.0],
        }
    )


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


def _build_filter(
    settings: config.Settings,
    corr_df: pl.DataFrame,
    uncorrelated_train: pl.DataFrame | None = None,
) -> redundant_column_filter.RedundantColumnFilter:
    """Construye un filtro con una matriz de correlación controlada.

    Args:
        settings: Configuración global del proyecto.
        corr_df: Matriz de correlación que se desea analizar.
        uncorrelated_train: Datos de entrenamiento sin correlaciones. Si no se
            indica, se utiliza un DataFrame vacío.

    Returns:
        Instancia lista para consultar sus correlaciones.
    """
    empty = pl.DataFrame()
    return redundant_column_filter.RedundantColumnFilter(
        reduced_train=empty,
        correlated_train=empty,
        correlated_test=empty,
        uncorrelated_train=(
            uncorrelated_train if uncorrelated_train is not None else empty
        ),
        constant_cols=(),
        imbalanced_binary_columns=(),
        correlated_columns=(),
        imbalanced_binary_threshold=0.1,
        correlation_threshold=0.8,
        settings=settings,
        _corr_df=corr_df,
    )


def test_should_return_only_true_column_names() -> None:
    """Verifica que solo se devuelvan las columnas con valor verdadero."""
    condition_flags = pl.DataFrame({"alpha": [True], "beta": [False], "gamma": [True]})

    result = redundant_column_filter._get_true_column_names(condition_flags)

    assert result == ("alpha", "gamma")


def test_should_identify_constant_columns_of_numeric_and_text_types() -> None:
    """Verifica la detección de columnas con un único valor en cualquier tipo."""
    df = pl.DataFrame(
        {
            "constant_int": [4, 4, 4],
            "varying_int": [1, 2, 3],
            "constant_text": ["x", "x", "x"],
            "varying_text": ["x", "y", "y"],
        }
    )

    result = redundant_column_filter._get_constant_columns(df)

    assert result == ("constant_int", "constant_text")


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
    df = pl.DataFrame(
        {
            "Target": [0] * 10 + [1] * 10,
            "feature": [0] * zeros + [1] * ones,
        }
    )

    result = redundant_column_filter._get_imbalanced_binary_columns(df, 0.1, settings)

    expected_columns = ("feature",) if expected else ()
    assert result == expected_columns


def test_should_not_classify_columns_with_more_than_two_values_as_binary(
    settings: config.Settings,
) -> None:
    """Verifica que las columnas con más de dos valores no se descarten."""
    df = pl.DataFrame(
        {
            "Target": [0, 1] * 5,
            "binary": [0] * 9 + [1],
            "balanced": [0, 1] * 5,
            "multi_value": list(range(10)),
        }
    )

    result = redundant_column_filter._get_imbalanced_binary_columns(df, 0.2, settings)

    assert result == ("binary",)


def test_should_exclude_target_from_imbalanced_binary_detection(
    settings: config.Settings,
) -> None:
    """Verifica que una target desbalanceada no se marque como columna redundante."""
    df = pl.DataFrame(
        {
            "Target": [0] * 19 + [1],
            "feature": [0, 1] * 10,
        }
    )

    result = redundant_column_filter._get_imbalanced_binary_columns(df, 0.1, settings)

    assert result == ()


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(0.5, id="one-half"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.0, id="one"),
    ],
)
def test_should_raise_value_error_for_invalid_imbalanced_threshold(
    settings: config.Settings,
    threshold: float,
) -> None:
    """Verifica que se rechacen umbrales fuera del intervalo permitido.

    Args:
        settings: Configuración con el nombre de la columna target.
        threshold: Umbral inválido para columnas binarias.
    """
    df = pl.DataFrame({"Target": [0, 1], "feature": [0, 1]})

    with pytest.raises(ValueError, match="Invalid threshold"):
        redundant_column_filter._get_imbalanced_binary_columns(df, threshold, settings)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        pytest.param(0.8, ("b", "c"), id="two-above-threshold"),
        pytest.param(0.9, ("c",), id="strict-boundary-excludes-b"),
        pytest.param(0.95, (), id="strict-boundary-excludes-c"),
    ],
)
def test_should_return_columns_whose_previous_correlation_exceeds_threshold(
    correlation_matrix: pl.DataFrame,
    threshold: float,
    expected: tuple[str, ...],
) -> None:
    """Verifica la detección de columnas redundantes según su correlación previa.

    Args:
        correlation_matrix: Matriz con correlaciones de distinta intensidad.
        threshold: Umbral de correlación absoluta.
        expected: Columnas que deben descartarse.
    """
    result = redundant_column_filter._get_redundant_correlated_columns(
        correlation_matrix, threshold
    )

    assert result == expected


def test_should_use_absolute_correlation_values_to_detect_redundancy() -> None:
    """Verifica que las correlaciones negativas también indiquen redundancia."""
    corr_df = pl.DataFrame({"a": [1.0, -0.95], "b": [-0.95, 1.0]})

    result = redundant_column_filter._get_redundant_correlated_columns(corr_df, 0.9)

    assert result == ("b",)


def test_should_keep_one_feature_when_all_are_correlated() -> None:
    """Verifica que de un grupo correlacionado se conserven N-1 columnas."""
    corr_df = pl.DataFrame(
        {
            "a": [1.0, 0.9, 0.9],
            "b": [0.9, 1.0, 0.85],
            "c": [0.9, 0.85, 1.0],
        }
    )

    result = redundant_column_filter._get_redundant_correlated_columns(corr_df, 0.8)

    assert result == ("b", "c")


def test_should_ignore_correlations_from_lower_triangle() -> None:
    """Verifica que solo se evalúe cada columna contra las columnas previas."""
    corr_df = pl.DataFrame({"a": [1.0, 0.95], "b": [0.1, 1.0]})

    result = redundant_column_filter._get_redundant_correlated_columns(corr_df, 0.8)

    assert result == ()


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.0, id="one"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.5, id="greater-than-one"),
    ],
)
def test_should_raise_value_error_for_invalid_correlation_threshold(
    correlation_matrix: pl.DataFrame,
    threshold: float,
) -> None:
    """Verifica que se rechacen umbrales de correlación inválidos.

    Args:
        correlation_matrix: Matriz de correlación válida.
        threshold: Umbral de correlación inválido.
    """
    with pytest.raises(ValueError, match="Invalid threshold"):
        redundant_column_filter._get_redundant_correlated_columns(
            correlation_matrix, threshold
        )


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
            "constant": [7, 7, 7, 7],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    test = pl.DataFrame(
        {
            "client_id": ["e", "f"],
            "Target": [0, 1],
            "constant": [7, 7],
            "feature_a": [5.0, 6.0],
            "feature_b": [3.0, 2.0],
        }
    )

    result = redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        settings=settings,
    )

    assert result.constant_cols == ("constant",)
    polars.testing.assert_frame_equal(result.reduced_train, train.drop("constant"))
    polars.testing.assert_frame_equal(result.correlated_test, test.drop("constant"))


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
    assert redundant_filter._corr_df.columns == ["balanced", "feature_a", "feature_b"]


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

    assert result._corr_df.columns == ["feat_a", "feat_b"]
    assert result.correlated_columns == ("feat_b",)
    assert result.uncorrelated_train.columns == ["customer", "outcome", "feat_a"]


@pytest.mark.parametrize(
    "threshold",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(0.5, id="one-half"),
        pytest.param(-0.1, id="negative"),
    ],
)
def test_should_raise_value_error_during_generate_for_invalid_imbalanced_threshold(
    settings: config.Settings,
    threshold: float,
) -> None:
    """Verifica que la generación falle con un umbral binario inválido.

    Args:
        settings: Configuración global del proyecto.
        threshold: Umbral inválido para columnas binarias.
    """
    df = pl.DataFrame({"Target": [0, 1], "feature": [0, 1]})

    with pytest.raises(ValueError, match="Invalid threshold"):
        redundant_column_filter.RedundantColumnFilter.generate(
            train=df,
            test=df,
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


@pytest.mark.parametrize(
    ("threshold", "expected_names"),
    [
        pytest.param(0.4, ["b", "c"], id="two-above-threshold"),
        pytest.param(0.5, ["b"], id="strict-boundary-excludes-c"),
        pytest.param(0.9, [], id="strict-boundary-excludes-b"),
        pytest.param(0.0, ["b", "c", "d"], id="all-other-features"),
    ],
)
def test_should_filter_and_order_correlations_by_absolute_value(
    settings: config.Settings,
    correlation_matrix: pl.DataFrame,
    threshold: float,
    expected_names: list[str],
) -> None:
    """Verifica el filtrado y orden de las correlaciones de una columna.

    Args:
        settings: Configuración global del proyecto.
        correlation_matrix: Matriz de correlación controlada.
        threshold: Umbral mínimo de correlación absoluta.
        expected_names: Variables esperadas en orden de correlación absoluta.
    """
    redundant_filter = _build_filter(settings, correlation_matrix)

    result = redundant_filter._get_correlations_for("a", threshold)

    assert result.columns == ["feature_name", "correlation"]
    assert result.get_column("feature_name").to_list() == expected_names
    absolute_values = result.get_column("correlation").abs().to_list()
    assert absolute_values == sorted(absolute_values, reverse=True)
    assert "a" not in result.get_column("feature_name").to_list()


def test_should_group_correlations_for_each_requested_column(
    settings: config.Settings,
    correlation_matrix: pl.DataFrame,
) -> None:
    """Verifica que se devuelva un resultado independiente por columna solicitada.

    Args:
        settings: Configuración global del proyecto.
        correlation_matrix: Matriz de correlación controlada.
    """
    redundant_filter = _build_filter(settings, correlation_matrix)

    result = redundant_filter.correlations_for_each(["a", "b"], 0.4)

    expected_a = pl.DataFrame({"feature_name": ["b", "c"], "correlation": [0.9, -0.5]})
    expected_b = pl.DataFrame({"feature_name": ["c", "a"], "correlation": [-0.95, 0.9]})
    assert isinstance(result, types.MappingProxyType)
    assert tuple(result) == ("a", "b")
    polars.testing.assert_frame_equal(result["a"], expected_a)
    polars.testing.assert_frame_equal(result["b"], expected_b)


def test_should_train_grouped_model_and_return_top_feature_correlations(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    correlation_matrix: pl.DataFrame,
    train_data: pl.DataFrame,
) -> None:
    """Verifica el entrenamiento pesado y las correlaciones de las mejores variables.

    Args:
        mocker: Utilidad para sustituir los entrenadores de LightGBM.
        settings: Configuración global del proyecto.
        correlation_matrix: Matriz de correlación controlada.
        train_data: Conjunto de entrenamiento sin correlaciones.
    """
    group_trainer = mocker.Mock()
    group_trainer.get_top_grouped_features.return_value = ("a", "b")
    trainer = mocker.Mock()
    trainer.get_top_ranked_features.return_value = ("a", "c")
    group_fit = mocker.patch.object(
        training.GroupsLGBMTrainer,
        "fit_and_build",
        return_value=group_trainer,
    )
    main_fit = mocker.patch.object(
        training.LGBMTrainer,
        "fit_and_build",
        return_value=trainer,
    )
    redundant_filter = _build_filter(settings, correlation_matrix, train_data)

    result = redundant_filter.heavy_operations_and_correlations("top_features")

    result_group_trainer, result_trainer, correlations = result
    assert result_group_trainer is group_trainer
    assert result_trainer is trainer
    assert tuple(correlations) == ("a", "c")
    polars.testing.assert_frame_equal(
        correlations["a"],
        pl.DataFrame({"feature_name": ["b"], "correlation": [0.9]}),
    )
    polars.testing.assert_frame_equal(
        correlations["c"],
        pl.DataFrame({"feature_name": ["b"], "correlation": [-0.95]}),
    )
    group_fit.assert_called_once_with(train_data, settings=settings)
    group_trainer.plot_top_features.assert_called_once_with()
    group_trainer.get_top_grouped_features.assert_called_once_with()
    main_fit.assert_called_once_with(train_data, ("a", "b"), settings)
    trainer.plot_top_features.assert_called_once_with("top_features")
    trainer.get_top_ranked_features.assert_called_once_with()
