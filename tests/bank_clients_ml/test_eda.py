"""Tests unitarios para el módulo de análisis exploratorio de datos."""

from collections.abc import Callable
from datetime import date
from typing import cast

import polars as pl
import polars.testing
import pytest

from bank_clients_ml import config, eda

SHAPE_METRIC = "Shape of the DataFrame"
NULL_METRIC = "Columns with null, None or NaT"
NAN_METRIC = "Columns with NaN"
INF_METRIC = "Columns with inf"
NON_NUMERIC_METRIC = "Columns with non-numeric values"
JOIN_X_METRIC = "Columns ended with _x (pandas join)"
JOIN_Y_RIGHT_METRIC = "Columns ended with _y (pandas join) or _right (polars join)"


@pytest.fixture
def custom_analysis_settings() -> config.Settings:
    """Devuelve una configuración con un nombre de columna de análisis distinto.

    ## Returns:
        Configuración con la columna de resultados renombrada.
    """
    return config.Settings(analysis=config.AnalysisColumns(column_name="Variable"))


@pytest.fixture
def custom_dataset_settings() -> config.Settings:
    """Devuelve una configuración con un identificador de cliente distinto.

    ## Returns:
        Configuración con la columna identificadora renombrada.
    """
    return config.Settings(dataset=config.DatasetColumns(id="customer_id"))


def _metric_values(result: pl.DataFrame, metric: str) -> list[str]:
    """Devuelve los valores asociados a una métrica de inspección.

    ## Args:
        result: Resultado de `inspect_dataframe`.
        metric: Nombre de la métrica que se desea consultar.

    ## Returns:
        Lista de columnas asociadas a la métrica.
    """
    return result.filter(pl.col("metric") == metric).get_column("values").to_list()[0]


def test_should_return_one_row_per_original_column_with_statistics() -> None:
    """Verifica que el resumen describa cada columna en una fila independiente."""
    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": [10, 20, 30]})

    result = eda.describe(df)

    assert result.height == 2
    assert result.get_column("column").to_list() == ["a", "b"]


def test_should_normalize_null_count_to_int64_in_description() -> None:
    """Verifica que el conteo de nulos sea entero y refleje los valores ausentes."""
    df = pl.DataFrame({"a": [1.0, 2.0, None], "b": [10, 20, 30]})

    result = eda.describe(df)

    row_a = result.filter(pl.col("column") == "a").row(0, named=True)
    row_b = result.filter(pl.col("column") == "b").row(0, named=True)
    assert result.schema["null_count"] == pl.Int64
    assert row_a["count"] == 2
    assert row_a["null_count"] == 1
    assert row_b["null_count"] == 0


def test_should_report_shape_of_inspected_dataframe() -> None:
    """Verifica que la inspección informe filas y columnas del DataFrame."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    result = eda.inspect_dataframe(df)

    shape_total = (
        result.filter(pl.col("metric") == SHAPE_METRIC).get_column("total").to_list()
    )
    assert result.height == 7
    assert _metric_values(result, SHAPE_METRIC) == ["3 rows", "2 columns"]
    assert shape_total == [2]


def test_should_list_columns_with_null_values() -> None:
    """Verifica que se detecten columnas con nulos de distintos tipos."""
    df = pl.DataFrame(
        {
            "number": [1, None],
            "text": ["x", None],
            "complete": [1, 2],
        }
    )

    result = eda.inspect_dataframe(df)

    assert _metric_values(result, NULL_METRIC) == ["number", "text"]


def test_should_detect_nan_and_infinite_values_only_in_float_columns() -> None:
    """Verifica que NaN e infinitos se reporten en sus métricas respectivas."""
    df = pl.DataFrame(
        {
            "with_nan": [1.0, float("nan")],
            "with_inf": [1.0, float("inf")],
            "clean": [1.0, 2.0],
        }
    )

    result = eda.inspect_dataframe(df)

    assert _metric_values(result, NAN_METRIC) == ["with_nan"]
    assert _metric_values(result, INF_METRIC) == ["with_inf"]


def test_should_list_text_and_date_columns_as_non_numeric() -> None:
    """Verifica que las columnas no numéricas se reporten en la inspección."""
    df = pl.DataFrame(
        {
            "name": ["a", "b"],
            "signup": [date(2024, 1, 1), date(2024, 2, 1)],
            "amount": [1.5, 2.5],
            "count": [1, 2],
        }
    )

    result = eda.inspect_dataframe(df)

    assert _metric_values(result, NON_NUMERIC_METRIC) == ["name", "signup"]


def test_should_detect_columns_with_join_suffixes() -> None:
    """Verifica que se detecten restos de joins con sufijos conocidos."""
    df = pl.DataFrame(
        {
            "customer_x": [1],
            "customer_y": [2],
            "account_right": [3],
            "normal": [4],
        }
    )

    result = eda.inspect_dataframe(df)

    assert _metric_values(result, JOIN_X_METRIC) == ["customer_x"]
    assert _metric_values(result, JOIN_Y_RIGHT_METRIC) == [
        "customer_y",
        "account_right",
    ]


def test_should_inspect_empty_dataframe_without_errors() -> None:
    """Verifica que un DataFrame vacío reporte cero filas y columnas."""
    df = pl.DataFrame()

    result = eda.inspect_dataframe(df)

    assert result.height == 7
    assert _metric_values(result, SHAPE_METRIC) == ["0 rows", "0 columns"]
    assert _metric_values(result, NULL_METRIC) == []


@pytest.mark.parametrize(
    ("condition", "expected_count"),
    [
        pytest.param("<", 1, id="less-than"),
        pytest.param(">", 2, id="greater-than"),
        pytest.param("<=", 2, id="less-or-equal"),
        pytest.param(">=", 3, id="greater-or-equal"),
        pytest.param("==", 1, id="equal"),
        pytest.param("!=", 3, id="not-equal"),
    ],
)
def test_should_count_rows_for_each_comparison_condition(
    settings: config.Settings,
    condition: eda.ConditionSymbol,
    expected_count: int,
) -> None:
    """Verifica el conteo de filas para cada símbolo de comparación.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
        condition: Símbolo de comparación aplicado.
        expected_count: Cantidad de filas que cumplen la condición.
    """
    df = pl.DataFrame({"value": [1, 2, 3, 4]})

    result = eda.count_row_matches(df, condition=condition, threshold=2)

    output_column = f"Number of rows {condition} 2"
    assert result.columns == [settings.analysis.column_name, output_column]
    assert result.get_column(output_column).to_list() == [expected_count]


def test_should_count_matches_per_column_and_format_float_threshold(
    settings: config.Settings,
) -> None:
    """Verifica el conteo por columna y el formato del umbral decimal.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"low": [1.0, 2.0, 3.0], "high": [0.0, 5.0, 0.0]})

    result = eda.count_row_matches(df, condition=">", threshold=1.0)

    output_column = "Number of rows > 1.0"
    assert result.columns == [settings.analysis.column_name, output_column]
    assert result.get_column(output_column).to_list() == [2, 1]


def test_should_not_count_null_values_as_matching_rows(
    settings: config.Settings,
) -> None:
    """Verifica que los valores nulos no se cuenten como coincidencias.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"value": [1, None, 0, 3]})

    result = eda.count_row_matches(df, condition="<", threshold=1, settings=settings)

    output_column = "Number of rows < 1"
    assert result.get_column(output_column).to_list() == [1]


def test_should_use_custom_analysis_column_name_when_counting_matches(
    custom_analysis_settings: config.Settings,
) -> None:
    """Verifica que se respete el nombre configurado para la columna de resultados.

    ## Args:
        custom_analysis_settings: Configuración con la columna renombrada.
    """
    df = pl.DataFrame({"value": [1, 2, 3]})

    result = eda.count_row_matches(
        df,
        condition=">",
        threshold=1,
        settings=custom_analysis_settings,
    )

    assert result.columns == ["Variable", "Number of rows > 1"]


@pytest.mark.parametrize("condition", ["=<", "~", "between"])
@pytest.mark.parametrize(
    "function",
    [eda.count_row_matches, eda.filter_columns_by_cardinality],
    ids=["count-row-matches", "filter-columns-by-cardinality"],
)
def test_should_raise_value_error_when_condition_is_invalid(
    function: Callable[..., pl.DataFrame],
    condition: str,
) -> None:
    """Verifica que un símbolo de comparación inválido genere un error.

    ## Args:
        function: Función pública que valida el símbolo.
        condition: Símbolo de comparación inválido.
    """
    df = pl.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(ValueError, match="Invalid condition"):
        function(df, condition=cast(eda.ConditionSymbol, condition))


def test_should_keep_columns_with_more_unique_values_than_default_threshold(
    settings: config.Settings,
) -> None:
    """Verifica el filtro por cardinalidad con el umbral predeterminado.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame(
        {
            "few": [index % 2 for index in range(12)],
            "many": list(range(12)),
        }
    )

    result = eda.filter_columns_by_cardinality(df)

    assert result.columns == [settings.analysis.column_name, "unique_values"]
    assert result.get_column(settings.analysis.column_name).to_list() == ["many"]
    assert result.get_column("unique_values").to_list() == [12]


@pytest.mark.parametrize(
    ("condition", "expected_columns"),
    [
        pytest.param(">", ["three"], id="greater-than"),
        pytest.param(">=", ["two", "three"], id="greater-or-equal"),
        pytest.param("<", ["one"], id="less-than"),
        pytest.param("<=", ["one", "two"], id="less-or-equal"),
        pytest.param("==", ["two"], id="equal"),
        pytest.param("!=", ["one", "three"], id="not-equal"),
    ],
)
def test_should_filter_columns_using_requested_cardinality_comparison(
    settings: config.Settings,
    condition: eda.ConditionSymbol,
    expected_columns: list[str],
) -> None:
    """Verifica que se aplique el símbolo de comparación sobre la cardinalidad.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
        condition: Símbolo de comparación aplicado a los valores únicos.
        expected_columns: Columnas que deben permanecer en el resultado.
    """
    df = pl.DataFrame(
        {
            "one": [1, 1, 1],
            "two": [1, 2, 1],
            "three": [1, 2, 3],
        }
    )

    result = eda.filter_columns_by_cardinality(df, condition=condition, threshold=2)

    assert (
        result.get_column(settings.analysis.column_name).to_list() == expected_columns
    )


def test_should_use_custom_analysis_column_name_when_filtering_cardinality(
    custom_analysis_settings: config.Settings,
) -> None:
    """Verifica el nombre configurado al filtrar columnas por cardinalidad.

    ## Args:
        custom_analysis_settings: Configuración con la columna renombrada.
    """
    df = pl.DataFrame({"value": [1, 2, 3]})

    result = eda.filter_columns_by_cardinality(
        df,
        condition=">=",
        threshold=3,
        settings=custom_analysis_settings,
    )

    assert result.columns == ["Variable", "unique_values"]


def test_should_return_empty_cardinality_result_when_no_column_matches(
    settings: config.Settings,
) -> None:
    """Verifica el resultado vacío cuando ninguna columna cumple la condición.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"value": [1, 1, 1]})

    result = eda.filter_columns_by_cardinality(df, condition=">", threshold=10)

    assert result.height == 0
    assert result.columns == [settings.analysis.column_name, "unique_values"]


def test_should_count_values_of_low_cardinality_columns(
    settings: config.Settings,
) -> None:
    """Verifica el conteo de ocurrencias de columnas con baja cardinalidad.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame(
        {
            "color": ["red", "red", "red", "blue", "blue", "green"],
            "size": ["L", "M", "S", "M", "S", "S"],
            "client_id": [1, 2, 3, 4, 5, 6],
        }
    )

    result = eda.low_cardinality_value_counts(df, max_unique_values=3)

    expected = pl.DataFrame(
        {
            settings.analysis.column_name: [
                "color",
                "color",
                "color",
                "size",
                "size",
                "size",
            ],
            "value": ["red", "blue", "green", "S", "M", "L"],
            "count": [3, 2, 1, 3, 2, 1],
        },
        schema_overrides={"count": pl.UInt32},
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_cast_low_cardinality_values_to_string() -> None:
    """Verifica que los valores numéricos se conviertan a texto."""
    df = pl.DataFrame({"flag": [0, 1, 0]})

    result = eda.low_cardinality_value_counts(df, max_unique_values=2)

    assert result.get_column("value").to_list() == ["0", "1"]
    assert result.get_column("count").to_list() == [2, 1]
    assert result.schema["value"] == pl.String


def test_should_return_empty_value_counts_with_expected_schema_when_no_column_is_low_cardinality(
    settings: config.Settings,
) -> None:
    """Verifica el esquema vacío cuando ninguna columna cumple la cardinalidad.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"client_id": list(range(5))})

    result = eda.low_cardinality_value_counts(df, max_unique_values=2)

    expected = pl.DataFrame(
        schema={
            settings.analysis.column_name: pl.String,
            "value": pl.String,
            "count": pl.UInt32,
        }
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_use_custom_analysis_column_name_in_low_cardinality_value_counts(
    custom_analysis_settings: config.Settings,
) -> None:
    """Verifica el nombre configurado en el conteo de valores.

    ## Args:
        custom_analysis_settings: Configuración con la columna renombrada.
    """
    df = pl.DataFrame({"color": ["red", "blue", "red"]})

    result = eda.low_cardinality_value_counts(
        df,
        settings=custom_analysis_settings,
    )

    assert result.columns == ["Variable", "value", "count"]


def test_should_count_clients_grouped_by_number_of_registered_months(
    settings: config.Settings,
) -> None:
    """Verifica el resumen de clientes según la cantidad de meses registrados.

    ## Args:
        settings: Configuración con el identificador de cliente.
    """
    df = pl.DataFrame(
        {
            settings.dataset.id: [
                "a",
                "a",
                "b",
                "b",
                "c",
                "c",
                "d",
                "e",
                "f",
                "f",
                "f",
            ]
        }
    )

    result = eda.get_clients_by_months(df)

    expected = pl.DataFrame(
        {
            "month_count": [2, 1, 3],
            "clients_count": [3, 2, 1],
        },
        schema_overrides={"month_count": pl.UInt32, "clients_count": pl.UInt32},
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_use_custom_client_identifier_when_summarizing_months(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se use el identificador configurado para agrupar clientes.

    ## Args:
        custom_dataset_settings: Configuración con el identificador renombrado.
    """
    df = pl.DataFrame({"customer_id": ["a", "a", "b", "b", "c"]})

    result = eda.get_clients_by_months(df, settings=custom_dataset_settings)

    assert result.get_column("month_count").to_list() == [2, 1]
    assert result.get_column("clients_count").to_list() == [2, 1]


def test_should_return_numeric_columns_whose_minimum_is_inside_open_interval(
    settings: config.Settings,
) -> None:
    """Verifica la detección de mínimos dentro del intervalo abierto.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame(
        {
            "residual": [-0.5, 0.2],
            "positive": [0.5, 2.0],
            "zero_min": [0.0, 1.0],
            "low_bound": [-1.0, 0.5],
            "high_bound": [1.0, 2.0],
            "below": [-2.0, 0.5],
            "above": [1.5, 2.0],
            "label": ["a", "b"],
        }
    )

    result = eda.mins_in_range(df)

    expected = pl.DataFrame(
        {
            settings.analysis.column_name: ["residual", "positive"],
            "minimum": [-0.5, 0.5],
        }
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_apply_custom_bounds_to_minimum_detection(
    settings: config.Settings,
) -> None:
    """Verifica que se respeten los límites personalizados del intervalo.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame(
        {
            "far_negative": [-1.5, 0.0],
            "near_negative": [-0.5, 0.0],
            "positive": [0.5, 1.0],
        }
    )

    result = eda.mins_in_range(df, low=-2.0, high=2.0)

    assert result.get_column(settings.analysis.column_name).to_list() == [
        "far_negative",
        "near_negative",
        "positive",
    ]


def test_should_return_empty_minimum_result_when_no_column_matches(
    settings: config.Settings,
) -> None:
    """Verifica el resultado vacío cuando ningún mínimo está en el intervalo.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"constant": [2.0, 3.0]})

    result = eda.mins_in_range(df)

    expected = pl.DataFrame(
        schema={
            settings.analysis.column_name: pl.String,
            "minimum": pl.Float64,
        }
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_use_custom_analysis_column_name_in_minimum_result(
    custom_analysis_settings: config.Settings,
) -> None:
    """Verifica el nombre configurado en la detección de mínimos.

    ## Args:
        custom_analysis_settings: Configuración con la columna renombrada.
    """
    df = pl.DataFrame({"value": [0.5, 1.0]})

    result = eda.mins_in_range(df, settings=custom_analysis_settings)

    assert result.columns == ["Variable", "minimum"]


def test_should_count_zeros_in_numeric_columns(settings: config.Settings) -> None:
    """Verifica el conteo de ceros en columnas numéricas.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame(
        {
            "twice": [0, 1, 0, None],
            "none": [1, 2, 3, 4],
            "once": [None, 0, 1, 2],
            "label": ["0", "1", "0", "x"],
        }
    )

    result = eda.columns_with_zeros(df)

    expected = pl.DataFrame(
        {
            settings.analysis.column_name: ["twice", "once"],
            "zeros_quantity": [2, 1],
        },
        schema_overrides={"zeros_quantity": pl.UInt32},
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_return_empty_zero_count_when_no_numeric_column_has_zeros(
    settings: config.Settings,
) -> None:
    """Verifica el resultado vacío cuando no hay ceros numéricos.

    ## Args:
        settings: Configuración con el nombre de la columna de resultados.
    """
    df = pl.DataFrame({"positive": [1, 2], "label": ["0", "0"]})

    result = eda.columns_with_zeros(df)

    expected = pl.DataFrame(
        schema={
            settings.analysis.column_name: pl.String,
            "zeros_quantity": pl.UInt32,
        }
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_use_custom_analysis_column_name_in_zero_count(
    custom_analysis_settings: config.Settings,
) -> None:
    """Verifica el nombre configurado en el conteo de ceros.

    ## Args:
        custom_analysis_settings: Configuración con la columna renombrada.
    """
    df = pl.DataFrame({"value": [0, 1]})

    result = eda.columns_with_zeros(df, settings=custom_analysis_settings)

    assert result.columns == ["Variable", "zeros_quantity"]


def test_should_keep_rows_where_all_columns_are_nonzero() -> None:
    """Verifica que solo se conserven filas sin ceros en ninguna columna."""
    df = pl.DataFrame({"a": [1, 0, 2, 3], "b": [3, 4, 0, 5]})

    result = eda.filter_nonzero(df)

    expected = pl.DataFrame({"a": [1, 3], "b": [3, 5]})
    polars.testing.assert_frame_equal(result, expected)


def test_should_keep_negative_values_when_filtering_nonzero_rows() -> None:
    """Verifica que los valores negativos se consideren distintos de cero."""
    df = pl.DataFrame({"value": [-1, 0, 2]})

    result = eda.filter_nonzero(df)

    assert result.get_column("value").to_list() == [-1, 2]
