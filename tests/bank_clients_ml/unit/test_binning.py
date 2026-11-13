"""Tests unitarios para el módulo de agrupación de bines."""

import polars as pl
import polars.testing
import pytest

import tests.builders.unit.binning
import tests.support.settings
from bank_clients_ml import binning, config


def encode(
    binner: binning.Binner,
    bivariate_table: pl.DataFrame,
    values: list[float | None],
    settings: config.Settings,
) -> pl.Series:
    """Aplica la expresión de codificación a una columna de valores.

    ## Args:
        binner: Asociación entre la feature y los intervalos de bines.
        bivariate_table: Tabla bivariada de referencia.
        values: Valores de la feature que se desean codificar.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        Serie codificada con el nombre de la feature.
    """
    feature = binner.snake_case_name
    frame = pl.DataFrame(
        {feature: values},
        schema={feature: pl.Float64},
    )
    return frame.with_columns(
        binner.build_encoding_expr(bivariate_table, settings)
    ).get_column(feature)


def test_should_create_single_bin_interval_when_end_is_omitted() -> None:
    """Verifica que se cree un intervalo de un solo bin si se omite el fin."""
    binner = binning.Binner("feature", "Feature").merge_bins(3)

    assert [interval.range for interval in binner.bin_intervals] == [range(3, 4)]


def test_should_create_inclusive_interval_when_end_is_provided() -> None:
    """Verifica que el intervalo incluya ambos extremos cuando se indica el fin."""
    binner = binning.Binner("feature", "Feature").merge_bins(3, 5)

    assert [interval.range for interval in binner.bin_intervals] == [range(3, 6)]


def test_should_accumulate_multiple_intervals() -> None:
    """Verifica que se conserven todos los intervalos agregados en orden."""
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(3, 4)

    assert [interval.range for interval in binner.bin_intervals] == [
        range(1, 3),
        range(3, 5),
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(0, None, id="start-zero"),
        pytest.param(-1, None, id="start-negative"),
        pytest.param(1, 0, id="end-zero"),
        pytest.param(1, -5, id="end-negative"),
    ],
)
def test_should_raise_value_error_when_any_bound_is_less_than_one(
    start: int,
    end: int | None,
) -> None:
    """Verifica que se rechacen intervalos con extremos menores que uno.

    ## Args:
        start: Número de bin inicial.
        end: Número de bin final, o None para un único bin.
    """
    with pytest.raises(ValueError, match="must be >= 1"):
        binning.Binner("feature", "Feature").merge_bins(start, end)


def test_should_raise_value_error_when_start_is_greater_than_end() -> None:
    """Verifica que se rechace un intervalo cuyo inicio supera al fin."""
    with pytest.raises(ValueError, match="must be <= 'end'"):
        binning.Binner("feature", "Feature").merge_bins(3, 2)


def test_should_raise_value_error_when_no_bin_intervals_are_defined(
    settings: config.Settings,
) -> None:
    """Verifica que se rechace una feature sin intervalos de bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature")

    with pytest.raises(ValueError, match="must not be empty"):
        binner.build_encoding_expr(bivariate_table, settings)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(5, 6, id="end-exceeds"),
        pytest.param(6, None, id="start-and-end-exceed"),
    ],
)
def test_should_raise_value_error_when_interval_exceeds_available_bins(
    settings: config.Settings,
    start: int,
    end: int | None,
) -> None:
    """Verifica que se rechace un intervalo fuera de la cantidad de bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        start: Número de bin inicial.
        end: Número de bin final, o None para un único bin.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(start, end)

    with pytest.raises(ValueError, match="invalid interval"):
        binner.build_encoding_expr(bivariate_table, settings)


def test_should_allow_interval_ending_at_last_bin(
    settings: config.Settings,
) -> None:
    """Verifica que se permita un intervalo que termina en el último bin.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(5, 5)

    result = encode(binner, bivariate_table, [45.0], settings)

    expected = pl.Series("feature", [50.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "intervals",
    [
        pytest.param([(1, 3), (2, 4)], id="partial-overlap"),
        pytest.param([(1, 4), (2, 2)], id="contained-interval"),
        pytest.param([(1, 1), (1, 1)], id="duplicated-interval"),
    ],
)
def test_should_raise_value_error_when_intervals_overlap(
    settings: config.Settings,
    intervals: list[tuple[int, int]],
) -> None:
    """Verifica que se rechacen intervalos con bines solapados.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        intervals: Pares de extremos de los intervalos a validar.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature")
    for start, end in intervals:
        binner.merge_bins(start, end)

    with pytest.raises(ValueError, match="overlapping bins"):
        binner.build_encoding_expr(bivariate_table, settings)


def test_should_allow_adjacent_intervals(
    settings: config.Settings,
) -> None:
    """Verifica que dos intervalos contiguos no se consideren solapados.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(3, 4)

    result = encode(binner, bivariate_table, [5.0, 25.0, 45.0], settings)

    expected = pl.Series("feature", [25.0, 28.0, 50.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_weighted_target_pct_when_bins_have_different_sizes(
    settings: config.Settings,
) -> None:
    """Verifica que el porcentaje de target se pondere por clientes.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)

    result = encode(binner, bivariate_table, [5.0, 15.0], settings)

    expected = pl.Series("feature", [25.0, 25.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_encode_values_on_both_interval_boundaries(
    settings: config.Settings,
) -> None:
    """Verifica que se incluyan ambos extremos del intervalo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)

    result = encode(binner, bivariate_table, [0.0, 20.0], settings)

    expected = pl.Series("feature", [25.0, 25.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_include_gap_values_when_bins_are_not_contiguous(
    settings: config.Settings,
) -> None:
    """Verifica que un valor entre bines del mismo grupo reciba su porcentaje.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    table = pl.DataFrame(
        {
            "Bin": [1, 2, 3],
            "Min": [0.0, 21.0, 31.0],
            "Max": [10.0, 30.0, 40.0],
            "Clients": [100, 100, 100],
            "Target": [10, 20, 30],
        }
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)

    result = encode(binner, table, [0.0, 15.0, 30.0, 35.0], settings)

    expected = pl.Series("feature", [15.0, 15.0, 15.0, 30.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("value", [-1.0, 51.0])
def test_should_use_residual_target_pct_when_value_is_outside_all_intervals(
    settings: config.Settings,
    value: float,
) -> None:
    """Verifica que los valores externos reciban el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        value: Valor externo a todos los intervalos.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [value], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_residual_target_pct_when_bin_is_not_in_any_interval(
    settings: config.Settings,
) -> None:
    """Verifica que un bin no cubierto reciba el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [25.0], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_residual_target_pct_when_value_is_null(
    settings: config.Settings,
) -> None:
    """Verifica que los valores nulos reciban el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [None], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_zero_as_residual_when_all_bins_are_covered(
    settings: config.Settings,
) -> None:
    """Verifica que el residual sea cero cuando se cubren todos los bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    binner = binning.Binner("feature", "Feature").merge_bins(1, 5)

    result = encode(binner, bivariate_table, [25.0, -1.0, 51.0, None], settings)

    expected = pl.Series("feature", [29.0, 0.0, 0.0, 0.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_prefer_first_interval_when_intervals_share_boundary(
    settings: config.Settings,
) -> None:
    """Verifica que prevalezca el primer intervalo ante una frontera compartida.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    table = pl.DataFrame(
        {
            "Bin": [1, 2],
            "Min": [0.0, 10.0],
            "Max": [10.0, 20.0],
            "Clients": [100, 100],
            "Target": [10, 20],
        }
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1).merge_bins(2)

    result = encode(binner, table, [0.0, 10.0, 20.0], settings)

    expected = pl.Series("feature", [10.0, 10.0, 20.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_ignore_interval_with_null_bounds_and_use_residual_target_pct(
    settings: config.Settings,
) -> None:
    """Verifica que un intervalo sin límites no coincida con ningún valor.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    table = pl.DataFrame(
        {
            "Bin": [1, 2, 3],
            "Min": [0.0, None, 21.0],
            "Max": [10.0, None, 30.0],
            "Clients": [100, 100, 100],
            "Target": [10, 20, 30],
        }
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1).merge_bins(2)

    result = encode(binner, table, [5.0, 25.0, None], settings)

    expected = pl.Series("feature", [10.0, 30.0, 30.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("column", ["Min", "Max"])
def test_should_raise_type_error_when_bound_column_is_not_numeric(
    settings: config.Settings,
    column: str,
) -> None:
    """Verifica que se rechacen columnas de límites no numéricas.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        column: Nombre de la columna que se reemplaza por valores de texto.
    """
    bivariate_table = tests.builders.unit.binning.build_bivariate_table()
    table = bivariate_table.with_columns(
        pl.Series(column, ["not-a-number"] * bivariate_table.height)
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1)

    with pytest.raises(TypeError, match="must be numeric"):
        binner.build_encoding_expr(table, settings)


def test_should_use_column_names_from_settings() -> None:
    """Verifica que se respeten los nombres de columna de la configuración."""
    settings = tests.support.settings.build_settings_for_tests(
        dataset=config.DatasetColumns(target="target"),
        analysis=config.AnalysisColumns(
            bin="bucket",
            min="lower",
            max="upper",
            clients="clients",
        ),
    )
    table = pl.DataFrame(
        {
            "bucket": [1, 2],
            "lower": [0.0, 11.0],
            "upper": [10.0, 20.0],
            "clients": [100, 100],
            "target": [10, 20],
        }
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)

    result = encode(binner, table, [5.0], settings)

    expected = pl.Series("feature", [15.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_replace_mapped_values_and_keep_unmapped_values() -> None:
    """Verifica que se reemplacen los valores mapeados y se conserven los demás."""
    mapping = binning.NameMapping.from_dict({"age": "Age"})
    frame = pl.DataFrame(
        {"feature": ["age", "other"]},
        schema={"feature": pl.String},
    )

    result = frame.select(mapping.to_expr("feature"))

    expected = pl.DataFrame(
        {"feature": ["Age", "other"]},
        schema={"feature": pl.String},
    )
    polars.testing.assert_frame_equal(result, expected)


def test_should_keep_null_values_when_applying_mapping() -> None:
    """Verifica que los valores nulos se conserven al aplicar el mapeo."""
    mapping = binning.NameMapping.from_dict({"age": "Age"})
    frame = pl.DataFrame(
        {"feature": ["age", None]},
        schema={"feature": pl.String},
    )

    result = frame.select(mapping.to_expr("feature"))

    expected = pl.DataFrame(
        {"feature": ["Age", None]},
        schema={"feature": pl.String},
    )
    polars.testing.assert_frame_equal(result, expected)
