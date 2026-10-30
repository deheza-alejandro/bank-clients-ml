"""Tests unitarios para el módulo de agrupación de bines."""

import polars as pl
import polars.testing
import pytest

from bank_clients_ml import binning, config


@pytest.fixture
def bivariate_table() -> pl.DataFrame:
    """Construye una tabla bivariada canónica para los tests.

    La tabla permite comprobar el cálculo ponderado del porcentaje de target:
    el grupo de bines [1, 2] representa el 25 %, el grupo [4, 5] el 42.5 %
    y el bin residual [3] el 10 %.

    ## Returns:
        Tabla con las columnas de bin, mínimo, máximo, clientes y target.
    """
    return pl.DataFrame(
        {
            "Bin": [1, 2, 3, 4, 5],
            "Min": [0.0, 11.0, 21.0, 31.0, 41.0],
            "Max": [10.0, 20.0, 30.0, 40.0, 50.0],
            "Clients": [100, 300, 200, 300, 100],
            "Target": [10, 90, 20, 120, 50],
            "Target_pct": [10.0, 30.0, 10.0, 40.0, 50.0],
        }
    )


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


def test_should_return_same_binner_instance_when_merging_bins() -> None:
    """Verifica que `merge_bins` devuelva la misma instancia para encadenar."""
    binner = binning.Binner("feature", "Feature")

    result = binner.merge_bins(1)

    assert result is binner


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
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que se rechace una feature sin intervalos de bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
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
    bivariate_table: pl.DataFrame,
    start: int,
    end: int | None,
) -> None:
    """Verifica que se rechace un intervalo fuera de la cantidad de bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
        start: Número de bin inicial.
        end: Número de bin final, o None para un único bin.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(start, end)

    with pytest.raises(ValueError, match="invalid interval"):
        binner.build_encoding_expr(bivariate_table, settings)


def test_should_allow_interval_ending_at_last_bin(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que se permita un intervalo que termina en el último bin.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
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
    bivariate_table: pl.DataFrame,
    intervals: list[tuple[int, int]],
) -> None:
    """Verifica que se rechacen intervalos con bines solapados.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
        intervals: Pares de extremos de los intervalos a validar.
    """
    binner = binning.Binner("feature", "Feature")
    for start, end in intervals:
        binner.merge_bins(start, end)

    with pytest.raises(ValueError, match="overlapping bins"):
        binner.build_encoding_expr(bivariate_table, settings)


def test_should_allow_adjacent_intervals(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que dos intervalos contiguos no se consideren solapados.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(3, 4)

    result = encode(binner, bivariate_table, [5.0, 25.0, 45.0], settings)

    expected = pl.Series("feature", [25.0, 28.0, 50.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_weighted_target_pct_when_bins_have_different_sizes(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que el porcentaje de target se pondere por clientes.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)

    result = encode(binner, bivariate_table, [5.0, 15.0], settings)

    expected = pl.Series("feature", [25.0, 25.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_encode_values_on_both_interval_boundaries(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que se incluyan ambos extremos del intervalo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
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
    bivariate_table: pl.DataFrame,
    value: float,
) -> None:
    """Verifica que los valores externos reciban el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
        value: Valor externo a todos los intervalos.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [value], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_residual_target_pct_when_bin_is_not_in_any_interval(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que un bin no cubierto reciba el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [25.0], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_residual_target_pct_when_value_is_null(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que los valores nulos reciban el porcentaje residual.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2).merge_bins(4, 5)

    result = encode(binner, bivariate_table, [None], settings)

    expected = pl.Series("feature", [10.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result, expected)


def test_should_use_zero_as_residual_when_all_bins_are_covered(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que el residual sea cero cuando se cubren todos los bines.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
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


def test_should_alias_result_with_feature_name(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que la expresión resultante conserve el nombre de la feature.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)
    expression = binner.build_encoding_expr(bivariate_table, settings)

    result = pl.DataFrame({"feature": [5.0]}).select(expression)

    assert result.columns == ["feature"]


def test_should_replace_feature_column_when_expression_is_applied_with_columns(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
) -> None:
    """Verifica que la expresión reemplace la columna original sin duplicarla.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
    """
    binner = binning.Binner("feature", "Feature").merge_bins(1, 2)
    frame = pl.DataFrame({"feature": [5.0], "Target": [0]})

    result = frame.with_columns(binner.build_encoding_expr(bivariate_table, settings))

    assert result.columns == ["feature", "Target"]
    assert result.get_column("Target").to_list() == [0]
    assert result.get_column("feature").to_list() == [25.0]


@pytest.mark.parametrize("column", ["Min", "Max"])
def test_should_raise_type_error_when_bound_column_is_not_numeric(
    settings: config.Settings,
    bivariate_table: pl.DataFrame,
    column: str,
) -> None:
    """Verifica que se rechacen columnas de límites no numéricas.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        bivariate_table: Tabla bivariada de referencia.
        column: Nombre de la columna que se reemplaza por valores de texto.
    """
    table = bivariate_table.with_columns(
        pl.Series(column, ["not-a-number"] * bivariate_table.height)
    )
    binner = binning.Binner("feature", "Feature").merge_bins(1)

    with pytest.raises(TypeError, match="must be numeric"):
        binner.build_encoding_expr(table, settings)


def test_should_use_column_names_from_settings() -> None:
    """Verifica que se respeten los nombres de columna de la configuración."""
    settings = config.Settings(
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


def test_should_preserve_order_when_created_from_dict() -> None:
    """Verifica que se conserve el orden de inserción de los nombres."""
    mapping = binning.NameMapping.from_dict({"age": "Age", "income": "Income"})

    assert mapping.snake_case_names == ("age", "income")
    assert mapping.pretty_names == ("Age", "Income")


def test_should_return_pretty_name_when_key_exists() -> None:
    """Verifica que se devuelva el nombre descriptivo asociado a una clave."""
    mapping = binning.NameMapping.from_dict({"age": "Age"})

    assert mapping.get("age") == "Age"


def test_should_raise_key_error_when_key_is_missing() -> None:
    """Verifica que se lance KeyError cuando el nombre interno no existe."""
    mapping = binning.NameMapping.from_dict({"age": "Age"})

    with pytest.raises(KeyError):
        mapping.get("missing")


def test_should_build_mapping_from_dual_named_objects() -> None:
    """Verifica que se construya el mapeo a partir de objetos con doble nombre."""
    binners = [
        binning.Binner("age", "Age"),
        binning.Binner("income", "Income"),
    ]

    mapping = binning.NameMapping.from_dual_named(binners)

    assert mapping.snake_case_names == ("age", "income")
    assert mapping.pretty_names == ("Age", "Income")


def test_should_expose_pairs_when_items_is_called() -> None:
    """Verifica que se expongan los pares de nombres internos y descriptivos."""
    mapping = binning.NameMapping.from_dict({"age": "Age", "income": "Income"})

    assert tuple(mapping.items()) == (("age", "Age"), ("income", "Income"))


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


def test_should_not_reflect_mutations_of_source_dict() -> None:
    """Verifica que el mapeo no cambie si se modifica el diccionario de origen."""
    source = {"age": "Age"}
    mapping = binning.NameMapping.from_dict(source)

    source["age"] = "Age_modified"

    assert mapping.get("age") == "Age"
