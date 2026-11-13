"""Tests unitarios para la construcción de tablas de análisis bivariado."""

import polars as pl
import polars.testing
import pytest

import tests.builders.unit.binning
import tests.support.settings
from bank_clients_ml import binning, bivariate_table, config


def test_should_group_by_original_values_and_compute_target_pct_when_cardinality_is_low(
    settings: config.Settings,
) -> None:
    """Verifica el agrupamiento por valores originales y el porcentaje de target.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    age_mapping = tests.builders.unit.binning.build_age_mapping()
    df = pl.DataFrame(
        {
            "age": [40, 20, 20, 30, 40, 40],
            "Target": [0, 1, 0, 1, 1, 0],
        }
    )

    result = bivariate_table.build_bivariate_tables(
        df,
        age_mapping,
        max_bins_quantity=10,
        settings=settings,
    )

    expected = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": pl.Series([20, 30, 40], dtype=pl.Int64),
            "Max": pl.Series([20, 30, 40], dtype=pl.Int64),
            "Clients": pl.Series([2, 1, 3], dtype=pl.UInt32),
            "Target": pl.Series([1, 1, 1], dtype=pl.Int64),
            "Target_pct": pl.Series(
                [50.0, 100.0, 1 / 3 * 100],
                dtype=pl.Float64,
            ),
        }
    )
    assert len(result) == 1
    assert result[0].snake_case_name == "age"
    assert result[0].pretty_name == "Age"
    polars.testing.assert_frame_equal(result[0].table, expected)


def test_should_return_tables_in_mapping_order_with_pretty_names(
    settings: config.Settings,
) -> None:
    """Verifica que las tablas respeten el orden y los nombres del mapeo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    df = pl.DataFrame(
        {
            "age": [10, 20, 10],
            "income": [100, 200, 300],
            "Target": [0, 1, 1],
        }
    )
    name_mapping = binning.NameMapping.from_dict({"income": "Income", "age": "Age"})

    result = bivariate_table.build_bivariate_tables(
        df,
        name_mapping,
        max_bins_quantity=10,
        settings=settings,
    )

    expected_income = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": pl.Series([100, 200, 300], dtype=pl.Int64),
            "Max": pl.Series([100, 200, 300], dtype=pl.Int64),
            "Clients": pl.Series([1, 1, 1], dtype=pl.UInt32),
            "Target": pl.Series([0, 1, 1], dtype=pl.Int64),
            "Target_pct": pl.Series([0.0, 100.0, 100.0], dtype=pl.Float64),
        }
    )
    assert isinstance(result, tuple)
    assert [
        (bivariate.snake_case_name, bivariate.pretty_name) for bivariate in result
    ] == [("income", "Income"), ("age", "Age")]
    polars.testing.assert_frame_equal(result[0].table, expected_income)


def test_should_bin_values_when_cardinality_exceeds_maximum_bins(
    settings: config.Settings,
) -> None:
    """Verifica que se generen bines cuando la cardinalidad supera el máximo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    age_mapping = tests.builders.unit.binning.build_age_mapping()
    df = pl.DataFrame(
        {
            "age": list(range(8)),
            "Target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    result = bivariate_table.build_bivariate_tables(
        df,
        age_mapping,
        max_bins_quantity=4,
        settings=settings,
    )

    table = result[0].table
    clients = table.get_column("Clients").to_list()
    targets = table.get_column("Target").to_list()
    expected_target_pct = [
        target / client * 100 for target, client in zip(targets, clients, strict=True)
    ]
    assert 1 < table.height <= 4
    assert table.height < df.height
    assert sum(clients) == df.height
    assert sum(targets) == df.get_column("Target").sum()
    assert table.get_column("Bin").to_list() == list(range(1, table.height + 1))
    assert table.get_column("Target_pct").to_list() == pytest.approx(
        expected_target_pct
    )


def test_should_preserve_original_values_when_cardinality_equals_maximum_bins(
    settings: config.Settings,
) -> None:
    """Verifica que no se generen bines si la cardinalidad iguala el máximo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    age_mapping = tests.builders.unit.binning.build_age_mapping()
    df = pl.DataFrame(
        {
            "age": [10] * 8 + [20, 30],
            "Target": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        }
    )

    result = bivariate_table.build_bivariate_tables(
        df,
        age_mapping,
        max_bins_quantity=3,
        settings=settings,
    )

    expected = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": pl.Series([10, 20, 30], dtype=pl.Int64),
            "Max": pl.Series([10, 20, 30], dtype=pl.Int64),
            "Clients": pl.Series([8, 1, 1], dtype=pl.UInt32),
            "Target": pl.Series([1, 1, 0], dtype=pl.Int64),
            "Target_pct": pl.Series([12.5, 100.0, 0.0], dtype=pl.Float64),
        }
    )
    polars.testing.assert_frame_equal(result[0].table, expected)


def test_should_place_null_group_last_with_bin_number_after_non_null_groups(
    settings: config.Settings,
) -> None:
    """Verifica que el grupo de nulos quede al final y conserve sus métricas.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    age_mapping = tests.builders.unit.binning.build_age_mapping()
    df = pl.DataFrame(
        {
            "age": [20, None, 10, None],
            "Target": [1, 0, 1, 1],
        }
    )

    result = bivariate_table.build_bivariate_tables(
        df,
        age_mapping,
        max_bins_quantity=10,
        settings=settings,
    )

    expected = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": pl.Series([10, 20, None], dtype=pl.Int64),
            "Max": pl.Series([10, 20, None], dtype=pl.Int64),
            "Clients": pl.Series([1, 1, 2], dtype=pl.UInt32),
            "Target": pl.Series([1, 1, 1], dtype=pl.Int64),
            "Target_pct": pl.Series([100.0, 100.0, 50.0], dtype=pl.Float64),
        }
    )
    polars.testing.assert_frame_equal(result[0].table, expected)


def test_should_use_custom_column_names_from_settings() -> None:
    """Verifica que las columnas de salida respeten la configuración."""
    age_mapping = tests.builders.unit.binning.build_age_mapping()
    settings = tests.support.settings.build_settings_for_tests(
        dataset=config.DatasetColumns(target="target"),
        analysis=config.AnalysisColumns(
            bin="bucket",
            min="lower",
            max="upper",
            clients="clients",
            target_pct="target_pct",
        ),
    )
    df = pl.DataFrame(
        {
            "age": [10, 10, 20],
            "target": [1, 0, 1],
        }
    )

    result = bivariate_table.build_bivariate_tables(
        df,
        age_mapping,
        max_bins_quantity=10,
        settings=settings,
    )

    expected = pl.DataFrame(
        {
            "bucket": pl.Series([1, 2], dtype=pl.Int64),
            "lower": pl.Series([10, 20], dtype=pl.Int64),
            "upper": pl.Series([10, 20], dtype=pl.Int64),
            "clients": pl.Series([2, 1], dtype=pl.UInt32),
            "target": pl.Series([1, 1], dtype=pl.Int64),
            "target_pct": pl.Series([50.0, 100.0], dtype=pl.Float64),
        }
    )
    polars.testing.assert_frame_equal(result[0].table, expected)
