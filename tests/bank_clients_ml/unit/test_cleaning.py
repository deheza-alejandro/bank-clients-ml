"""Tests unitarios para el módulo de limpieza inicial de datos."""

from collections.abc import Callable
from datetime import date

import polars as pl
import polars.testing
import pytest

import tests.builders.unit.cleaning
import tests.support.settings
from bank_clients_ml import cleaning, config


def test_should_drop_rows_with_null_target(
    settings: config.Settings,
) -> None:
    """Verifica que se descarten los registros cuyo target es nulo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    raw_data = tests.builders.unit.cleaning.build_raw_data()
    result = cleaning.clean_up_data(raw_data, settings)

    assert result.get_column("client_id").to_list() == [1, 3]


def test_should_keep_zero_target_value(settings: config.Settings) -> None:
    """Verifica que un target igual a cero no se confunda con un valor nulo.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    raw_data = pl.DataFrame(
        {
            "client_id": ["1", "2"],
            "Month": ["2024-01-01", "2024-01-01"],
            "First_product_dt": ["2020-01-15", "2020-01-15"],
            "Last_product_dt": ["2024-01-01", "2024-01-01"],
            "Target": [0, None],
        },
        schema={
            "client_id": pl.String,
            "Month": pl.String,
            "First_product_dt": pl.String,
            "Last_product_dt": pl.String,
            "Target": pl.Int64,
        },
    )

    result = cleaning.clean_up_data(raw_data, settings)

    assert result.get_column("Target").to_list() == [0]


def test_should_cast_client_id_to_int64(
    settings: config.Settings,
) -> None:
    """Verifica que el identificador de texto se convierta a entero de 64 bits.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    raw_data = tests.builders.unit.cleaning.build_raw_data()
    result = cleaning.clean_up_data(raw_data, settings)

    expected = pl.Series("client_id", [1, 3], dtype=pl.Int64)
    polars.testing.assert_series_equal(result.get_column("client_id"), expected)


@pytest.mark.parametrize(
    ("column", "expected_dates"),
    [
        pytest.param("Month", [date(2024, 1, 1), None], id="month"),
        pytest.param(
            "First_product_dt",
            [date(2020, 1, 15), None],
            id="first-product-date",
        ),
        pytest.param(
            "Last_product_dt",
            [date(2024, 1, 1), None],
            id="last-product-date",
        ),
    ],
)
def test_should_convert_date_column_to_date_type(
    settings: config.Settings,
    column: str,
    expected_dates: list[date | None],
) -> None:
    """Verifica que las columnas de fecha se conviertan a tipo fecha.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        column: Nombre de la columna de fecha que se valida.
        expected_dates: Fechas esperadas luego de la conversión.
    """
    raw_data = tests.builders.unit.cleaning.build_raw_data()
    result = cleaning.clean_up_data(raw_data, settings)

    expected = pl.Series(column, expected_dates, dtype=pl.Date)
    polars.testing.assert_series_equal(result.get_column(column), expected)


def test_should_preserve_columns_unrelated_to_cleaning(
    settings: config.Settings,
) -> None:
    """Verifica que las columnas ajenas a la limpieza se conserven intactas.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    raw_data = tests.builders.unit.cleaning.build_raw_data()
    result = cleaning.clean_up_data(raw_data, settings)

    expected = pl.Series("Balance", [100.0, 300.0], dtype=pl.Float64)
    polars.testing.assert_series_equal(result.get_column("Balance"), expected)


def test_should_return_empty_frame_when_all_targets_are_null(
    settings: config.Settings,
) -> None:
    """Verifica que el resultado quede vacío si todos los targets son nulos.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    raw_data = pl.DataFrame(
        {
            "client_id": ["1", "2"],
            "Month": ["2024-01-01", "2024-02-01"],
            "First_product_dt": ["2020-01-15", "2020-02-15"],
            "Last_product_dt": ["2024-01-01", "2024-02-01"],
            "Target": [None, None],
        },
        schema={
            "client_id": pl.String,
            "Month": pl.String,
            "First_product_dt": pl.String,
            "Last_product_dt": pl.String,
            "Target": pl.Int64,
        },
    )

    result = cleaning.clean_up_data(raw_data, settings)

    assert result.height == 0
    assert result.columns == raw_data.columns
    assert result.schema["client_id"] == pl.Int64
    assert result.schema["Month"] == pl.Date


def test_should_use_column_names_from_settings_when_provided(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se respeten los nombres de columna de la configuración.

    ## Args:
        custom_dataset_settings: Configuración con los nombres alternativos.
    """
    raw_data = pl.DataFrame(
        {
            "customer_id": ["7", "8"],
            "snapshot_month": ["2024-05-01", "2024-06-01"],
            "first_seen": ["2021-05-01", "2021-06-01"],
            "last_seen": ["2024-05-01", "2024-06-01"],
            "churn": [1, None],
        },
        schema={
            "customer_id": pl.String,
            "snapshot_month": pl.String,
            "first_seen": pl.String,
            "last_seen": pl.String,
            "churn": pl.Int64,
        },
    )

    result = cleaning.clean_up_data(raw_data, custom_dataset_settings)

    assert result.get_column("customer_id").to_list() == [7]
    assert result.get_column("snapshot_month").to_list() == [date(2024, 5, 1)]


def test_should_use_global_settings_when_settings_is_not_provided(
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    global_settings = tests.support.settings.build_settings_for_tests(
        dataset=config.DatasetColumns(target="label")
    )
    patch_global_settings(global_settings)
    raw_data = pl.DataFrame(
        {
            "client_id": ["1", "2"],
            "Month": ["2024-01-01", "2024-02-01"],
            "First_product_dt": ["2020-01-15", "2020-02-15"],
            "Last_product_dt": ["2024-01-01", "2024-02-01"],
            "label": [1, None],
        },
        schema={
            "client_id": pl.String,
            "Month": pl.String,
            "First_product_dt": pl.String,
            "Last_product_dt": pl.String,
            "label": pl.Int64,
        },
    )

    result = cleaning.clean_up_data(raw_data)

    assert result.get_column("client_id").to_list() == [1]
    assert result.get_column("Month").to_list() == [date(2024, 1, 1)]
