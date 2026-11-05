"""Tests de integración del flujo de preparación de datos."""

import datetime

import numpy as np
import polars as pl
import polars.testing
import pytest

from bank_clients_ml import cleaning, column_groups, config, sampling

TRAINING_MONTHS: tuple[datetime.date, ...] = tuple(
    datetime.date(2024, month, 1) for month in range(1, 8)
)
PREDICTION_MONTHS: tuple[datetime.date, ...] = (
    datetime.date(2024, 9, 1),
    datetime.date(2024, 10, 1),
)
LAST_TRAINING_MONTH = datetime.date(2024, 7, 1)
LEAD_MONTH = datetime.date(2024, 8, 1)
NULL_TARGET_COUNT = 3


def test_should_clean_raw_data_and_build_time_windows(
    raw_bank_data: pl.DataFrame,
    clean_data: pl.DataFrame,
    time_window_split: sampling.TimeWindowSplit,
    expected_eligible_ids: tuple[int, ...],
) -> None:
    """Verifica la limpieza y la construcción del universo temporal.

    ## Args:
        raw_bank_data: Datos crudos con clientes válidos, inválidos y nulos.
        clean_data: Datos resultantes de la limpieza.
        time_window_split: Partición temporal construida sobre los datos limpios.
        expected_eligible_ids: Clientes que deben formar el universo.
    """
    assert raw_bank_data.get_column("Target").null_count() == NULL_TARGET_COUNT
    assert clean_data.get_column("Target").null_count() == 0
    assert clean_data.height == raw_bank_data.height - NULL_TARGET_COUNT
    assert clean_data.schema["Month"] == pl.Date
    assert clean_data.schema["First_product_dt"] == pl.Date
    assert clean_data.schema["Last_product_dt"] == pl.Date
    assert clean_data.schema["client_id"] == pl.Int64

    assert time_window_split.training_months == TRAINING_MONTHS
    assert time_window_split.prediction_months == PREDICTION_MONTHS
    assert time_window_split.last_training_month == LAST_TRAINING_MONTH
    assert time_window_split.first_prediction_month == PREDICTION_MONTHS[0]
    assert LEAD_MONTH not in time_window_split.training_months
    assert LEAD_MONTH not in time_window_split.prediction_months
    assert LEAD_MONTH in set(
        time_window_split.universe_and_target_data.get_column("Month")
    )

    universe_ids = tuple(
        time_window_split.universe_and_target.get_column("client_id").to_list()
    )
    assert universe_ids == expected_eligible_ids
    assert time_window_split.universe_and_target.height == len(expected_eligible_ids)
    assert time_window_split.universe_and_target.get_column("Target").sum() == 24

    eligible_counts = (
        clean_data.filter(pl.col("client_id").is_in(expected_eligible_ids))
        .group_by("client_id")
        .len()
    )
    assert eligible_counts.get_column("len").unique().to_list() == [9]

    expected_targets = (
        clean_data.filter(
            pl.col("client_id").is_in(expected_eligible_ids)
            & pl.col("Month").is_in(time_window_split.prediction_months)
        )
        .group_by("client_id")
        .agg(pl.col("Target").max())
        .sort("client_id")
    )
    polars.testing.assert_frame_equal(
        time_window_split.universe_and_target.sort("client_id"),
        expected_targets,
    )

    assert set(time_window_split.training_data.get_column("Month")) == set(
        TRAINING_MONTHS
    )
    assert set(time_window_split.prediction_data.get_column("Month")) == set(
        PREDICTION_MONTHS
    )


def test_should_build_identity_features_and_monthly_transformations(
    prepared_data,
    expected_eligible_ids: tuple[int, ...],
) -> None:
    """Verifica las identity features y las transformaciones mensuales.

    ## Args:
        prepared_data: Artefactos intermedios del flujo de preparación.
        expected_eligible_ids: Clientes que forman el universo.
    """
    identity_features = prepared_data.identity_features
    transformed = prepared_data.transformed_training_data
    time_window_split = prepared_data.time_window_split

    assert identity_features.height == len(expected_eligible_ids)
    assert (
        identity_features.get_column("client_id").n_unique() == identity_features.height
    )
    assert set(identity_features.get_column("client_id").to_list()) == set(
        expected_eligible_ids
    )

    for column in column_groups.CATEGORICAL_COLS:
        assert identity_features.schema[column].is_numeric()
        assert identity_features.get_column(column).min() >= 0.0
        assert identity_features.get_column(column).max() <= 100.0

    assert "First_product_dt" not in identity_features.columns
    assert "Last_product_dt" not in identity_features.columns
    assert "Days_between_first_and_last_product" in identity_features.columns
    assert "Recency_in_days" in identity_features.columns
    assert identity_features.get_column("Days_between_first_and_last_product").min() > 0
    assert identity_features.get_column("Recency_in_days").min() >= 0

    assert transformed.height == time_window_split.training_data.height
    for column in column_groups.CATEGORICAL_COLS:
        assert column not in transformed.columns

    percentage_columns = [
        column for column in transformed.columns if column.endswith("_pct")
    ]
    assert percentage_columns
    for column in percentage_columns:
        series = transformed.get_column(column)
        assert series.null_count() == 0
        assert series.is_finite().all()

    operation_columns = [
        "Operations_Bank",
        "Operations_Terminal",
        "Operations_HomeBanking",
        "Operations_Mobile",
        "Operations_Ivr",
        "Operations_Telemarketer",
        "Operations_ATM",
    ]
    expected_operations_total = transformed.select(
        pl.sum_horizontal(operation_columns).alias("Operations_total")
    ).get_column("Operations_total")
    np.testing.assert_allclose(
        transformed.get_column("Operations_total").to_numpy(),
        expected_operations_total.to_numpy(),
    )

    payment_columns = [
        "CreditCard_Payment_Aut_Debit",
        "CreditCard_Payment_External",
        "CreditCard_Payment_Cash",
        "CreditCard_Payment_Web",
        "CreditCard_Payment_ATM",
        "CreditCard_Payment_TAS",
    ]
    expected_payment_total = transformed.select(
        pl.sum_horizontal(payment_columns).alias("CreditCard_Payment_total")
    ).get_column("CreditCard_Payment_total")
    np.testing.assert_allclose(
        transformed.get_column("CreditCard_Payment_total").to_numpy(),
        expected_payment_total.to_numpy(),
    )


def test_should_build_abt_and_split_preserving_target_rate(
    abt: pl.DataFrame,
    train_test_abt: tuple[pl.DataFrame, pl.DataFrame],
    expected_eligible_ids: tuple[int, ...],
) -> None:
    """Verifica la ABT final y la partición estratificada.

    ## Args:
        abt: Analytical Base Table a nivel de cliente.
        train_test_abt: Conjuntos de entrenamiento y prueba.
        expected_eligible_ids: Clientes que forman el universo.
    """
    train, test = train_test_abt

    assert abt.height == len(expected_eligible_ids)
    assert abt.get_column("client_id").n_unique() == abt.height
    assert "First_product_dt" not in abt.columns
    assert "Last_product_dt" not in abt.columns

    feature_columns = [
        column for column in abt.columns if column not in {"client_id", "Target"}
    ]
    for column in feature_columns:
        assert abt.get_column(column).null_count() == 0
        if abt.schema[column] in (pl.Float32, pl.Float64):
            assert abt.get_column(column).is_finite().all()

    train_ids = set(train.get_column("client_id").to_list())
    test_ids = set(test.get_column("client_id").to_list())
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(abt.get_column("client_id").to_list())
    assert train.height + test.height == abt.height

    abt_rate = abt.get_column("Target").mean()
    assert train.get_column("Target").mean() == pytest.approx(abt_rate, abs=0.05)
    assert test.get_column("Target").mean() == pytest.approx(abt_rate, abs=0.08)
    assert "_group_id" not in train.columns
    assert "_group_total" not in train.columns
    assert "_group_id" not in test.columns
    assert "_group_total" not in test.columns


def test_should_run_data_preparation_with_custom_column_names(
    raw_bank_data: pl.DataFrame,
    custom_columns_settings: config.Settings,
    rename_raw_data,
    prepare_data,
    expected_eligible_ids: tuple[int, ...],
) -> None:
    """Verifica que el flujo respete identificador, target, mes y fechas propios.

    ## Args:
        raw_bank_data: Datos crudos con los nombres por defecto.
        custom_columns_settings: Configuración con columnas personalizadas.
        rename_raw_data: Helper para renombrar las columnas configuradas.
        prepare_data: Helper que ejecuta el flujo de preparación.
        expected_eligible_ids: Clientes que deben sobrevivir al filtrado.
    """
    renamed_raw_data = rename_raw_data(raw_bank_data, custom_columns_settings)
    clean_custom = cleaning.clean_up_data(renamed_raw_data, custom_columns_settings)
    prepared = prepare_data(clean_custom, custom_columns_settings)
    identity = prepared.identity_features
    abt = prepared.abt

    for frame in (prepared.clean_data, prepared.time_window_split.training_data):
        assert "customer_id" in frame.columns
        assert "churn" in frame.columns
        assert "snapshot_month" in frame.columns
        assert "client_id" not in frame.columns
        assert "Target" not in frame.columns
        assert "Month" not in frame.columns

    assert "customer_id" in identity.columns
    assert "churn" in identity.columns
    assert "client_id" not in identity.columns
    assert "Target" not in identity.columns
    assert "first_seen" not in identity.columns
    assert "last_seen" not in identity.columns
    assert "Days_between_first_and_last_product" in identity.columns
    assert "Recency_in_days" in identity.columns

    assert abt.height == len(expected_eligible_ids)
    assert abt.get_column("customer_id").n_unique() == abt.height
    assert "snapshot_month" not in abt.columns
    assert "Month" not in abt.columns
    assert "Target" not in abt.columns

    train_ids = set(prepared.train.get_column("customer_id").to_list())
    test_ids = set(prepared.test.get_column("customer_id").to_list())
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(abt.get_column("customer_id").to_list())
