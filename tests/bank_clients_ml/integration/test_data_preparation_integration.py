"""Tests de integración del flujo de preparación de datos."""

from typing import cast

import numpy as np
import polars as pl
import polars.testing
import pytest

import tests.builders.integration.data
import tests.support.scenario
from bank_clients_ml import column_groups, config


def test_should_clean_raw_data_and_build_time_windows(
    integration_settings: config.Settings,
) -> None:
    """Verifica la limpieza y la construcción del universo temporal.

    ## Args:
        integration_settings: Configuración con los nombres de las columnas.
    """
    raw_bank_data = tests.builders.integration.data.build_raw_bank_data()
    clean_data = tests.builders.integration.data.build_clean_data(
        raw_bank_data, integration_settings
    )
    time_window_split = tests.builders.integration.data.build_time_window_split(
        clean_data, integration_settings
    )
    expected_eligible_ids = tests.builders.integration.data.ELIGIBLE_CLIENT_IDS
    null_target_count = raw_bank_data.get_column("Target").null_count()
    assert null_target_count == tests.support.scenario.NULL_TARGET_COUNT
    assert clean_data.get_column("Target").null_count() == 0
    assert clean_data.height == raw_bank_data.height - null_target_count
    assert clean_data.schema["Month"] == pl.Date
    assert clean_data.schema["First_product_dt"] == pl.Date
    assert clean_data.schema["Last_product_dt"] == pl.Date
    assert clean_data.schema["client_id"] == pl.Int64

    assert time_window_split.training_months == tests.support.scenario.TRAINING_MONTHS
    assert (
        time_window_split.prediction_months == tests.support.scenario.PREDICTION_MONTHS
    )
    assert (
        time_window_split.last_training_month
        == tests.support.scenario.LAST_TRAINING_MONTH
    )
    assert (
        time_window_split.first_prediction_month
        == tests.support.scenario.FIRST_PREDICTION_MONTH
    )
    assert tests.support.scenario.LEAD_MONTH not in time_window_split.training_months
    assert tests.support.scenario.LEAD_MONTH not in time_window_split.prediction_months
    assert tests.support.scenario.LEAD_MONTH in set(
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
        tests.support.scenario.TRAINING_MONTHS
    )
    assert set(time_window_split.prediction_data.get_column("Month")) == set(
        tests.support.scenario.PREDICTION_MONTHS
    )


def _assert_identity_features(identity_features: pl.DataFrame) -> None:
    """Valida las identity features codificadas y las fechas derivadas.

    ## Args:
        identity_features: Identity features ya codificadas y con fechas derivadas.
    """
    expected_eligible_ids = tests.builders.integration.data.ELIGIBLE_CLIENT_IDS

    assert identity_features.height == len(expected_eligible_ids)
    assert (
        identity_features.get_column("client_id").n_unique() == identity_features.height
    )
    assert set(identity_features.get_column("client_id").to_list()) == set(
        expected_eligible_ids
    )

    for column in column_groups.CATEGORICAL_COLS:
        assert identity_features.schema[column].is_numeric()
        assert cast(float, identity_features.get_column(column).min()) >= 0.0
        assert cast(float, identity_features.get_column(column).max()) <= 100.0

    assert "First_product_dt" not in identity_features.columns
    assert "Last_product_dt" not in identity_features.columns
    assert "Days_between_first_and_last_product" in identity_features.columns
    assert "Recency_in_days" in identity_features.columns
    days_between = cast(
        int,
        identity_features.get_column("Days_between_first_and_last_product").min(),
    )
    recency = cast(int, identity_features.get_column("Recency_in_days").min())
    assert days_between > 0
    assert recency >= 0


def _assert_monthly_transformations(
    transformed: pl.DataFrame,
    training_data: pl.DataFrame,
) -> None:
    """Valida la codificación mensual y los totales derivados.

    ## Args:
        transformed: Historia mensual con transformaciones.
        training_data: Historia mensual de entrenamiento sin transformaciones.
    """
    assert transformed.height == training_data.height
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


def test_should_build_identity_features_and_monthly_transformations(
    integration_settings: config.Settings,
) -> None:
    """Verifica las identity features y las transformaciones mensuales.

    ## Args:
        integration_settings: Configuración con los nombres de las columnas.
    """
    prepared_data = tests.builders.integration.data.build_prepared_data(
        integration_settings
    )
    _assert_identity_features(prepared_data.identity_features)
    _assert_monthly_transformations(
        prepared_data.transformed_training_data,
        prepared_data.time_window_split.training_data,
    )


@pytest.mark.slow
def test_should_build_abt_and_split_preserving_target_rate(
    integration_settings: config.Settings,
) -> None:
    """Verifica la ABT final y la partición estratificada.

    ## Args:
        integration_settings: Configuración con los nombres de las columnas.
    """
    prepared_data = tests.builders.integration.data.build_prepared_data(
        integration_settings
    )
    abt = prepared_data.abt
    train, test = prepared_data.train, prepared_data.test
    expected_eligible_ids = tests.builders.integration.data.ELIGIBLE_CLIENT_IDS

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
    custom_columns_settings: config.Settings,
) -> None:
    """Verifica que el flujo respete identificador, target, mes y fechas propios.

    ## Args:
        custom_columns_settings: Configuración con columnas personalizadas.
    """
    raw_bank_data = tests.builders.integration.data.build_raw_bank_data()
    renamed_raw_data = tests.builders.integration.data.rename_raw_data(
        raw_bank_data, custom_columns_settings
    )
    clean_custom = tests.builders.integration.data.build_clean_data(
        renamed_raw_data, custom_columns_settings
    )
    prepared = tests.builders.integration.data.prepare_data(
        clean_custom, custom_columns_settings
    )
    expected_eligible_ids = tests.builders.integration.data.ELIGIBLE_CLIENT_IDS
    identity = prepared.identity_features
    abt = prepared.abt

    for frame in (prepared.clean_data, prepared.time_window_split.training_data):
        assert {"customer_id", "churn", "snapshot_month"} <= set(frame.columns)
        assert {"client_id", "Target", "Month"}.isdisjoint(frame.columns)

    assert {"customer_id", "churn"} <= set(identity.columns)
    assert {
        "client_id",
        "Target",
        "first_seen",
        "last_seen",
    }.isdisjoint(identity.columns)
    assert "Days_between_first_and_last_product" in identity.columns
    assert "Recency_in_days" in identity.columns

    assert abt.height == len(expected_eligible_ids)
    assert abt.get_column("customer_id").n_unique() == abt.height
    assert {"snapshot_month", "Month", "Target"}.isdisjoint(abt.columns)

    train_ids = set(prepared.train.get_column("customer_id").to_list())
    test_ids = set(prepared.test.get_column("customer_id").to_list())
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(abt.get_column("customer_id").to_list())
