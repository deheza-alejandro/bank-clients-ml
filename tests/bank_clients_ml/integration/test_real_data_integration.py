"""Smoke test del pipeline sobre el conjunto de datos real del proyecto."""

import polars as pl
import polars.testing
import pytest

import tests.support.settings
from bank_clients_ml import cleaning, config, sampling


@pytest.fixture(scope="module")
def real_settings() -> config.Settings:
    """Settings reales del proyecto con n_jobs=1."""
    return tests.support.settings.build_settings_for_tests(n_jobs=1)


@pytest.fixture(scope="module")
def real_raw_data(real_settings: config.Settings) -> pl.DataFrame:
    """Carga los datos reales del proyecto si están disponibles."""
    real_raw_data_path = real_settings.paths.raw_data
    if not real_raw_data_path.is_file():
        pytest.skip(f"The actual dataset was not found in {real_raw_data_path}")
    return pl.read_parquet(real_raw_data_path)


@pytest.mark.slow
def test_should_run_real_data_until_time_windows(
    real_raw_data: pl.DataFrame,
    real_settings: config.Settings,
) -> None:
    """Verifica la limpieza y las ventanas temporales sobre datos reales.

    ## Args:
        real_raw_data: DataFrame crudo leído desde el parquet del proyecto.
    """
    clean_data = cleaning.clean_up_data(real_raw_data, real_settings)
    time_window_split = sampling.TimeWindowSplit.generate(
        clean_data,
        2,
        settings=real_settings,
    )

    assert real_raw_data.height > 0
    assert clean_data.height > 0
    assert clean_data.get_column(real_settings.dataset.target).null_count() == 0
    assert clean_data.schema[real_settings.dataset.month] == pl.Date
    assert clean_data.schema[real_settings.dataset.id] == pl.Int64

    assert time_window_split.training_months
    assert time_window_split.prediction_months
    assert (
        time_window_split.last_training_month < time_window_split.first_prediction_month
    )
    assert time_window_split.universe_and_target.height > 0
    assert (
        time_window_split.universe_and_target.get_column(real_settings.dataset.target)
        .is_in([0, 1])
        .all()
    )

    universe_ids = set(
        time_window_split.universe_and_target.get_column(
            real_settings.dataset.id
        ).to_list()
    )
    universe_counts = (
        clean_data.filter(pl.col(real_settings.dataset.id).is_in(universe_ids))
        .group_by(real_settings.dataset.id)
        .len()
    )
    assert universe_counts.get_column("len").unique().to_list() == [9]

    expected_targets = (
        clean_data.filter(
            pl.col(real_settings.dataset.id).is_in(universe_ids)
            & pl.col(real_settings.dataset.month).is_in(
                time_window_split.prediction_months
            )
        )
        .group_by(real_settings.dataset.id)
        .agg(pl.col(real_settings.dataset.target).max())
        .sort(real_settings.dataset.id)
    )
    polars.testing.assert_frame_equal(
        time_window_split.universe_and_target.sort(real_settings.dataset.id),
        expected_targets,
    )

    assert set(
        time_window_split.training_data.get_column(real_settings.dataset.month)
    ) == set(time_window_split.training_months)
    assert set(
        time_window_split.prediction_data.get_column(real_settings.dataset.month)
    ) == set(time_window_split.prediction_months)
