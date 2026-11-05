"""Smoke test del pipeline sobre el conjunto de datos real del proyecto."""

import pathlib

import polars as pl
import polars.testing
import pytest

from bank_clients_ml import cleaning, config, sampling

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_DATA_PATH = PROJECT_ROOT / "data" / "data.parquet"


@pytest.fixture(scope="module")
def real_raw_data() -> pl.DataFrame:
    """Carga los datos reales del proyecto si están disponibles."""
    if not REAL_DATA_PATH.is_file():
        pytest.skip(f"No se encontró el conjunto de datos real en {REAL_DATA_PATH}")
    return pl.read_parquet(REAL_DATA_PATH)


def test_should_run_real_data_until_time_windows(
    real_raw_data: pl.DataFrame,
) -> None:
    """Verifica la limpieza y las ventanas temporales sobre datos reales.

    ## Args:
        real_raw_data: DataFrame crudo leído desde el parquet del proyecto.
    """
    settings = config.Settings(n_jobs=1)

    clean_data = cleaning.clean_up_data(real_raw_data, settings)
    time_window_split = sampling.TimeWindowSplit.generate(
        clean_data,
        2,
        settings=settings,
    )

    assert real_raw_data.height > 0
    assert clean_data.height > 0
    assert clean_data.get_column(settings.dataset.target).null_count() == 0
    assert clean_data.schema[settings.dataset.month] == pl.Date
    assert clean_data.schema[settings.dataset.id] == pl.Int64

    assert time_window_split.training_months
    assert time_window_split.prediction_months
    assert (
        time_window_split.last_training_month < time_window_split.first_prediction_month
    )
    assert time_window_split.universe_and_target.height > 0
    assert (
        time_window_split.universe_and_target.get_column(settings.dataset.target)
        .is_in([0, 1])
        .all()
    )

    universe_ids = set(
        time_window_split.universe_and_target.get_column(settings.dataset.id).to_list()
    )
    universe_counts = (
        clean_data.filter(pl.col(settings.dataset.id).is_in(universe_ids))
        .group_by(settings.dataset.id)
        .len()
    )
    assert universe_counts.get_column("len").unique().to_list() == [9]

    expected_targets = (
        clean_data.filter(
            pl.col(settings.dataset.id).is_in(universe_ids)
            & pl.col(settings.dataset.month).is_in(time_window_split.prediction_months)
        )
        .group_by(settings.dataset.id)
        .agg(pl.col(settings.dataset.target).max())
        .sort(settings.dataset.id)
    )
    polars.testing.assert_frame_equal(
        time_window_split.universe_and_target.sort(settings.dataset.id),
        expected_targets,
    )

    assert set(
        time_window_split.training_data.get_column(settings.dataset.month)
    ) == set(time_window_split.training_months)
    assert set(
        time_window_split.prediction_data.get_column(settings.dataset.month)
    ) == set(time_window_split.prediction_months)
