"""Fixtures compartidas por los tests de integración del proyecto."""

import pathlib

import polars as pl
import pytest

import tests.support.settings
import tests.support.synthetic
from bank_clients_ml import config, training


@pytest.fixture
def integration_settings(tmp_path: pathlib.Path) -> config.Settings:
    """Devuelve una configuración aislada para los tests de integración."""
    return tests.support.settings.build_settings_for_tests(
        paths=config.DirectoryPaths(project=tmp_path),
        n_jobs=1,
    )


@pytest.fixture
def custom_columns_settings() -> config.Settings:
    """Devuelve una configuración con identificador, target, mes y fechas propios."""
    return tests.support.settings.build_settings_for_tests(
        dataset=config.DatasetColumns(
            id="customer_id",
            target="churn",
            month="snapshot_month",
            first_product_date="first_seen",
            last_product_date="last_seen",
        )
    )


@pytest.fixture(scope="module")
def training_settings() -> config.Settings:
    """Devuelve la configuración usada por los tests de entrenamiento."""
    return tests.support.settings.build_settings_for_tests(
        analysis=config.AnalysisColumns(feature="Variable", importance="Weight"),
        n_jobs=1,
    )


@pytest.fixture(scope="module")
def synthetic_train(training_settings: config.Settings) -> pl.DataFrame:
    """Genera el conjunto sintético compartido por los tests de entrenamiento."""
    return tests.support.synthetic.build_synthetic_train(training_settings.random_state)


@pytest.fixture(scope="module")
def trained_model(
    training_settings: config.Settings,
    synthetic_train: pl.DataFrame,
) -> training.LGBMTrainer:
    """Entrena una única vez el modelo real para los tests de training."""
    return training.LGBMTrainer.fit_and_build(
        synthetic_train,
        tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS,
        training_settings,
        n_iter=tests.support.synthetic.SYNTHETIC_N_ITER,
        splits_cross_validation=(
            tests.support.synthetic.SYNTHETIC_SPLITS_CROSS_VALIDATION
        ),
    )
