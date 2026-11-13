"""Tests de integración de la evaluación con un modelo LightGBM real."""

from typing import cast

import numpy as np
import polars as pl
import pytest

import tests.builders.integration.evaluation
import tests.support.settings
from bank_clients_ml import config, evaluation, training


@pytest.fixture(scope="module")
def evaluation_settings() -> config.Settings:
    """Devuelve la configuración usada por el entrenamiento de evaluación."""
    return tests.support.settings.build_settings_for_tests(n_jobs=1)


@pytest.fixture(scope="module")
def evaluation_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera un entrenamiento y un holdout sintéticos con target binario."""
    return tests.builders.integration.evaluation.build_evaluation_frames()


@pytest.fixture(scope="module")
def evaluation_trainer(
    evaluation_frames: tuple[pl.DataFrame, pl.DataFrame],
    evaluation_settings: config.Settings,
) -> training.LGBMTrainer:
    """Entrena un modelo real sobre la partición de entrenamiento."""
    train, _ = evaluation_frames
    return training.LGBMTrainer.fit_and_build(
        train,
        tests.builders.integration.evaluation.FEATURE_COLUMNS,
        evaluation_settings,
        n_iter=tests.builders.integration.evaluation.N_ITER,
        splits_cross_validation=tests.builders.integration.evaluation.SPLITS_CROSS_VALIDATION,
    )


def test_should_evaluate_real_model_and_build_consistent_deciles(
    evaluation_frames: tuple[pl.DataFrame, pl.DataFrame],
    evaluation_trainer: training.LGBMTrainer,
    evaluation_settings: config.Settings,
) -> None:
    """Verifica las métricas globales y la consistencia de los deciles.

    ## Args:
        evaluation_frames: Conjuntos de entrenamiento y holdout sintéticos.
        evaluation_trainer: Entrenador real ajustado sobre el entrenamiento.
        evaluation_settings: Configuración con el nombre del target.
    """
    train, test = evaluation_frames

    probabilities_train_matrix = cast(
        np.ndarray,
        evaluation_trainer.best_model.predict_proba(
            train.select(tests.builders.integration.evaluation.FEATURE_COLUMNS)
        ),
    )
    probabilities_train = probabilities_train_matrix[:, 1]

    probabilities_test_matrix = cast(
        np.ndarray,
        evaluation_trainer.best_model.predict_proba(
            test.select(tests.builders.integration.evaluation.FEATURE_COLUMNS)
        ),
    )
    probabilities_test = probabilities_test_matrix[:, 1]

    metrics = evaluation.evaluate(
        train=train,
        test=test,
        probabilities_train=probabilities_train,
        probabilities_test=probabilities_test,
        settings=evaluation_settings,
    )

    assert metrics.roc_auc > 0.6
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.fpr.min() >= 0.0
    assert metrics.fpr.max() <= 1.0
    assert metrics.tpr.min() >= 0.0
    assert metrics.tpr.max() <= 1.0
    assert len(metrics.fpr) == len(metrics.tpr)

    for deciles, frame in (
        (metrics.train_deciles, train),
        (metrics.test_deciles, test),
    ):
        assert deciles.get_column("count").sum() == frame.height
        assert (
            deciles.get_column("target_1_count").sum()
            == frame.get_column("Target").sum()
        )
        assert deciles.get_column("cum_gain").tail(1).item() == pytest.approx(100.0)
        assert np.isfinite(deciles.get_column("lift").to_numpy()).all()
        assert np.isfinite(deciles.get_column("ks").to_numpy()).all()

    test_deciles = metrics.test_deciles.sort("decile", descending=True)
    assert test_deciles.height >= 2
    rates = test_deciles.get_column("target_1_rate").to_list()
    assert rates[0] > rates[-1]
