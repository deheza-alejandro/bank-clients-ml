"""Tests de integración del entrenamiento con LightGBM real."""

import lightgbm as lgb
import numpy as np
import polars as pl

from bank_clients_ml import config, training

FEATURE_COLUMNS: tuple[str, ...] = (
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4",
    "Feature_5",
)
N_SAMPLES = 2_000
N_ITER = 1
SPLITS_CROSS_VALIDATION = 2


def test_should_train_real_lightgbm_and_expose_search_metadata(
    trained_model: training.LGBMTrainer,
    custom_settings: config.Settings,
) -> None:
    """Verifica el ajuste real de LightGBM y los metadatos de la búsqueda.

    ## Args:
        trained_model: Entrenador ajustado con LightGBM real.
        custom_settings: Configuración usada durante el entrenamiento.
    """
    assert isinstance(trained_model, training.LGBMTrainer)
    assert isinstance(trained_model.best_model, lgb.LGBMClassifier)
    assert trained_model.settings is custom_settings
    assert trained_model.n_iter == N_ITER
    assert trained_model.splits_cross_validation == SPLITS_CROSS_VALIDATION
    assert trained_model.random_state == custom_settings.random_state
    assert 0.0 <= trained_model.best_score <= 1.0
    assert trained_model.search_logs.strip()
    assert "Fitting" in trained_model.search_logs

    assert trained_model.best_model.get_params()["n_jobs"] == custom_settings.n_jobs

    cross_validation_results = trained_model.cv_results
    assert cross_validation_results.height == 1
    assert "params" not in cross_validation_results.columns
    assert cross_validation_results.get_column("rank_test_score").to_list() == [1]


def test_should_build_importances_for_used_features(
    trained_model: training.LGBMTrainer,
    custom_settings: config.Settings,
) -> None:
    """Verifica las variables usadas y el orden de sus importancias.

    ## Args:
        trained_model: Entrenador ajustado con LightGBM real.
        custom_settings: Configuración con los nombres de las columnas.
    """
    feature_column = custom_settings.analysis.feature
    importance_column = custom_settings.analysis.importance

    assert trained_model.best_model.booster_.num_feature() == len(FEATURE_COLUMNS)
    assert trained_model.importances.columns == [feature_column, importance_column]
    assert trained_model.importances.height == len(FEATURE_COLUMNS)
    assert set(trained_model.features) == set(FEATURE_COLUMNS)
    assert "Ignored" not in trained_model.features

    importances = trained_model.importances.get_column(importance_column).to_list()
    assert importances == sorted(importances, reverse=True)
    assert sum(importances) > 0.0


def test_should_predict_probabilities_and_rank_top_features(
    trained_model: training.LGBMTrainer,
    synthetic_train: pl.DataFrame,
) -> None:
    """Verifica las probabilidades predichas y las variables mejor rankeadas.

    ## Args:
        trained_model: Entrenador ajustado con LightGBM real.
        synthetic_train: Datos sintéticos usados durante el entrenamiento.
    """
    probabilities = trained_model.best_model.predict_proba(
        synthetic_train.select(FEATURE_COLUMNS)
    )
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (N_SAMPLES, 2)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0

    top_features = trained_model.get_top_ranked_features(3)
    assert len(top_features) == 3
    assert set(top_features).issubset(FEATURE_COLUMNS)
