"""Tests de integración del entrenamiento con LightGBM real."""

import lightgbm as lgb
import numpy as np
import polars as pl

import tests.support.synthetic
from bank_clients_ml import config, training


def test_should_train_real_lightgbm_and_expose_search_metadata(
    trained_model: training.LGBMTrainer,
    training_settings: config.Settings,
) -> None:
    """Verifica el ajuste real de LightGBM y los metadatos de la búsqueda.

    ## Args:
        trained_model: Entrenador ajustado con LightGBM real.
        training_settings: Configuración usada durante el entrenamiento.
    """
    assert isinstance(trained_model, training.LGBMTrainer)
    assert isinstance(trained_model.best_model, lgb.LGBMClassifier)
    assert trained_model.settings is training_settings
    assert trained_model.n_iter == tests.support.synthetic.SYNTHETIC_N_ITER
    assert (
        trained_model.splits_cross_validation
        == tests.support.synthetic.SYNTHETIC_SPLITS_CROSS_VALIDATION
    )
    assert trained_model.random_state == training_settings.random_state
    assert 0.0 <= trained_model.best_score <= 1.0
    assert trained_model.search_logs.strip()
    assert "Fitting" in trained_model.search_logs

    assert trained_model.best_model.get_params()["n_jobs"] == training_settings.n_jobs

    cross_validation_results = trained_model.cv_results
    assert cross_validation_results.height == tests.support.synthetic.SYNTHETIC_N_ITER
    assert "params" not in cross_validation_results.columns
    assert cross_validation_results.get_column("rank_test_score").to_list() == [1]


def test_should_build_importances_for_used_features(
    trained_model: training.LGBMTrainer,
    training_settings: config.Settings,
) -> None:
    """Verifica las variables usadas y el orden de sus importancias.

    ## Args:
        trained_model: Entrenador ajustado con LightGBM real.
        training_settings: Configuración con los nombres de las columnas.
    """
    feature_column = training_settings.analysis.feature
    importance_column = training_settings.analysis.importance

    assert trained_model.best_model.booster_.num_feature() == len(
        tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS
    )
    assert trained_model.importances.columns == [feature_column, importance_column]
    assert trained_model.importances.height == len(
        tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS
    )
    assert set(trained_model.features) == set(
        tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS
    )
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
        synthetic_train.select(tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS)
    )
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (tests.support.synthetic.SYNTHETIC_N_SAMPLES, 2)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0

    top_features = trained_model.get_top_ranked_features(3)
    assert len(top_features) == 3
    assert set(top_features).issubset(tests.support.synthetic.SYNTHETIC_FEATURE_COLUMNS)
