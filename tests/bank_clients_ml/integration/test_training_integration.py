"""Tests de integración del entrenamiento con LightGBM real."""

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import sklearn.datasets

from bank_clients_ml import config, training

pytestmark = pytest.mark.integration_lgbm

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


def _build_synthetic_train(random_state: int) -> pl.DataFrame:
    """Genera un conjunto sintético compatible con el esquema del proyecto.

    El espacio de búsqueda adapta `min_child_samples` al tamaño de los pliegues,
    por lo que este conjunto alcanza para que el modelo realice al menos una
    partición sin necesidad de agrandar los datos artificialmente.

    ## Args:
        random_state: Semilla para reproducir la generación de los datos.

    ## Returns:
        DataFrame con cinco variables, una columna extra que no debe usarse y
            un target binario desbalanceado.
    """
    features, target = sklearn.datasets.make_classification(
        n_samples=N_SAMPLES,
        n_features=len(FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.5,
        flip_y=0.01,
        random_state=random_state,
    )
    return pl.DataFrame(features, schema=list(FEATURE_COLUMNS)).with_columns(
        pl.Series("Target", target, dtype=pl.Int64),
        pl.lit(0.0).alias("Ignored"),
    )


@pytest.fixture(scope="module")
def custom_settings() -> config.Settings:
    """Devuelve la configuración personalizada usada por los tests.

    ## Returns:
        Configuración con nombres propios para las columnas de análisis.
    """
    return config.Settings(
        analysis=config.AnalysisColumns(feature="Variable", importance="Weight"),
        n_jobs=1,
    )


@pytest.fixture(scope="module")
def synthetic_train(custom_settings: config.Settings) -> pl.DataFrame:
    """Genera el conjunto sintético con la semilla configurada.

    ## Args:
        custom_settings: Configuración que define la semilla de los datos.

    ## Returns:
        DataFrame sintético compatible con el esquema del proyecto.
    """
    return _build_synthetic_train(custom_settings.random_state)


@pytest.fixture(scope="module")
def trained_model(
    custom_settings: config.Settings,
    synthetic_train: pl.DataFrame,
) -> training.LGBMTrainer:
    """Entrena una única vez el modelo real usado por los tests.

    ## Args:
        custom_settings: Configuración del entrenamiento.
        synthetic_train: Datos sintéticos de entrenamiento.

    ## Returns:
        Entrenador ajustado con la búsqueda de hiperparámetros.
    """
    return training.LGBMTrainer.fit_and_build(
        synthetic_train,
        FEATURE_COLUMNS,
        custom_settings,
        n_iter=N_ITER,
        splits_cross_validation=SPLITS_CROSS_VALIDATION,
    )


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
