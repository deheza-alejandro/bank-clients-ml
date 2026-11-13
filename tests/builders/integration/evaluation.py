"""Constructores de datos usados por la evaluación de integración."""

import polars as pl
import sklearn.datasets
import sklearn.model_selection

FEATURE_COLUMNS: tuple[str, ...] = tuple(f"Feature_{index}" for index in range(1, 9))
N_SAMPLES = 1_200
TEST_SIZE = 0.25
N_ITER = 2
SPLITS_CROSS_VALIDATION = 3


def build_evaluation_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera un entrenamiento y un holdout sintéticos con target binario.

    ## Returns:
        Conjuntos de entrenamiento y prueba sintéticos.
    """
    features, target = sklearn.datasets.make_classification(
        n_samples=N_SAMPLES,
        n_features=len(FEATURE_COLUMNS),
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        weights=[0.7, 0.3],
        class_sep=1.5,
        flip_y=0.02,
        random_state=314,
    )
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=314,
        stratify=target,
    )
    train = pl.DataFrame(x_train, schema=list(FEATURE_COLUMNS)).with_columns(
        pl.Series("Target", y_train, dtype=pl.Int64)
    )
    test = pl.DataFrame(x_test, schema=list(FEATURE_COLUMNS)).with_columns(
        pl.Series("Target", y_test, dtype=pl.Int64)
    )
    return train, test
