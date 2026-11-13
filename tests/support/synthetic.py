"""Parámetros y constructores del conjunto sintético compartido por los tests."""

import polars as pl
import sklearn.datasets

SYNTHETIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4",
    "Feature_5",
)
SYNTHETIC_N_SAMPLES = 2_000
SYNTHETIC_N_ITER = 1
SYNTHETIC_SPLITS_CROSS_VALIDATION = 2


def build_synthetic_train(random_state: int) -> pl.DataFrame:
    """Genera un conjunto sintético compatible con el esquema del proyecto.

    ## Args:
        random_state: Semilla usada para generar los datos.

    ## Returns:
        DataFrame con features numéricas, target binario e `Ignored`.
    """
    features, target = sklearn.datasets.make_classification(
        n_samples=SYNTHETIC_N_SAMPLES,
        n_features=len(SYNTHETIC_FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.5,
        flip_y=0.01,
        random_state=random_state,
    )
    return pl.DataFrame(
        features,
        schema=list(SYNTHETIC_FEATURE_COLUMNS),
    ).with_columns(
        pl.Series("Target", target, dtype=pl.Int64),
        pl.lit(0.0).alias("Ignored"),
    )
