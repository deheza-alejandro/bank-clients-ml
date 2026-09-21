import io
from contextlib import redirect_stderr, redirect_stdout
from typing import cast

import lightgbm as lgb
import marimo as mo
import numpy as np
import polars as pl
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from bank_clients_ml.column_groups import group_columns_by_source
from bank_clients_ml.config import Settings, get_settings
from bank_clients_ml.visualization import (
    plot_deciles,
    plot_evaluation_metrics,
    plot_top_features,
)


def _get_feature_importances(
    train: pl.DataFrame,
    columns: list[str],
    n_iter: int,
    splits_cross_validation: int,
    settings: Settings | None = None,
) -> tuple[RandomizedSearchCV, pl.DataFrame, str]:
    """Entrena un modelo LightGBM usando RandomizedSearchCV y devuelve las
    importancias de features en un DataFrame de Polars.

    Args:
        train : Datos de entrenamiento; debe contener
            ``columns`` y la columna target.
        columns : Columnas de features usadas para entrenar el modelo.
        No debe tener la columna target.
        n_iter : Cantidad de combinaciones de hiperparámetros a probar al azar con
            ``RandomizedSearchCV``.
        target : Nombre de la columna target.
        splits_cross_validation : Cantidad de k-fold cross-validation para usar con
        ``RandomizedSearchCV``. por defecto es 3

    Returns:
        ``(searcher, importances)`` El objeto searcher entrenado
        y el DataFrame con las importancias ordenadas de forma descendente.
    """
    if settings is None:
        settings = get_settings()

    verbose: int = 3 if settings.debug else 1

    model = lgb.LGBMClassifier(
        random_state=settings.random_state,
        n_jobs=1,
        verbose=verbose,
        metric="auc",
    )

    param_test = {
        "n_estimators": np.arange(6, 50, 1),
        "max_depth": np.arange(
            4, 10, 1
        ),  # [4, 5, 6, 7, 8, 9] de 4 a (10-1) aumentando de a 1
        "num_leaves": np.arange(3, 20, 1),
        "subsample": sp_uniform(loc=0.2, scale=0.8),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": np.arange(1000, 3000, 100),
    }

    searcher = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_test,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=StratifiedKFold(
            n_splits=splits_cross_validation,
            shuffle=True,
            random_state=settings.random_state,
        ),
        verbose=verbose,
        random_state=settings.random_state,
    )

    X_train = train.select(columns)
    y_train = train[settings.col_target]

    buffer = io.StringIO()
    with (
        redirect_stdout(buffer),
        redirect_stderr(buffer),
    ):
        searcher.fit(X_train, y_train)

    output = buffer.getvalue()

    model: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.col_feature: columns,
            settings.col_importance: model.feature_importances_,
        }
    ).sort(settings.col_importance, descending=True)
    return searcher, importances, output


def _rename_columns(
    importances: pl.DataFrame,
    renames_dict: dict[str, str],
    settings: Settings | None = None,
) -> pl.DataFrame:
    settings = settings or get_settings()

    return importances.with_columns(
        pl.col(settings.col_feature)
        .replace_strict(renames_dict, default=pl.col(settings.col_feature))
        .alias(settings.col_feature)
    )


DECILE_LABELS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
DECILE_DTYPE = pl.Enum(DECILE_LABELS)


def _compute_prediction_deciles(
    df: pl.DataFrame,
    probabilities: np.ndarray,
    bins: list[float] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Combina los datos del cliente con sus probabilidades de predicción,
    calcula los deciles y calcula métricas por decil utilizando Polars.

    Parámetros:
    -----------
    df :
        DataFrame original que contiene las columnas 'Target' y 'client_id'.
    probabilities :
        Array unidimensional con las probabilidades predichas por el modelo.
    bins : opcional
        Si no es nulo, usa los bins (por defecto es nulo).

    Retorna:
    --------
        DataFrame con métricas de los deciles
    """
    if settings is None:
        settings = get_settings()

    decile_expr = (
        pl.col("probabilities").cut(breaks=bins, labels=DECILE_LABELS)
        if bins
        else pl.col("probabilities").qcut(
            10, labels=DECILE_LABELS, allow_duplicates=True
        )
    )

    deciles_df = (
        df.select(
            settings.col_target,
            probabilities=probabilities,
        )
        .with_columns(decile=decile_expr.cast(DECILE_DTYPE))
        .group_by("decile")
        .agg(
            count=pl.len(),
            target_1_count=pl.col(settings.col_target).sum().cast(pl.UInt32),
            min_prob=(pl.col("probabilities").min() * 100).round(2),
            max_prob=(pl.col("probabilities").max() * 100).round(2),
        )
        .sort("decile", descending=True)
    )

    target_1_count = pl.col("target_1_count")
    count = pl.col("count")
    target_0_count = count - target_1_count

    total_target_1_rate = (target_1_count.sum() / count.sum() * 100).round(2)
    target_1_rate = (target_1_count / count * 100).round(2)
    cum_gain = (target_1_count.cum_sum() / target_1_count.sum() * 100).round(2)
    cum_target_0_rate = (target_0_count.cum_sum() / target_0_count.sum() * 100).round(2)

    return deciles_df.with_columns(
        target_1_rate=target_1_rate,
        cum_gain=cum_gain,
        lift=(target_1_rate / total_target_1_rate).round(2),
        ks=cum_gain - cum_target_0_rate,
    )


QUANTILES = np.linspace(0.1, 0.9, 9)


def _evaluate(
    searcher: RandomizedSearchCV,
    train: pl.DataFrame,
    test: pl.DataFrame,
    columns: list[str],
    settings: Settings | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame]:
    """y_pred predice si es 0 o 1, si la probabilidad es > 0.5 lo pone como 1

    train_based_bins son los 9 puntos de corte (cuantiles 10% a 90%)
    basados exclusivamente en el set de entrenamiento.
    """
    settings = settings or get_settings()

    model: lgb.LGBMClassifier = searcher.best_estimator_

    y_pred = cast(np.ndarray, model.predict(test.select(columns)))

    probabilities_train = cast(np.ndarray, model.predict_proba(train.select(columns)))[
        :, 1
    ]
    probabilities_test = cast(np.ndarray, model.predict_proba(test.select(columns)))[
        :, 1
    ]

    train_based_bins = np.quantile(probabilities_train, QUANTILES).tolist()

    train_deciles = _compute_prediction_deciles(
        train, probabilities_train, settings=settings
    )
    test_deciles = _compute_prediction_deciles(
        test, probabilities_test, train_based_bins, settings=settings
    )

    return y_pred, probabilities_test, train_deciles, test_deciles


class LGBMTrainer:
    def __init__(
        self,
        train: pl.DataFrame,
        columns: list[str],
        top_n: int = 5,
        n_iter: int = 2,
        splits_cross_validation: int = 3,
        test: pl.DataFrame | None = None,
        renames_dict: dict[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.columns = columns
        self.top_n = top_n

        self.searcher, self.importances, self.output = _get_feature_importances(
            train,
            self.columns,
            n_iter,
            splits_cross_validation,
            settings=self.settings,
        )

        if renames_dict is not None:
            self.importances = _rename_columns(
                self.importances, renames_dict, self.settings
            )

        self.test = test
        if self.test is not None:
            (
                self.y_pred,
                self.probabilities_test,
                self.train_deciles,
                self.test_deciles,
            ) = _evaluate(self.searcher, train, self.test, self.columns, self.settings)

    def get_columns(self) -> list[str]:
        return self.columns

    def print_output(self) -> None:
        mo.output.append(mo.md(f"```text\n{self.output}\n```"))

    def print_searcher(self) -> None:
        mo.output.append(self.searcher)

    def plot_top_features(self, graphic_name: str) -> None:
        plot_top_features(
            self.importances, graphic_name, self.searcher, settings=self.settings
        )

    def get_most_important_features(self) -> list[str]:
        return (
            self.importances.head(self.top_n)
            .get_column(self.settings.col_feature)
            .to_list()
        )

    def plot_evaluation_metrics(self, graphic_name: str) -> None:
        if self.test is None:
            raise RuntimeError("Esta instancia no fue inicializada con un set de test")
        plot_evaluation_metrics(
            self.test[self.settings.col_target],
            self.probabilities_test,
            self.y_pred,
            graphic_name=graphic_name,
        )

    def plot_deciles(self, graphic_name: str) -> None:
        if self.test is None:
            raise RuntimeError("Esta instancia no fue inicializada con un set de test")
        plot_deciles(
            self.train_deciles.drop("min_prob", "max_prob"),
            self.test_deciles.drop("min_prob", "max_prob"),
            graphic_name=graphic_name,
        )


class GroupsLGBMTrainer:
    def __init__(
        self,
        uncorrelated_train: pl.DataFrame,
        n_iter: int = 2,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()

        columns_groups = group_columns_by_source(uncorrelated_train, settings)

        self.trainers: dict[str, LGBMTrainer] = {
            group_name: LGBMTrainer(
                uncorrelated_train, cols, top_n, n_iter, settings=settings
            )
            for group_name, (cols, top_n) in columns_groups.items()
        }

    def print_groups_lengths(self) -> None:
        for group_name, trainer in self.trainers.items():
            mo.output.append(f"{group_name}: {len(trainer.get_columns())}")

        mo.output.append(mo.md("\n\n### others:"))
        mo.output.append(self.trainers["others"].get_columns())

    def print_searchers(self) -> None:
        for group_name, trainer in self.trainers.items():
            mo.output.append(mo.md(f"### {group_name}:"))
            trainer.print_searcher()

    def plot_top_features(self) -> None:
        for group_name, trainer in self.trainers.items():
            trainer.plot_top_features(group_name)

    def print_outputs(self) -> None:
        for group_name, trainer in self.trainers.items():
            mo.output.append(mo.md(f"### {group_name}:"))
            trainer.print_output()

    def get_most_important_features(self) -> list[str]:
        return [
            feature
            for group_name, trainer in self.trainers.items()
            if group_name != "all_columns"
            for feature in trainer.get_most_important_features()
        ]
