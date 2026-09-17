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

from bank_clients_ml.config import Settings, get_settings
from bank_clients_ml.features import group_columns_by_source
from bank_clients_ml.graphs import (
    plot_deciles,
    plot_evaluation_metrics,
    plot_top_features,
)

RANDOM_STATE: int = 314


def stratified_train_test_split(
    df: pl.DataFrame,
    test_ratio: float = 0.3,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Genera particiones de entrenamiento y test estratificadas
    (manteniendo la proporción de buenos y malos en ambos sets de train y test) usando Polars.

    Parámetros:
    -----------
    df : DataFrame de Polars con los datos.
    target : Nombre de la columna objetivo.
    test_size : Proporción del conjunto de test.
    random_state : Semilla aleatoria.

    Retorna:
    --------
        Tupla con los DataFrames de entrenamiento y test.
    """
    if settings is None:
        settings = get_settings()

    df_flagged = df.sample(fraction=1.0, shuffle=True, seed=random_state).with_columns(
        _group_id=pl.col(settings.col_target).cum_count().over(settings.col_target),
        _group_total=pl.len().over(settings.col_target),
    )

    is_test = pl.col("_group_id") <= (pl.col("_group_total") * test_ratio).round()

    test = df_flagged.filter(is_test).drop("_group_id", "_group_total")
    train = df_flagged.filter(~is_test).drop("_group_id", "_group_total")

    return train, test


def get_feature_importances(
    train: pl.DataFrame,
    columns: list[str],
    n_iter: int = 2,
    splits_cross_validation: int = 3,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> tuple[RandomizedSearchCV, pl.DataFrame]:
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
        random_state :
        debug :

    Returns:
        ``(searcher, importances)`` El objeto searcher entrenado
        y el DataFrame con las importancias ordenadas de forma descendente.
    """
    if settings is None:
        settings = get_settings()

    verbose: int = 3 if settings.debug else 1

    model = lgb.LGBMClassifier(
        random_state=random_state,
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

    searcher: RandomizedSearchCV = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_test,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=StratifiedKFold(
            n_splits=splits_cross_validation,
            shuffle=True,
            random_state=random_state,
        ),
        verbose=verbose,
        random_state=random_state,
    )

    X_train = train.select(columns)
    y_train = train[settings.col_target]
    searcher.fit(X_train, y_train)

    best_estimator: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.col_feature: columns,
            settings.col_importance: best_estimator.feature_importances_,
        }
    ).sort(settings.col_importance, descending=True)
    return searcher, importances


class GroupsLGBMTrainer:
    def __init__(
        self,
        uncorrelated_train: pl.DataFrame,
        settings: Settings | None = None,
    ):
        if settings is None:
            settings = get_settings()

        self.all_cols = [
            col
            for col in uncorrelated_train.columns
            if col not in {settings.col_id, settings.col_target}
        ]
        self.columns_by_source = group_columns_by_source(self.all_cols)

        self.all_cols_buffer = io.StringIO()
        with (
            redirect_stdout(self.all_cols_buffer),
            redirect_stderr(self.all_cols_buffer),
        ):
            self.all_cols_searcher, self.all_cols_importances = get_feature_importances(
                uncorrelated_train, self.all_cols
            )

        self.saving_account_days_transactions_buffer = io.StringIO()
        with (
            redirect_stdout(self.saving_account_days_transactions_buffer),
            redirect_stderr(self.saving_account_days_transactions_buffer),
        ):
            (
                self.saving_account_days_transactions_searcher,
                self.saving_account_days_transactions_importances,
            ) = get_feature_importances(
                uncorrelated_train,
                self.columns_by_source["saving_account_days_transactions"],
            )

        self.saving_account_monetary_buffer = io.StringIO()
        with (
            redirect_stdout(self.saving_account_monetary_buffer),
            redirect_stderr(self.saving_account_monetary_buffer),
        ):
            (
                self.saving_account_monetary_searcher,
                self.saving_account_monetary_importances,
            ) = get_feature_importances(
                uncorrelated_train, self.columns_by_source["saving_account_monetary"]
            )

        self.operations_buffer = io.StringIO()
        with (
            redirect_stdout(self.operations_buffer),
            redirect_stderr(self.operations_buffer),
        ):
            self.operations_searcher, self.operations_importances = (
                get_feature_importances(
                    uncorrelated_train, self.columns_by_source["operations"]
                )
            )

        self.credit_card_payment_buffer = io.StringIO()
        with (
            redirect_stdout(self.credit_card_payment_buffer),
            redirect_stderr(self.credit_card_payment_buffer),
        ):
            self.credit_card_payment_searcher, self.credit_card_payment_importances = (
                get_feature_importances(
                    uncorrelated_train, self.columns_by_source["credit_card_payment"]
                )
            )

        self.credit_card_monetary_buffer = io.StringIO()
        with (
            redirect_stdout(self.credit_card_monetary_buffer),
            redirect_stderr(self.credit_card_monetary_buffer),
        ):
            (
                self.credit_card_monetary_searcher,
                self.credit_card_monetary_importances,
            ) = get_feature_importances(
                uncorrelated_train, self.columns_by_source["credit_card_monetary"]
            )

        self.others_buffer = io.StringIO()
        with (
            redirect_stdout(self.others_buffer),
            redirect_stderr(self.others_buffer),
        ):
            self.others_searcher, self.others_importances = get_feature_importances(
                uncorrelated_train, self.columns_by_source["others"]
            )

    def print_groups_lengths(self) -> None:
        mo.output.append(f"all_cols: {len(self.all_cols)}")
        mo.output.append(
            f"saving_account_days_transactions: "
            f"{len(self.columns_by_source['saving_account_days_transactions'])}"
        )
        mo.output.append(
            f"saving_account_monetary: {len(self.columns_by_source['saving_account_monetary'])}"
        )
        mo.output.append(f"operations: {len(self.columns_by_source['operations'])}")
        mo.output.append(
            f"credit_card_payment: {len(self.columns_by_source['credit_card_payment'])}"
        )
        mo.output.append(
            f"credit_card_monetary: {len(self.columns_by_source['credit_card_monetary'])}"
        )
        mo.output.append(f"others: {len(self.columns_by_source['others'])}\n\n")

        mo.output.append(mo.md("### others:"))
        mo.output.append(self.columns_by_source["others"])

    def print_searchers(self) -> None:
        mo.output.append(mo.md("### all_cols:"))
        mo.output.append(self.all_cols_searcher)
        mo.output.append(mo.md("### saving_account_days_transactions:"))
        mo.output.append(self.saving_account_days_transactions_searcher)
        mo.output.append(mo.md("### saving_account_monetary:"))
        mo.output.append(self.saving_account_monetary_searcher)
        mo.output.append(mo.md("### operations:"))
        mo.output.append(self.operations_searcher)
        mo.output.append(mo.md("### credit_card_payment:"))
        mo.output.append(self.credit_card_payment_searcher)
        mo.output.append(mo.md("### credit_card_monetary:"))
        mo.output.append(self.credit_card_monetary_searcher)
        mo.output.append(mo.md("### others:"))
        mo.output.append(self.others_searcher)

    def plot_top_features(self) -> None:
        plot_top_features(
            self.all_cols_importances, "all_cols_importances", self.all_cols_searcher
        )
        plot_top_features(
            self.saving_account_days_transactions_importances,
            "cols_saving_account_days_transactions_importances",
            self.saving_account_days_transactions_searcher,
        )
        plot_top_features(
            self.saving_account_monetary_importances,
            "cols_saving_account_monetary_importances",
            self.saving_account_monetary_searcher,
            30,
        )
        plot_top_features(
            self.operations_importances,
            "cols_operations_importances",
            self.operations_searcher,
        )
        plot_top_features(
            self.credit_card_payment_importances,
            "cols_credit_card_payment_importances",
            self.credit_card_payment_searcher,
        )
        plot_top_features(
            self.credit_card_monetary_importances,
            "cols_credit_card_monetary_importances",
            self.credit_card_monetary_searcher,
        )
        plot_top_features(
            self.others_importances, "cols_others_importances", self.others_searcher
        )

    def print_outputs(self) -> None:
        mo.output.append(mo.md("all_cols:"))
        mo.output.append(mo.md(f"```text\n{self.all_cols_buffer.getvalue()}\n```"))
        mo.output.append(mo.md("saving_account_days_transactions:"))
        mo.output.append(
            mo.md(
                f"```text\n{self.saving_account_days_transactions_buffer.getvalue()}\n```"
            )
        )
        mo.output.append(mo.md("saving_account_monetary:"))
        mo.output.append(
            mo.md(f"```text\n{self.saving_account_monetary_buffer.getvalue()}\n```")
        )
        mo.output.append(mo.md("operations:"))
        mo.output.append(mo.md(f"```text\n{self.operations_buffer.getvalue()}\n```"))
        mo.output.append(mo.md("credit_card_payment:"))
        mo.output.append(
            mo.md(f"```text\n{self.credit_card_payment_buffer.getvalue()}\n```")
        )
        mo.output.append(mo.md("credit_card_monetary:"))
        mo.output.append(
            mo.md(f"```text\n{self.credit_card_monetary_buffer.getvalue()}\n```")
        )
        mo.output.append(mo.md("others:"))
        mo.output.append(mo.md(f"```text\n{self.others_buffer.getvalue()}\n```"))

    def get_most_important_features(self) -> list[str]:
        most_important_features = [
            *self.saving_account_days_transactions_importances.head(1)
            .get_column("Feature")
            .to_list(),
            *self.saving_account_monetary_importances.head(1)
            .get_column("Feature")
            .to_list(),
            *self.operations_importances.head(1).get_column("Feature").to_list(),
            *self.credit_card_payment_importances.head(1)
            .get_column("Feature")
            .to_list(),
            *self.credit_card_monetary_importances.head(2)
            .get_column("Feature")
            .to_list(),
            *self.others_importances.head(3).get_column("Feature").to_list(),
        ]
        return most_important_features


class LGBMTrainer:
    def __init__(
        self,
        train: pl.DataFrame,
        columns: list[str],
        n_iter: int = 2,
        test: pl.DataFrame | None = None,
        renames_dict: dict[str, str] | None = None,
        settings: Settings | None = None,
    ):
        if settings is None:
            settings = get_settings()

        self.test = test

        self.buffer = io.StringIO()
        with (
            redirect_stdout(self.buffer),
            redirect_stderr(self.buffer),
        ):
            self.searcher, self.importances = get_feature_importances(
                train, columns, n_iter
            )

        if renames_dict is not None:
            self.importances = self.importances.with_columns(
                pl.col(settings.col_feature)
                .replace_strict(renames_dict, default=pl.col(settings.col_feature))
                .alias(settings.col_feature)
            )

        self.evaluator = None
        if test is not None:
            self.evaluator = PerformanceEvaluator(
                self.searcher, train, test, columns, settings
            )

    def print_output(self) -> None:
        mo.output.append(mo.md(f"```text\n{self.buffer.getvalue()}\n```"))

    def print_searcher(self) -> None:
        mo.output.append(self.searcher)

    def plot_top_features(self, graphic_name: str) -> None:
        plot_top_features(
            self.importances,
            graphic_name,
            self.searcher,
        )

    def get_most_important_features(self) -> list[str]:
        return self.importances.head(5).get_column("Feature").to_list()

    def plot_evaluation_metrics(self, graphic_name: str) -> None:
        if self.evaluator is None:
            raise RuntimeError("Esta instancia no fue inicializada con un set de test")
        self.evaluator.plot_evaluation_metrics(graphic_name=graphic_name)

    def plot_deciles(self, graphic_name: str) -> None:
        if self.evaluator is None:
            raise RuntimeError("Esta instancia no fue inicializada con un set de test")
        self.evaluator.plot_deciles(graphic_name=graphic_name)


def oversample_with_unique_ids(
    train: pl.DataFrame,
    target_proportion: float = 0.5,
    random_state: int = RANDOM_STATE,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Asigna IDs únicos a las filas nuevas generadas por el oversampling

    No deberías usar esta función si usas LightGBM + RandomizedSearchCV + StratifiedKFold.
    En ese caso deberías usar algo como imbalanced-learn para hacer oversampling
    solo sobre los datos de entrenamiento de cada fold de StratifiedKFold,
    dejando intactos los datos de validación de cada fold"""
    if not (0 < target_proportion < 1):
        raise ValueError(
            "El parámetro 'target_proportion' debe estar entre 0 y 1 (excluyentes)."
        )

    if settings is None:
        settings = get_settings()

    df_majority = train.filter(pl.col(settings.col_target) == 0)
    df_minority = train.filter(pl.col(settings.col_target) == 1)

    max_id = train.select(pl.col(settings.col_id).max()).item()

    count_majority = len(df_majority)
    count_minority_target = round(
        count_majority * (target_proportion / (1 - target_proportion))
    )

    df_minority_oversampled = df_minority.sample(
        n=count_minority_target, with_replacement=True, seed=random_state
    ).with_columns((max_id + 1 + pl.int_range(0, pl.len())).alias(settings.col_id))

    balanced_train = pl.concat([df_majority, df_minority_oversampled]).sample(
        fraction=1.0, shuffle=True, seed=random_state
    )
    return balanced_train


QUANTILES = np.linspace(0.1, 0.9, 9)


def get_scoring(
    searcher: RandomizedSearchCV,
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """y_pred predice si es 0 o 1, si la probabilidad es > 0.5 lo pone como 1

    train_based_bins son los 9 puntos de corte (cuantiles 10% a 90%)
    basados exclusivamente en el set de entrenamiento.
    """
    final_model: lgb.LGBMClassifier = searcher.best_estimator_

    y_pred = cast(np.ndarray, final_model.predict(test.select(features)))

    probabilities_train = cast(
        np.ndarray, final_model.predict_proba(train.select(features))
    )[:, 1]
    probabilities_test = cast(
        np.ndarray, final_model.predict_proba(test.select(features))
    )[:, 1]

    train_based_bins = np.quantile(probabilities_train, QUANTILES).tolist()

    return y_pred, probabilities_train, probabilities_test, train_based_bins


DECILE_LABELS = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
DECILE_DTYPE = pl.Enum(DECILE_LABELS)


def compute_prediction_deciles(
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


class PerformanceEvaluator:
    def __init__(
        self,
        searcher: RandomizedSearchCV,
        train: pl.DataFrame,
        test: pl.DataFrame,
        columns: list[str],
        settings: Settings | None = None,
    ):
        self.test = test

        if settings is None:
            self.settings = get_settings()
        else:
            self.settings = settings

        (
            self.y_pred,
            probabilities_train,
            self.probabilities_test,
            train_based_bins,
        ) = get_scoring(searcher, train, test, columns)

        self.train_deciles = compute_prediction_deciles(train, probabilities_train)
        self.test_deciles = compute_prediction_deciles(
            test, self.probabilities_test, train_based_bins
        )

    def plot_evaluation_metrics(self, graphic_name="lightgbm") -> None:
        plot_evaluation_metrics(
            self.test[self.settings.col_target],
            self.probabilities_test,
            self.y_pred,
            graphic_name=graphic_name,
        )

    def plot_deciles(self, graphic_name="deciles") -> None:
        plot_deciles(
            self.train_deciles.drop("min_prob", "max_prob"),
            self.test_deciles.drop("min_prob", "max_prob"),
            graphic_name=graphic_name,
        )
