"""Entrenamiento y evaluación de modelos LightGBM para "bank_clients_ml".

Centraliza el ajuste de hiperparámetros con RandomizedSearchCV y StratifiedKFold,
el cálculo de deciles de probabilidad y la evaluación sobre el conjunto de prueba.

Clases exportadas:
    LGBMTrainer: Entrenador de un modelo LightGBM, expone métricas y gráficos.
    GroupsLGBMTrainer: Entrenador de modelos LightGBM por grupos de variables (usa LGBMTrainer).
"""

import io
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from typing import cast

import lightgbm as lgb
import marimo as mo
import numpy as np
import polars as pl
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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


def _fit_lgbm_random_search(
    train: pl.DataFrame,
    columns: list[str],
    n_iter: int,
    splits_cross_validation: int,
    settings: Settings | None = None,
) -> tuple[RandomizedSearchCV, pl.DataFrame, str]:
    """Entrena un modelo clasificador LightGBM con RandomizedSearchCV y StratifiedKFold.

    Configura el modelo clasificador con StratifiedKFold, ejecuta la
    búsqueda capturando los registros de salida y construye la tabla de
    importancias ordenada de mayor a menor.

    Args:
        train: Datos de entrenamiento que incluyen la variable target.
        columns: Columnas utilizadas para el entrenamiento. No debe tener la columna target.
        n_iter: Cantidad de combinaciones de hiperparámetros a evaluar al azar
            con `RandomizedSearchCV`.
        splits_cross_validation: Cantidad de splits de la validación cruzada
            para usar con ``RandomizedSearchCV``.
        settings: Configuración con semilla, nivel de detalle y nombres de
            columnas. Si no se indica, se obtiene la configuración global.

    Returns:
        Tupla con el buscador entrenado, la tabla de importancias ordenada de
            forma descendente y los registros capturados durante el entrenamiento.
    """
    if settings is None:
        settings = get_settings()

    verbose: int = 3 if settings.debug else 1

    classifier = lgb.LGBMClassifier(
        random_state=settings.random_state,
        n_jobs=1,
        verbose=verbose,
        metric="auc",
    )

    param_distributions = {
        "n_estimators": np.arange(6, 50, 1),
        "max_depth": np.arange(
            4, 10, 1
        ),  # [4, 5, 6, 7, 8, 9] from 4 to (10-1) increasing by 1
        "num_leaves": np.arange(3, 20, 1),
        "subsample": sp_uniform(loc=0.2, scale=0.8),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": np.arange(1000, 3000, 100),
    }

    searcher = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_distributions,
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

    log_buffer = io.StringIO()
    with (
        redirect_stdout(log_buffer),
        redirect_stderr(log_buffer),
    ):
        searcher.fit(X_train, y_train)

    search_logs = log_buffer.getvalue()

    fitted_classifier: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.col_feature: columns,
            settings.col_importance: fitted_classifier.feature_importances_,
        }
    ).sort(settings.col_importance, descending=True)
    return searcher, importances, search_logs


def _with_rename_columns(
    importances: pl.DataFrame,
    renames: Mapping[str, str],
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Aplica nombres distintos a la tabla de importancias de variables.

    Reemplaza los valores de la columna feature según el diccionario de
    renombres y conserva el valor original cuando no existe correspondencia.

    Args:
        importances: Tabla de importancias ordenada de mayor a menor.
        renames: Nombre nuevo asociado al nombre original de una feature.
        settings: Configuración con el nombre de la columna feature. Si no
            se indica, se obtiene la configuración global.

    Returns:
        Nueva tabla de importancias con los valores de la columna feature
        renombrada.
    """
    settings = settings or get_settings()

    return importances.with_columns(
        pl.col(settings.col_feature)
        .replace_strict(renames, default=pl.col(settings.col_feature))
        .alias(settings.col_feature)
    )


def _compute_prediction_deciles(
    df: pl.DataFrame,
    probabilities: np.ndarray,
    bins: list[float] | None = None,
    settings: Settings | None = None,
) -> pl.DataFrame:
    """Calcula métricas de desempeño por decil de probabilidad predicha.

    Segmenta las probabilidades en diez grupos y calcula por cada decil el
    conteo de clientes, la tasa de la clase positiva, la ganancia acumulada,
    el lift y la estadística KS.

    Args:
        df: Datos que incluyen la variable target.
        probabilities: Probabilidades predichas por el modelo para la clase positiva.
        bins: bins para segmentar las probabilidades (util para test).
            Si no se indica, se calcula en deciles según `probabilities`.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    Returns:
        Tabla por decil con conteos, probabilidades mínima y máxima, tasa de
            la clase positiva, ganancia acumulada, lift y KS, ordenada de
            forma descendente por decil.
    """
    if settings is None:
        settings = get_settings()

    decile_labels = ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1"]
    decile_dtype = pl.Enum(decile_labels)

    decile_expr = (
        pl.col("probabilities").cut(breaks=bins, labels=decile_labels)
        if bins
        else pl.col("probabilities").qcut(
            10, labels=decile_labels, allow_duplicates=True
        )
    )

    deciles_df = (
        df.select(
            settings.col_target,
            probabilities=probabilities,
        )
        .with_columns(decile=decile_expr.cast(decile_dtype))
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


def _evaluate(
    searcher: RandomizedSearchCV,
    train: pl.DataFrame,
    test: pl.DataFrame,
    columns: list[str],
    settings: Settings | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame]:
    """Evalúa el mejor modelo sobre el conjunto de prueba.

    Calcula las probabilidades de entrenamiento y prueba, obtiene las
    métricas ROC AUC y accuracy junto con los puntos de la curva ROC, y
    construye las tablas de deciles. Los bins de los deciles de prueba se
    derivan de los bins de las probabilidades de entrenamiento.

    Args:
        searcher: Buscador entrenado cuyo mejor estimador se desea evaluar.
        train: Datos de entrenamiento utilizados para definir los bins.
        test: Datos de prueba sobre los que se calculan las métricas.
        columns: Columnas utilizadas de `train`.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    Returns:
        Tupla con ROC AUC, exactitud, tasas de falsos positivos y de
            verdaderos positivos de la curva ROC, y tablas de deciles de
            entrenamiento y de prueba.
    """
    settings = settings or get_settings()

    best_model: lgb.LGBMClassifier = searcher.best_estimator_
    X_train = train.select(columns)
    X_test = test.select(columns)

    probabilities_train = cast(np.ndarray, best_model.predict_proba(X_train))[:, 1]
    probabilities_test = cast(np.ndarray, best_model.predict_proba(X_test))[:, 1]
    y_true_arr = test.get_column(settings.col_target).to_numpy()
    y_pred = (probabilities_test >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_true_arr, probabilities_test))
    accuracy = float(accuracy_score(y_true_arr, y_pred))
    fpr, tpr, _ = roc_curve(y_true_arr, probabilities_test)

    quantiles = np.linspace(0.1, 0.9, 9)
    train_based_bins = np.quantile(probabilities_train, quantiles).tolist()

    train_deciles = _compute_prediction_deciles(
        train, probabilities_train, settings=settings
    )
    test_deciles = _compute_prediction_deciles(
        test, probabilities_test, train_based_bins, settings=settings
    )

    return (
        roc_auc,
        accuracy,
        cast(np.ndarray, fpr),  # pyrefly: ignore[redundant-cast]
        cast(np.ndarray, tpr),  # pyrefly: ignore[redundant-cast]
        train_deciles,
        test_deciles,
    )


class LGBMTrainer:
    """Entrenador de un modelo LightGBM, expone métricas y gráficos.

    Attributes:
        columns: Columnas utilizadas de `train`.
        searcher: Buscador entrenado con la mejor combinación encontrada.
        search_logs: Registros capturados durante el entrenamiento.
        is_testable: Indica si se realizo la instanciación con un conjunto de prueba.
    """

    def __init__(
        self,
        train: pl.DataFrame,
        columns: list[str],
        top_n: int = 5,
        n_iter: int = 2,
        splits_cross_validation: int = 3,
        test: pl.DataFrame | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Entrena el modelo LightGBM con las columnas indicadas.

        Ejecuta la búsqueda aleatoria de hiperparámetros y, cuando se provee
        un conjunto de prueba, calcula métricas de evaluación y tablas de
        deciles sobre el mejor modelo

        Args:
            train: Datos de entrenamiento que incluyen la variable target.
            columns: Columnas utilizadas de `train`.
            top_n: Cantidad de variables consideradas al consultar el ranking.
            n_iter: Cantidad de combinaciones de hiperparámetros a evaluar.
            splits_cross_validation: Cantidad de splits de validación cruzada.
            test: Datos de prueba para evaluar el modelo. Si no se indica, la
                instancia de LGBMTrainer queda sin métricas de evaluación.
            settings: Configuración global del proyecto. Si no se indica, se
                obtiene la configuración global.
        """
        self._settings = settings or get_settings()

        self.columns = columns
        self._top_n = top_n

        self.searcher, self._importances, self.search_logs = _fit_lgbm_random_search(
            train,
            self.columns,
            n_iter,
            splits_cross_validation,
            settings=self._settings,
        )

        self.is_testable = False
        if test is not None:
            (
                self._roc_auc,
                self._accuracy,
                self._fpr,
                self._tpr,
                self._train_deciles,
                self._test_deciles,
            ) = _evaluate(self.searcher, train, test, self.columns, self._settings)
            self.is_testable = True

    def print_search_logs(self) -> None:
        """Muestra los registros de la búsqueda en la salida del notebook."""
        mo.output.append(mo.md(f"```text\n{self.search_logs}\n```"))

    def print_searcher(self) -> None:
        """Muestra el objeto de búsqueda entrenado en la salida del notebook."""
        mo.output.append(self.searcher)

    def plot_top_features(
        self, plot_name: str, renames: Mapping[str, str] | None = None
    ) -> None:
        """Genera el gráfico con las variables más importantes del modelo.

        Args:
            plot_name: Nombre base del archivo SVG a generar, sin extensión.
            renames: Nombre nuevo asociado al nombre original de una feature.
                Si no se indica, se conservan los nombres originales.
        """
        if renames is not None:
            importances_renamed = _with_rename_columns(
                self._importances, renames, self._settings
            )
            plot_top_features(
                importances_renamed,
                plot_name,
                self.searcher.best_score_,
                settings=self._settings,
            )
        else:
            plot_top_features(
                self._importances,
                plot_name,
                self.searcher.best_score_,
                settings=self._settings,
            )

    def get_top_ranked_features(self) -> list[str]:
        """Retorna los nombres de las variables mas relevantes según importancia."""
        return (
            self._importances.head(self._top_n)
            .get_column(self._settings.col_feature)
            .to_list()
        )

    def _ensure_testable(self) -> None:
        """Verifica que el entrenador disponga de conjunto de prueba.

        Raises:
            RuntimeError: Si el entrenador se inicializó sin conjunto de
                prueba y no es posible evaluar ni graficar métricas.
        """
        if not self.is_testable:
            raise RuntimeError(
                "Cannot plot: LGBMTrainer was initialized without a test set"
            )

    def plot_evaluation_metrics(self, plot_name: str) -> None:
        """Genera la curva ROC con las métricas de evaluación sobre el set de prueba.

        Las métricas incluyen: ROC AUC, accuracy, Tasas de falsos positivos,
        Tasas de verdaderos positivos

        Args:
            plot_name: Nombre base del archivo SVG a generar, sin extensión.

        Raises:
            RuntimeError: Si el entrenador se inicializó sin conjunto de prueba.
        """
        self._ensure_testable()
        plot_evaluation_metrics(
            self._roc_auc, self._accuracy, self._fpr, self._tpr, plot_name
        )

    def plot_deciles(self, plot_name: str) -> None:
        """Genera el gráfico con las tablas de deciles de entrenamiento y prueba.

        Args:
            plot_name: Nombre base del archivo SVG a generar, sin extensión.

        Raises:
            RuntimeError: Si el entrenador se inicializó sin conjunto de prueba.
        """
        self._ensure_testable()
        plot_deciles(
            self._train_deciles.drop("min_prob", "max_prob"),
            self._test_deciles.drop("min_prob", "max_prob"),
            plot_name,
        )


class GroupsLGBMTrainer:
    """Entrenador de modelos LightGBM por grupos de variables (usa LGBMTrainer).

    Agrupa las columnas por fuente de negocio y entrena un `LGBMTrainer`
    independiente por cada grupo. Permite comparar la importancia de las
    variables dentro de cada fuente antes de la selección final.

    Attributes:
        trainers: Instancias de `LGBMTrainer` indexados por nombre de grupo.

    Example:
        from bank_clients_ml.training import GroupsLGBMTrainer

        groups_trainer = GroupsLGBMTrainer(uncorrelated_train)
        selected = groups_trainer.get_top_grouped_features()
    """

    def __init__(
        self,
        uncorrelated_train: pl.DataFrame,
        n_iter: int = 2,
        settings: Settings | None = None,
    ) -> None:
        """Entrena cada grupo de variables con un LGBMTrainer distinto.

        Args:
            uncorrelated_train: Datos de entrenamiento sin columnas redundantes,
                que incluyen la variable target.
            n_iter: Cantidad de combinaciones de hiperparámetros a evaluar en
                cada grupo.
            settings: Configuración global del proyecto. Si no se indica, se
                obtiene la configuración global.
        """
        settings = settings or get_settings()

        columns_groups = group_columns_by_source(uncorrelated_train, settings)

        self.trainers: dict[str, LGBMTrainer] = {
            group_name: LGBMTrainer(
                uncorrelated_train, cols, top_n, n_iter, settings=settings
            )
            for group_name, (cols, top_n) in columns_groups.items()
        }

    def print_groups_lengths(self) -> None:
        """Muestra la cantidad de columnas de cada grupo en el notebook."""
        for group_name, trainer in self.trainers.items():
            mo.output.append(f"{group_name}: {len(trainer.columns)}")

        mo.output.append(mo.md("\n\n### others:"))
        mo.output.append(self.trainers["others"].columns)

    def print_searchers(self) -> None:
        """Muestra los buscadores entrenados de cada grupo en el notebook."""
        for group_name, trainer in self.trainers.items():
            mo.output.append(mo.md(f"### {group_name}:"))
            trainer.print_searcher()

    def plot_top_features(self) -> None:
        """Genera el gráfico de variables importantes para cada grupo."""
        for group_name, trainer in self.trainers.items():
            trainer.plot_top_features(group_name)

    def print_search_logs(self) -> None:
        """Muestra los registros de búsqueda de cada grupo en el notebook."""
        for group_name, trainer in self.trainers.items():
            mo.output.append(mo.md(f"### {group_name}:"))
            trainer.print_search_logs()

    def get_top_grouped_features(self) -> list[str]:
        """Retorna las mejores variables de cada grupo, sin el grupo base.

        Combina las variables mejor posicionadas de todos los grupos, con
        excepción del grupo de referencia que contiene todas las columnas.

        Returns:
            Nombres de las variables seleccionadas por grupo.
        """
        return [
            feature
            for group_name, trainer in self.trainers.items()
            if group_name != "all_columns"
            for feature in trainer.get_top_ranked_features()
        ]
