"""Entrenamiento y evaluación de modelos LightGBM para "bank_clients_ml".

Centraliza el ajuste de hiperparámetros con RandomizedSearchCV y StratifiedKFold,
el cálculo de deciles de probabilidad y la evaluación sobre el conjunto de prueba.

## Clases exportadas:
    LGBMTrainer: Entrenador de un modelo LightGBM, expone métricas y gráficos.
    GroupsLGBMTrainer: Entrenador de modelos LightGBM por grupos de variables (usa LGBMTrainer).
"""

import contextlib
import dataclasses
import io
import types
from collections.abc import ItemsView, Iterable
from typing import Self, cast

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.stats
import sklearn.metrics
import sklearn.model_selection

from bank_clients_ml import binning, column_groups, config, evaluation, visualization


def _fit_lgbm_random_search(
    train: pl.DataFrame,
    columns: Iterable[str],
    *,
    n_iter: int,
    splits_cross_validation: int,
    settings: config.Settings,
) -> tuple[sklearn.model_selection.RandomizedSearchCV, pl.DataFrame, str]:
    """Entrena un modelo clasificador LightGBM con RandomizedSearchCV y StratifiedKFold.

    Configura el modelo clasificador con StratifiedKFold, ejecuta la
    búsqueda capturando los registros de salida y construye la tabla de
    importancias ordenada de mayor a menor.

    ## Args:
        train: Datos de entrenamiento que incluyen la variable target.
        columns: Columnas utilizadas para el entrenamiento. No debe tener la columna target.
        n_iter: Cantidad de combinaciones de hiperparámetros a evaluar al azar
            con `RandomizedSearchCV`.
        splits_cross_validation: Cantidad de splits de la validación cruzada
            para usar con ``RandomizedSearchCV``.
        settings: Configuración con semilla, nivel de detalle y nombres de
            columnas. Si no se indica, se obtiene la configuración global.

    ## Returns:
        Tupla con el buscador entrenado, la tabla de importancias ordenada de
            forma descendente y los registros capturados durante el entrenamiento.
    """
    verbose: int = 3 if settings.debug else 1

    classifier = lgb.LGBMClassifier(
        random_state=settings.random_state,
        n_jobs=1,
        verbose=-1,
        metric="auc",
    )

    param_distributions = {
        "n_estimators": np.arange(6, 50, 1),
        "max_depth": np.arange(
            4, 10, 1
        ),  # [4, 5, 6, 7, 8, 9] from 4 to (10-1) increasing by 1
        "num_leaves": np.arange(3, 20, 1),
        "subsample": scipy.stats.uniform(loc=0.2, scale=0.8),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_samples": np.arange(1000, 3000, 100),
    }

    searcher = sklearn.model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        cv=sklearn.model_selection.StratifiedKFold(
            n_splits=splits_cross_validation,
            shuffle=True,
            random_state=settings.random_state,
        ),
        verbose=verbose,
        random_state=settings.random_state,
    )

    X_train = train.select(columns)
    y_train = train.get_column(settings.dataset.target)

    with (
        io.StringIO() as log_buffer,
        contextlib.redirect_stdout(log_buffer),
        contextlib.redirect_stderr(log_buffer),
    ):
        searcher.fit(X_train, y_train)
        search_logs = log_buffer.getvalue()

    fitted_classifier: lgb.LGBMClassifier = searcher.best_estimator_
    importances = pl.DataFrame(
        {
            settings.analysis.feature: columns,
            settings.analysis.importance: fitted_classifier.feature_importances_,
        }
    ).sort(settings.analysis.importance, descending=True)
    return searcher, importances, search_logs


@dataclasses.dataclass(frozen=True, slots=True)
class LGBMTrainer:
    """Entrenador de un modelo LightGBM, expone métricas y gráficos.

    ## Attributes:
        columns: Columnas utilizadas de `train`.
        searcher: Buscador entrenado con la mejor combinación encontrada.
        search_logs: Registros capturados durante el entrenamiento.
        is_testable: Indica si se realizo la instanciación con un conjunto de prueba.
    """

    train: pl.DataFrame
    columns: tuple[str, ...]
    importances: pl.DataFrame
    search_logs: str
    n_iter: int
    splits_cross_validation: int
    settings: config.Settings
    _searcher: sklearn.model_selection.RandomizedSearchCV

    @classmethod
    def fit_and_build(
        cls,
        train: pl.DataFrame,
        columns: Iterable[str],
        settings: config.Settings,
        *,
        n_iter: int = 2,
        splits_cross_validation: int = 3,
    ) -> Self:
        """Entrena el modelo LightGBM con las columnas indicadas.

        Ejecuta la búsqueda aleatoria de hiperparámetros y, cuando se provee
        un conjunto de prueba, calcula métricas de evaluación y tablas de
        deciles sobre el mejor modelo

        ## Args:
            train: Datos de entrenamiento que incluyen la variable target.
            columns: Columnas utilizadas de `train`.
            n_iter: Cantidad de combinaciones de hiperparámetros a evaluar.
            splits_cross_validation: Cantidad de splits de validación cruzada.
            test: Datos de prueba para evaluar el modelo. Si no se indica, la
                instancia de LGBMTrainer queda sin métricas de evaluación.
            settings: Configuración global del proyecto. Si no se indica, se
                obtiene la configuración global.
        """
        _searcher, importances, search_logs = _fit_lgbm_random_search(
            train,
            columns,
            n_iter=n_iter,
            splits_cross_validation=splits_cross_validation,
            settings=settings,
        )

        return cls(
            train,
            tuple(columns),
            importances,
            search_logs,
            n_iter,
            splits_cross_validation,
            settings,
            _searcher,
        )

    @property
    def cv_results(self) -> pl.DataFrame:
        """Retorna componentes de marimo para visualizar los resultados del entrenamiento.

        Permite visualizar las combinaciones de hiperparámetros de los modelos
        probados con validación cruzada, ordenados por la métrica obtenida
        (`rank_test_score`). Opcionalmente permite ver los logs.

        ## Args:
            include_logs: Si es True, incluye los logs de búsqueda en formato de texto.

        ## Returns:
            elementos visuales de marimo listos para mostrarse en el notebook.
        """
        cross_validation_results = (
            pl.DataFrame(self._searcher.cv_results_)
            .select(
                [
                    pl.col("rank_test_score"),
                    pl.all().exclude(["rank_test_score", "params"]),
                ]
            )
            .sort("rank_test_score")
        )
        return cross_validation_results

    def plot_top_features(
        self,
        plot_name: str,
        name_mapping: binning.NameMapping | None = None,
    ) -> None:
        """Genera el gráfico con las variables más importantes del modelo.

        ## Args:
            plot_name: Nombre base del archivo SVG a generar, sin extensión.
            name_mapping: Nombre nuevo asociado al nombre original de una feature.
                Si no se indica, se conservan los nombres originales.
        """
        if name_mapping is not None:
            temp_importances = self.importances.with_columns(
                name_mapping.to_expr(self.settings.analysis.feature)
            )
        else:
            temp_importances = self.importances

        visualization.plot_top_features(
            temp_importances,
            plot_name,
            roc_auc=self._searcher.best_score_,
            settings=self.settings,
        )

    def get_top_ranked_features(self, top_n: int = 5) -> tuple[str, ...]:
        """Retorna los nombres de las variables mas relevantes según importancia.

        ## Args:
            top_n: Cantidad de variables consideradas al consultar el ranking.
        """
        return tuple(
            self.importances.head(top_n)
            .get_column(self.settings.analysis.feature)
            .to_list()
        )

    def evaluate_and_plot_metrics(
        self, test: pl.DataFrame, *, roc_curve_name: str, deciles_name: str
    ) -> None:
        """Genera gráficos con las métricas de evaluación sobre el set de prueba.

        Las métricas incluyen: curva ROC AUC, accuracy, Tasas de falsos positivos,
        Tasas de verdaderos positivos, tablas de deciles de entrenamiento y prueba.

        ## Args:
            plot_name: Nombre base del archivo SVG a generar, sin extensión.
        """
        best_model: lgb.LGBMClassifier = self._searcher.best_estimator_
        X_train = self.train.select(self.columns)
        X_test = test.select(self.columns)
        probabilities_train = cast(np.ndarray, best_model.predict_proba(X_train))[:, 1]
        probabilities_test = cast(np.ndarray, best_model.predict_proba(X_test))[:, 1]

        evaluation_metrics = evaluation.evaluate(
            train=self.train,
            test=test,
            probabilities_train=probabilities_train,
            probabilities_test=probabilities_test,
            settings=self.settings,
        )

        visualization.plot_roc_curve(evaluation_metrics, roc_curve_name, self.settings)
        visualization.plot_deciles(
            evaluation_metrics, deciles_name, settings=self.settings
        )


def _generate_trainers(
    uncorrelated_train: pl.DataFrame,
    *,
    n_iter: int,
    splits_cross_validation: int,
    settings: config.Settings,
) -> types.MappingProxyType[column_groups.ValidColumnGroup, LGBMTrainer]:
    """Entrena cada grupo de variables con un LGBMTrainer distinto.

    ## Args:
        uncorrelated_train: Datos de entrenamiento sin columnas redundantes,
            que incluyen la variable target.
        n_iter: Cantidad de combinaciones de hiperparámetros a evaluar en
            cada grupo.
        settings: Configuración global del proyecto.
    """
    generated_columns_groups = column_groups.group_columns_by_source(
        uncorrelated_train, settings
    )

    trainers = {
        group: LGBMTrainer.fit_and_build(
            uncorrelated_train,
            columns,
            settings,
            n_iter=n_iter,
            splits_cross_validation=splits_cross_validation,
        )
        for group, columns in generated_columns_groups.items()
    }
    return types.MappingProxyType(trainers)


@dataclasses.dataclass(frozen=True, slots=True)
class GroupsLGBMTrainer:
    """Entrenador de modelos LightGBM por grupos de variables (usa LGBMTrainer).

    Agrupa las columnas por fuente de negocio y entrena un `LGBMTrainer`
    independiente por cada grupo. Permite comparar la importancia de las
    variables dentro de cada fuente antes de la selección final.

    ## Attributes:
        trainers: Instancias de `LGBMTrainer` indexados por nombre de grupo.

    ## Example:
        from bank_clients_ml import training

        group_trainers = training.GroupsLGBMTrainer(uncorrelated_train)
        selected = group_trainers.get_top_grouped_features()
    """

    trainers: types.MappingProxyType[column_groups.ValidColumnGroup, LGBMTrainer]
    n_iter: int
    splits_cross_validation: int
    settings: config.Settings

    @classmethod
    def fit_and_build(
        cls,
        uncorrelated_train: pl.DataFrame,
        settings: config.Settings,
        *,
        n_iter: int = 2,
        splits_cross_validation: int = 3,
    ) -> Self:
        """Entrena cada grupo de variables con un LGBMTrainer distinto.

        ## Args:
            uncorrelated_train: Datos de entrenamiento sin columnas redundantes,
                que incluyen la variable target.
            n_iter: Cantidad de combinaciones de hiperparámetros a evaluar en
                cada grupo.
            settings: Configuración global del proyecto. Si no se indica, se
                obtiene la configuración global.
        """
        trainers = _generate_trainers(
            uncorrelated_train,
            n_iter=n_iter,
            splits_cross_validation=splits_cross_validation,
            settings=settings,
        )

        return cls(trainers, n_iter, splits_cross_validation, settings)

    def items(self) -> ItemsView[column_groups.ValidColumnGroup, LGBMTrainer]:
        """Add docs."""
        return self.trainers.items()

    @property
    def others_columns(self) -> tuple[str, ...]:
        """Add docs."""
        return self.trainers[column_groups.ValidColumnGroup.OTHERS].columns

    @property
    def groups_name_mapping(self) -> binning.NameMapping:
        """Add docs."""
        return binning.NameMapping.from_dual_named(
            column_groups.ValidColumnGroup.members()
        )

    @property
    def groups_lengths(self) -> tuple[str, ...]:
        """Retorna componentes de marimo con la cantidad de columnas de cada grupo."""
        return tuple(
            [
                f"{group.pretty_name}: {len(trainer.columns)}"
                for group, trainer in self.trainers.items()
            ]
        )

    def plot_top_features(self) -> None:
        """Genera el gráfico de variables importantes para cada grupo."""
        for group, trainer in self.trainers.items():
            trainer.plot_top_features(group.snake_case_name)

    def get_top_grouped_features(self) -> tuple[str, ...]:
        """Retorna las mejores variables de cada grupo, sin el grupo base.

        Combina las variables mejor posicionadas de todos los grupos, con
        excepción del grupo de referencia que contiene todas las columnas.

        ## Returns:
            Nombres de las variables seleccionadas por grupo.
        """
        top_grouped_features = []

        for group, trainer in self.trainers.items():
            top_grouped_features.extend(trainer.get_top_ranked_features(group.top_n))

        return tuple(top_grouped_features)
