"""Análisis bivariado y selección de variables para "bank_clients_ml".

## Clases exportadas:
    FeaturesSelection: Selección de variables para el modelado.
    BivariateAnalyzer: Analizador bivariado sobre conjuntos con variables correlacionadas.
"""

import dataclasses
from collections.abc import Iterable
from typing import Self, cast

import numpy as np
import polars as pl

from bank_clients_ml import (
    binning,
    bivariate_table,
    config,
    evaluation,
    redundant_column_filter,
    training,
    visualization,
)


@dataclasses.dataclass(frozen=True, slots=True)
class FeaturesSelection:
    """Selección de variables para el modelado.

    Agrupa los conjuntos de entrenamiento y prueba ya transformados por bines
    junto con el mapeo de nombres y el modelo entrenado y listo para la
    visualización y evaluación final.

    ## Attributes:
        train: Conjunto de entrenamiento con las columnas identificadora,
            target y variables transformadas.
        test: Conjunto de prueba con las columnas identificadora, target y
            variables transformadas.
        name_mapping: Mapeo entre los nombres originales en snake_case y los
            nombres finales.
        trainer: Entrenador ya entrenado sobre el conjunto de entrenamiento.
        settings: Configuración general del proyecto.
        n_iter: Número de iteraciones usado en la optimización de hiperparámetros.
    """

    train: pl.DataFrame
    test: pl.DataFrame
    name_mapping: binning.NameMapping
    trainer: training.LGBMTrainer
    settings: config.Settings
    n_iter: int

    @classmethod
    def fit_and_build(
        cls,
        *,
        train: pl.DataFrame,
        test: pl.DataFrame,
        name_mapping: binning.NameMapping,
        settings: config.Settings,
        n_iter: int = 20,
    ) -> Self:
        """Crea la selección y entrena el entrenador sobre el entrenamiento.

        ## Args:
            train: Conjunto de entrenamiento con las variables transformadas.
            test: Conjunto de prueba con las variables transformadas.
            name_mapping: Mapeo entre nombres originales y finales.
            settings: Configuración general del proyecto.
            n_iter: Número de iteraciones para la optimización.

        ## Returns:
            Instancia con el entrenador ya entrenado y los conjuntos asociados.
        """
        trainer = training.LGBMTrainer.fit_and_build(
            train,
            name_mapping.snake_case_names,
            settings,
            n_iter=n_iter,
        )
        return cls(train, test, name_mapping, trainer, settings, n_iter)

    @property
    def snake_case_names(self) -> tuple[str, ...]:
        """Nombres finales de las variables seleccionadas.

        ## Returns:
            Nombres en snake_case definidos por el mapeo.
        """
        return self.name_mapping.snake_case_names

    def plot_bivariate_analysis(
        self, analysis_name: str, max_bins_quantity: int = 20
    ) -> None:
        """Construye tablas de análisis bivariado y genera los gráficos.

        ## Args:
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.
        """
        visualization.plot_bivariate_charts(
            bivariate_table.build_bivariate_tables(
                self.train, self.name_mapping, max_bins_quantity, settings=self.settings
            ),
            analysis_name,
            settings=self.settings,
        )

    def plot_top_features(self, plot_name: str) -> None:
        """Genera el gráfico con las variables más importantes del modelo.

        ## Args:
            plot_name: Nombre del archivo de salida para el gráfico.
        """
        self.trainer.plot_top_features(plot_name, self.name_mapping)

    def evaluate_and_plot_metrics(
        self, *, roc_curve_name: str, deciles_name: str
    ) -> None:
        """Evalúa el mejor modelo sobre el set de prueba y genera los gráficos.

        Calcula las probabilidades predichas para entrenamiento y prueba, obtiene
        las métricas globales y las tablas de deciles, y genera el gráfico de la
        curva ROC y el gráfico de deciles con los nombres indicados.

        ## Args:
            test: Datos de prueba que incluyen la variable target.
            roc_curve_name: Nombre del archivo de salida SVG de la curva ROC,
                sin extensión.
            deciles_name: Nombre del archivo de salida SVG de la tabla de
                análisis por deciles, sin extensión.
        """
        best_model = self.trainer.best_model
        X_train = self.train.select(self.snake_case_names)
        X_test = self.test.select(self.snake_case_names)
        probabilities_train = cast(np.ndarray, best_model.predict_proba(X_train))[:, 1]
        probabilities_test = cast(np.ndarray, best_model.predict_proba(X_test))[:, 1]

        evaluation_metrics = evaluation.evaluate(
            train=self.train,
            test=self.test,
            probabilities_train=probabilities_train,
            probabilities_test=probabilities_test,
            settings=self.settings,
        )

        visualization.plot_roc_curve(evaluation_metrics, roc_curve_name, self.settings)
        visualization.plot_deciles(
            evaluation_metrics, deciles_name, settings=self.settings
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BivariateAnalyzer:
    """Analizador bivariado sobre conjuntos con variables correlacionadas.

    Conserva los conjuntos previos a la eliminación por correlación, lo que
    permite comparar variables descartadas con sus equivalentes conservadas
    y definir la codificación por bines.

    ## Attributes:
        correlated_train: Conjunto de entrenamiento previo al filtrado por
            correlación.
        correlated_test: Conjunto de prueba previo al filtrado por correlación.
        settings: Configuración general del proyecto.
    """

    correlated_train: pl.DataFrame
    correlated_test: pl.DataFrame
    settings: config.Settings

    @classmethod
    def from_redundant_column_filter(
        cls, column_filter: redundant_column_filter.RedundantColumnFilter
    ) -> Self:
        """Crea un analizador a partir del filtro de columnas redundantes.

        ## Args:
            column_filter: Filtro con los conjuntos con variables
                correlacionadas y la configuración.

        ## Returns:
            Instancia inicializada con los conjuntos con variables correlacionadas.
        """
        return cls(
            column_filter.correlated_train,
            column_filter.correlated_test,
            column_filter.settings,
        )

    def _build_bivariate_tables(
        self, name_mapping: binning.NameMapping, max_bins_quantity: int
    ) -> tuple[bivariate_table.BivariateTable, ...]:
        """Construye las tablas de análisis bivariado para las variables indicadas.

        ## Args:
            name_mapping: Mapeo entre nombres originales y finales.
            max_bins_quantity: Cantidad máxima de bines por variable.

        ## Returns:
            Tablas de análisis bivariado calculadas sobre el conjunto de entrenamiento.
        """
        return bivariate_table.build_bivariate_tables(
            self.correlated_train,
            name_mapping,
            max_bins_quantity,
            settings=self.settings,
        )

    def plot_bivariate(
        self,
        name_mapping: binning.NameMapping,
        analysis_name: str,
        max_bins_quantity: int = 20,
    ) -> None:
        """Genera gráficos de análisis bivariado.

        Utiliza el conjunto previo a la eliminación por correlación, lo que permite
        comparar variables descartadas con sus equivalentes conservadas y evaluar
        posibles reemplazos más interpretables.

        ## Args:
            name_mapping: Mapeo entre nombres originales y finales.
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.
        """
        visualization.plot_bivariate_charts(
            self._build_bivariate_tables(name_mapping, max_bins_quantity),
            analysis_name,
            settings=self.settings,
        )

    def apply_binning(
        self,
        binners: Iterable[binning.Binner],
        max_bins_quantity: int = 20,
    ) -> FeaturesSelection:
        """Aplica transformaciones por bines a los conjuntos de entrenamiento y prueba.

        Reemplaza los valores originales de cada columna indicada por el porcentaje
        de target del grupo de bines al que pertenecen. La misma lógica se aplica
        en forma consistente a ambos conjuntos de entrenamiento y prueba.

        Conserva únicamente las columnas identificadora y target junto con las
        variables indicadas en el mapeo de renombres, tanto en entrenamiento
        como en prueba.

        ## Args:
            binners: Asociaciones entre columnas con sus intervalos de bines
                que definen cada transformación.
            max_bins_quantity: Cantidad máxima de bines por variable para el
                cálculo de las tablas de análisis bivariado.

        ## Returns:
            Selección final con los conjuntos de entrenamiento y prueba
            transformados, el mapeo de nombres y el entrenador ya entrenado.
        """
        name_mapping = binning.NameMapping.from_dual_named(binners)

        bivariate_tables = self._build_bivariate_tables(name_mapping, max_bins_quantity)

        expr = [
            binner.build_encoding_expr(bivariate_table.table, self.settings)
            for binner, bivariate_table in zip(binners, bivariate_tables, strict=True)
        ]

        final_cols = [
            self.settings.dataset.id,
            self.settings.dataset.target,
            *name_mapping.snake_case_names,
        ]

        final_train = self.correlated_train.with_columns(expr).select(final_cols)
        final_test = self.correlated_test.with_columns(expr).select(final_cols)

        return FeaturesSelection.fit_and_build(
            train=final_train,
            test=final_test,
            name_mapping=name_mapping,
            settings=self.settings,
        )
