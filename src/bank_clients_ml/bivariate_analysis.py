import dataclasses
from collections.abc import Iterable
from typing import Self

import polars as pl

from bank_clients_ml import (
    binning,
    bivariate_table,
    config,
    redundant_column_filter,
    training,
    visualization,
)


@dataclasses.dataclass(frozen=True, slots=True)
class FeaturesSelection:
    """Selección de variables para el modelado.

    Agrupa los conjuntos de entrenamiento y prueba ya filtrados con el
    mapeo de renombres y la lista de variables seleccionadas.

    ## Attributes:
        train: Conjunto de entrenamiento con las columnas identificadora,
            target y mejores variables.
        test: Conjunto de prueba con las columnas identificadora,
            target y mejores variables.
        features_name_mapping: Mapeo entre el nombre original de cada variable
            seleccionada y su nombre final.
        features_names: Nombres originales de las variables seleccionadas.
    """

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
        trainer = training.LGBMTrainer.fit_and_build(
            train,
            name_mapping.snake_case_names,
            settings,
            n_iter=n_iter,
        )
        return cls(test, name_mapping, trainer, settings, n_iter)

    @property
    def train(self) -> pl.DataFrame:
        return self.trainer.train

    @property
    def snake_case_names(self) -> tuple[str, ...]:
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
        self.trainer.plot_top_features(plot_name, self.name_mapping)

    def evaluate_and_plot_metrics(
        self, *, roc_curve_name: str, deciles_name: str
    ) -> None:
        self.trainer.evaluate_and_plot_metrics(
            self.test, roc_curve_name=roc_curve_name, deciles_name=deciles_name
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BivariateAnalyzer:
    correlated_train: pl.DataFrame
    correlated_test: pl.DataFrame
    settings: config.Settings

    @classmethod
    def from_redundant_column_filter(
        cls, col_filter: redundant_column_filter.RedundantColumnFilter
    ) -> Self:
        return cls(
            col_filter.correlated_train, col_filter.correlated_test, col_filter.settings
        )

    def _build_bivariate_tables(
        self, name_mapping: binning.NameMapping, max_bins_quantity: int
    ) -> tuple[bivariate_table.BivariateTable, ...]:
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
            columns: Columnas a analizar.
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
        de target del grupo de bines al que pertenecen, según el análisis bivariado
        previamente acumulado. La misma lógica se aplica en forma consistente a
        ambos conjuntos de entrenamiento y prueba.

        Conserva únicamente las columnas identificadora y target junto con las
        variables indicadas en el mapeo de renombres, tanto en entrenamiento
        como en prueba.

        ## Args:
            binners: Lista con las columnas con los rangos de bines que
                definen cada transformación.
            features_name_mapping: Mapeo entre el nombre original de cada variable
                seleccionada y su nombre final.

        ## Returns:
            Selección final con los conjuntos de entrenamiento y prueba
            transformados y filtrados, el mapeo de renombres y la lista de
            mejores variables.
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
