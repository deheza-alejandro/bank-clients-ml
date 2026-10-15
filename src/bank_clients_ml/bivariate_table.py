"""Construcción de tablas para análisis bivariado entre variables y el target.

Las tablas resultantes sirven como base para el análisis exploratorio y las
visualizaciones.

## Clases exportadas:
    BivariateTable: Contenedor inmutable con la tabla de análisis bivariado de una variable.
## Funciones exportadas:
    build_bivariate_tables: Construye tablas de análisis bivariado para las variables indicadas.
"""

import dataclasses

import polars as pl

from bank_clients_ml import binning, config


@dataclasses.dataclass(frozen=True, slots=True)
class BivariateTable:
    """Contenedor inmutable con la tabla de análisis bivariado de una variable.

    Agrupa el resultado del análisis bivariado junto con los nombres
    en snake_case y legible de la variable para su presentación.

    ## Attributes:
        snake_case_name: Nombre de la variable en formato snake_case.
        pretty_name: Nombre legible de la variable para mostrar en tablas y gráficos.
        table: Tabla de análisis bivariado con las columnas de bin, mínimo, máximo,
            clientes, target y porcentaje de target.
    """

    snake_case_name: str
    pretty_name: str
    table: pl.DataFrame

    @property
    def title_for_plot(self):
        """Título estandarizado para el gráfico de análisis de la variable."""
        return f"Variable analysis: {self.pretty_name}"

    def with_rounded_int_column(self, column: str) -> pl.DataFrame:
        """Devuelve una copia de la tabla con la columna indicada redondeada a int.

        ## Args:
            column: Nombre de la columna numérica a redondear.

        ## Returns:
            Copia de la tabla análisis bivariado con la columna redondeada y
            convertida a int.
        """
        return self.table.with_columns(pl.col(column).round().cast(pl.Int64))


def build_bivariate_tables(
    df: pl.DataFrame,
    name_mapping: binning.NameMapping,
    max_bins_quantity: int,
    settings: config.Settings,
) -> tuple[BivariateTable, ...]:
    """Construye tablas de análisis bivariado para las variables indicadas.

    Agrupa cada columna por sus valores originales o por bines
    cuando su cardinalidad supera el máximo permitido, y calcula por grupo
    los clientes, el target acumulado y su porcentaje. Los bines se numeran
    en forma secuencial ordenados por el valor mínimo del grupo.

    ## Args:
        df: DataFrame con la columna target y las variables a analizar,
            donde cada fila representa a un cliente.
        name_mapping: Mapeo entre nombres snake_case y nombres legibles de las variables.
        max_bins_quantity: Cantidad máxima de bines por variable. Si la cardinalidad
            de la columna supera este valor se generan bines, en caso
            contrario se conservan los valores originales.
        settings: Configuración con los nombres de las columnas de target,
            bin, mínimo, máximo, clientes y porcentaje de target.

    ## Returns:
        Tablas para análisis bivariado de cada variable con las columnas de bin, mínimo,
            máximo, clientes, target y porcentaje de target.
    """
    bivariate_tables: list[BivariateTable] = []

    for column in name_mapping.snake_case_names:
        if df.get_column(column).n_unique() > max_bins_quantity:
            group_expr = pl.col(column).qcut(
                quantiles=max_bins_quantity, allow_duplicates=True
            )
        else:
            group_expr = pl.col(column)

        table = (
            df.select(settings.dataset.target, column)
            .group_by(group_expr.alias("_bin"))
            .agg(
                pl.col(column).min().alias(settings.analysis.min),
                pl.col(column).max().alias(settings.analysis.max),
                pl.len().alias(settings.analysis.clients),
                pl.col(settings.dataset.target).sum().alias(settings.dataset.target),
            )
            .sort(by=settings.analysis.min, descending=False, nulls_last=True)
            .with_columns(
                pl.int_range(1, pl.len() + 1).alias(settings.analysis.bin),
                (
                    (
                        pl.col(settings.dataset.target)
                        / pl.col(settings.analysis.clients)
                    )
                    * 100
                ).alias(settings.analysis.target_pct),
            )
            .select(
                settings.analysis.bin,
                settings.analysis.min,
                settings.analysis.max,
                settings.analysis.clients,
                settings.dataset.target,
                settings.analysis.target_pct,
            )
        )
        bivariate_tables.append(BivariateTable(column, name_mapping.get(column), table))

    return tuple(bivariate_tables)
