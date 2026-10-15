import dataclasses

import polars as pl

from bank_clients_ml import binning, config


@dataclasses.dataclass(frozen=True, slots=True)
class BivariateTable:
    snake_case_name: str
    pretty_name: str
    table: pl.DataFrame

    @property
    def title_for_plot(self):
        return f"Variable analysis: {self.pretty_name}"

    def with_rounded_int_column(self, column: str) -> pl.DataFrame:
        return self.table.with_columns(pl.col(column).round().cast(pl.Int64))


def build_bivariate_tables(
    df: pl.DataFrame,
    name_mapping: binning.NameMapping,
    max_bins_quantity: int,
    settings: config.Settings,
) -> tuple[BivariateTable, ...]:
    """Construye tablas de análisis bivariado por variable.

    ## Args:
        df: DataFrame con la variable target y las columnas a analizar.
        Cada fila representa a un cliente
        columns: Columnas para las cuales se generan las tablas de análisis bivariado.
        max_bins_quantity: Cantidad máxima de bines por variable. Para cada
            columna, se generan bines cuando su cardinalidad supera este máximo
            permitido, o se utilizan los valores originales en caso contrario.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    ## Returns:
        Tablas para análisis bivariado con las columnas de: bin, mínimo, máximo,
        clientes, target y porcentaje de target, numerando los bines en forma
        secuencial.
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
