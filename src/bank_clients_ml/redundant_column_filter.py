from typing import NamedTuple

import marimo as mo
import numpy as np
import polars as pl

from bank_clients_ml.config import Settings, get_settings
from bank_clients_ml.eda import low_cardinality_value_counts
from bank_clients_ml.visualization import generate_bivariate_charts


def _get_true_column_names(df: pl.DataFrame) -> list[str]:
    """Retorna los nombres de las columnas que contienen valores verdaderos"""
    return df.unpivot().filter(pl.col("value")).get_column("variable").to_list()


def _get_constant_columns(df: pl.DataFrame) -> list[str]:
    return _get_true_column_names(df.select(pl.all().n_unique() == 1))


def _get_imbalanced_binary_columns(
    df: pl.DataFrame,
    threshold: float,
    settings: Settings | None = None,
) -> list[str]:
    if settings is None:
        settings = get_settings()

    c_threshold = 1.0 - threshold
    is_binary = pl.all().n_unique() == 2
    first_val_ratio = (pl.all() == pl.all().first()).mean()
    is_imbalanced = (first_val_ratio < threshold) | (first_val_ratio > c_threshold)

    return _get_true_column_names(
        df.drop(settings.col_target).select(is_binary & is_imbalanced)
    )


def _get_redundant_correlated_columns(corr_df, threshold: float) -> list[str]:
    """Deja siempre la primera columna fuera de la lista.
    Si N columnas están correlacionadas entre sí, devolverá N-1 en la lista.

    El triángulo inferior y la diagonal quedan en 0.0,
    lo que no afecta al max() ya que |r| >= 0"""
    abs_corr_np = np.abs(corr_df.to_numpy())
    upper_triangle = np.triu(abs_corr_np, k=1)

    maximums_per_column = upper_triangle.max(axis=0)
    correlated_columns = [
        col
        for col, max_val in zip(corr_df.columns, maximums_per_column, strict=True)
        if max_val > threshold
    ]
    return correlated_columns


def _get_bivariate_tables(
    df: pl.DataFrame,
    columns: list[str],
    max_bins_quantity: int,
    settings: Settings | None = None,
) -> dict[str, pl.DataFrame]:
    """
    Parámetros:
    df: cada fila representa a un cliente,
    max_bins_quantity: Cantidad maxima de bins en los que se puede dividir cada variable.
    """
    if settings is None:
        settings = get_settings()

    target_pct_col = f"{settings.col_target}_pct"
    tables: dict[str, pl.DataFrame] = {}

    for column in columns:
        if df[column].n_unique() > max_bins_quantity:
            group_expr = pl.col(column).qcut(
                quantiles=max_bins_quantity, allow_duplicates=True
            )
        else:
            group_expr = pl.col(column)

        tables[column] = (
            df.select(settings.col_target, column)
            .group_by(group_expr.alias("_bin"))
            .agg(
                pl.col(column).min().round(2).alias("Min"),
                pl.col(column).max().round(2).alias("Max"),
                pl.len().alias("Clients"),
                pl.col(settings.col_target).sum().alias(settings.col_target),
            )
            .sort(by="Min", descending=False, nulls_last=True)
            .with_columns(
                pl.int_range(1, pl.len() + 1).alias("Bin"),
                ((pl.col(settings.col_target) / pl.col("Clients")) * 100)
                .round()
                .cast(pl.Int64)
                .alias(target_pct_col),
            )
            .select("Bin", "Min", "Max", "Clients", settings.col_target, target_pct_col)
        )

    return tables


def _merge_without_duplicates(
    dict_1: dict[str, pl.DataFrame], dict_2: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    common_keys = dict_1.keys() & dict_2.keys()
    if common_keys:
        raise KeyError(f"Claves duplicadas detectadas: {list(common_keys)}")

    return dict_1 | dict_2


def _get_range_data(i: int, stats) -> tuple[float, float, float]:
    """Función auxiliar para extraer bounds y valor por índice"""
    min_val = stats[f"min_{i}"]
    max_val = stats[f"max_{i}"]
    cli_sum = stats[f"cli_{i}"] or 0
    tgt_sum = stats[f"tgt_{i}"] or 0

    low = (float(min_val) - 0.01) if isinstance(min_val, (int, float)) else 0.0
    high = (float(max_val) + 0.01) if isinstance(max_val, (int, float)) else 0.0
    val = (float(tgt_sum) / float(cli_sum) * 100.0) if cli_sum > 0 else 0.0
    return low, high, val


class BinRange(NamedTuple):
    start: float
    end: float


def _group_bins_by_ranges(
    column: str,
    bin_ranges: list[BinRange],
    table: pl.DataFrame,
    settings: Settings | None = None,
) -> pl.Expr:
    """Construye una cadena de expresiones de Polars para agrupar
    valores en bins definidos por rangos de Bins (bin_min, bin_max),
    calculando dinámicamente los valores por rango y el valor por defecto desde la tabla resumen.

    Args:
        column: Nombre de la columna a agrupar.
        ranges: Lista de tuplas con los límites de los rangos (bin_min, bin_max).
        table: DataFrame de Polars con las columnas 'Bin', 'Min', 'Max', 'Clients'
        y 'settings.col_target'.

    Returns:
        pl.Expr: Expresión de Polars con el agrupamiento aplicado.
    """
    if settings is None:
        settings = get_settings()

    range_conditions = [
        (pl.col("Bin") >= b_min) & (pl.col("Bin") <= b_max)
        for b_min, b_max in bin_ranges
    ]

    aggregations = []
    for i, cond in enumerate(range_conditions):
        aggregations.extend(
            [
                pl.col("Min").filter(cond).min().alias(f"min_{i}"),
                pl.col("Max").filter(cond).max().alias(f"max_{i}"),
                pl.col("Clients").filter(cond).sum().alias(f"cli_{i}"),
                pl.col(settings.col_target).filter(cond).sum().alias(f"tgt_{i}"),
            ]
        )

    out_of_range_cond = ~pl.any_horizontal(range_conditions)
    aggregations.extend(
        [
            pl.col("Clients").filter(out_of_range_cond).sum().alias("cli_def"),
            pl.col(settings.col_target)
            .filter(out_of_range_cond)
            .sum()
            .alias("tgt_def"),
        ]
    )

    stats = table.select(aggregations).row(0, named=True)
    cli_def = stats["cli_def"] or 0
    tgt_def = stats["tgt_def"] or 0
    default_val = (float(tgt_def) / float(cli_def) * 100.0) if cli_def > 0 else 0.0

    expr_col = pl.col(column)

    low, high, val = _get_range_data(0, stats)
    expr = pl.when(expr_col.is_between(low, high)).then(val)

    for i in range(1, len(bin_ranges)):
        low, high, val = _get_range_data(i, stats)
        expr = expr.when(expr_col.is_between(low, high)).then(val)

    return expr.otherwise(default_val)


class BinTransformation(NamedTuple):
    column: str
    bin_ranges: list[BinRange]


class RedundantColumnFilter:
    def __init__(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        imbalanced_binary_threshold: float = 0.10,
        correlation_threshold: float = 0.80,
        settings: Settings | None = None,
    ):
        if settings is None:
            self.settings = get_settings()
        else:
            self.settings = settings

        self.constant_cols = _get_constant_columns(train)
        self.reduced_train = train.drop(self.constant_cols)
        reduced_test = test.drop(self.constant_cols)
        self.imbalanced_binary_columns = _get_imbalanced_binary_columns(
            self.reduced_train, imbalanced_binary_threshold
        )
        self.correlated_train = self.reduced_train.drop(self.imbalanced_binary_columns)
        self.correlated_test = reduced_test.drop(self.imbalanced_binary_columns)
        self.corr_df = self.correlated_train.drop(
            self.settings.col_id, self.settings.col_target
        ).corr()
        to_delete = _get_redundant_correlated_columns(
            self.corr_df, correlation_threshold
        )
        self.uncorrelated_train = self.correlated_train.drop(to_delete)
        self.uncorrelated_test = self.correlated_test.drop(to_delete)
        self.train_analysis: dict[str, pl.DataFrame] = {}

    def print_constant_cols(self) -> None:
        mo.output.append(mo.md("### constant_cols:"))
        mo.output.append(self.constant_cols)
        mo.output.append(
            f"train sin columnas con valores únicos: {self.reduced_train.shape}"
        )

    def print_imbalanced_binary_columns(self) -> None:
        mo.output.append(mo.md("### imbalanced_binary_columns:"))
        mo.output.append(
            low_cardinality_value_counts(
                self.reduced_train.select(self.imbalanced_binary_columns)
            )
        )
        mo.output.append(
            mo.md(
                f"train sin columnas binarias poco representativas: {self.correlated_train.shape}"
            )
        )

    def get_uncorrelated(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        return (self.uncorrelated_train, self.uncorrelated_test)

    def plot_uncorrelated(
        self, columns, analysis_name, max_bins_quantity: int = 20
    ) -> None:
        temp_table = _get_bivariate_tables(
            self.uncorrelated_train, columns, max_bins_quantity, self.settings
        )
        self.train_analysis = _merge_without_duplicates(self.train_analysis, temp_table)
        generate_bivariate_charts(temp_table, analysis_name, settings=self.settings)

    def plot_correlated(
        self, correlated_columns, analysis_name, max_bins_quantity: int = 20
    ) -> None:
        if any(col in self.uncorrelated_train.columns for col in correlated_columns):
            raise RuntimeError(
                "correlated_columns posee columnas dentro de uncorrelated_train"
            )

        temp_table = _get_bivariate_tables(
            self.correlated_train,
            correlated_columns,
            max_bins_quantity,
            settings=self.settings,
        )
        self.train_analysis = _merge_without_duplicates(self.train_analysis, temp_table)
        generate_bivariate_charts(temp_table, analysis_name, settings=self.settings)

    def plot_specific(
        self, df, columns, analysis_name, max_bins_quantity: int = 20
    ) -> None:
        temp_tables = _get_bivariate_tables(
            df, columns, max_bins_quantity, settings=self.settings
        )
        generate_bivariate_charts(temp_tables, analysis_name, settings=self.settings)

    def _get_correlations_for(
        self, column: str, threshold: float = 0.80
    ) -> pl.DataFrame:
        return (
            self.corr_df.select(
                pl.Series("feature_name", self.corr_df.columns),
                pl.col(column).alias("correlation"),
            )
            .filter(
                (pl.col("feature_name") != column)
                & (pl.col("correlation").abs() > threshold)
            )
            .sort(pl.col("correlation").abs(), descending=True)
        )

    def print_correlations_for_each(self, columns) -> None:
        for column in columns:
            mo.output.append(mo.md(f"###  Columnas correlacionadas con {column}:"))
            mo.output.append(self._get_correlations_for(column))

    def group_bins_by_ranges(
        self, bins_transformations: list[BinTransformation]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        expr = [
            _group_bins_by_ranges(
                column,
                bin_ranges=bin_ranges,
                table=self.train_analysis[column],
            ).alias(column)
            for column, bin_ranges in bins_transformations
        ]
        final_train = self.correlated_train.with_columns(expr)
        final_test = self.correlated_test.with_columns(expr)

        return final_train, final_test
