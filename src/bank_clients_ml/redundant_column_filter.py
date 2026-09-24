"""Filtrado de columnas redundantes y transformación por bines con Polars.

Este módulo implementa el proceso de selección de variables utilizado en el notebook
para reducir la dimensionalidad antes del modelado. Elimina en forma secuencial
columnas constantes, columnas binarias desbalanceadas y columnas numéricas
altamente correlacionadas.

El flujo habitual consiste en crear `RedundantColumnFilter` con los conjuntos de
entrenamiento y prueba, inspeccionar los descartes, graficar el análisis bivariado
y finalmente aplicar transformaciones de bines con `apply_bin_transformations`.

Clases exportadas:
    BinRange: Rango inclusivo de bines que se agrupan en un mismo valor.
    BinTransformation: Asociación entre una columna y sus rangos de bines.
    RedundantColumnFilter: Filtro secuencial de columnas redundantes con análisis bivariado.
"""

from typing import NamedTuple

import marimo as mo
import numpy as np
import polars as pl

from bank_clients_ml.config import Settings, get_settings
from bank_clients_ml.eda import low_cardinality_value_counts
from bank_clients_ml.visualization import plot_bivariate_charts


def _get_true_column_names(df: pl.DataFrame) -> list[str]:
    """Obtiene los nombres de las columnas que contienen valores verdaderos.

    Args:
        df: DataFrame booleano donde cada celda indica si una columna cumple
            una condición evaluada previamente.

    Returns:
        Lista con los nombres de las columnas cuyo valor es verdadero.
    """
    return df.unpivot().filter(pl.col("value")).get_column("variable").to_list()


def _get_constant_columns(df: pl.DataFrame) -> list[str]:
    """Retorna las columnas con un único valor.

    Args:
        df: DataFrame sobre el que se evalúa la cardinalidad.

    Returns:
        Lista con los nombres de las columnas constantes, las cuales no aportan
        información para el modelado.
    """
    return _get_true_column_names(df.select(pl.all().n_unique() == 1))


def _get_imbalanced_binary_columns(
    df: pl.DataFrame,
    threshold: float,
    settings: Settings | None = None,
) -> list[str]:
    """Retorna las columnas binarias desbalanceadas.

    Detecta las columnas con exactamente dos valores únicos. La proporción se
    calcula contra el primer valor de cada columna y se considera desbalanceada
    cuando cae fuera del intervalo definido por el `threshold` y su complemento.
    La columna target se excluye del análisis.

    Args:
        df: DataFrame sobre el que se buscan columnas binarias desbalanceadas.
        threshold: Proporción mínima aceptada para la clase minoritaria. Debe
            encontrarse entre 0 y 0.5.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    Returns:
        Lista con los nombres de las columnas binarias desbalanceadas,
        las cuales no aportan información para el modelado.
    """
    if settings is None:
        settings = get_settings()

    c_threshold = 1.0 - threshold
    is_binary = pl.all().n_unique() == 2
    first_val_ratio = (pl.all() == pl.all().first()).mean()
    is_imbalanced = (first_val_ratio < threshold) | (first_val_ratio > c_threshold)

    return _get_true_column_names(
        df.drop(settings.col_target).select(is_binary & is_imbalanced)
    )


def _get_redundant_correlated_columns(
    corr_df: pl.DataFrame, threshold: float
) -> list[str]:
    """Identifica columnas numéricas redundantes por correlación.

    Calcula el valor absoluto de la matriz de correlación y conserva el triángulo
    superior para evaluar cada columna únicamente contra las columnas previas.
    Toda columna cuyo máximo de correlación supere el umbral se considera redundante.

    Deja siempre la primer columna fuera de la lista.
    Si N columnas están correlacionadas entre sí, devolverá N-1 en la lista.
    El triángulo inferior y la diagonal quedan en 0.0,
    lo que no afecta al max() ya que |r| >= 0

    Args:
        corr_df: Matriz de correlación cuadrada entre variables numéricas.
        threshold: Umbral de correlación absoluta a partir del cual una columna
            se considera redundante.

    Returns:
        Lista con los nombres de las columnas redundantes a eliminar.
    """
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
    """Construye tablas de análisis bivariado por variable.

    Args:
        df: DataFrame con la variable target y las columnas a analizar.
        Cada fila representa a un cliente
        columns: Columnas para las cuales se generan las tablas de análisis bivariado.
        max_bins_quantity: Cantidad máxima de bines por variable. Para cada
            columna, se generan bines cuando su cardinalidad supera este máximo
            permitido, o se utilizan los valores originales en caso contrario.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    Returns:
        Diccionario que asocia cada nombre de columna con su tabla de análisis
        bivariado, con las columnas de bin, mínimo, máximo, clientes, target
        y porcentaje de target, numerando los bines en forma secuencial.
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
    """Combina diccionarios sin claves repetidas.

    Verifica que no existan nombres de claves comunes antes de combinar los
    diccionarios, para preservar el análisis ya almacenado y evitar
    sobrescrituras accidentales.

    Args:
        dict_1: Diccionario base con las tablas ya acumuladas.
        dict_2: Diccionario con las nuevas tablas a incorporar.

    Returns:
        Nuevo diccionario con la unión de ambos contenidos.

    Raises:
        KeyError: Si ambos diccionarios comparten una o más claves.
    """
    common_keys = dict_1.keys() & dict_2.keys()
    if common_keys:
        raise KeyError(
            f"Cannot merge tables to save analysis before plot: "
            f"Duplicate {len(common_keys)} keys on merge:"
            f"\n{sorted(common_keys)}"
        )

    return dict_1 | dict_2


def _get_range_data(i: int, stats) -> tuple[float, float, float]:
    """Calcula los límites y el valor de reemplazo de un rango de bines.

    Expande levemente los valores mínimo y máximo observados para garantizar
    la inclusión de los bordes, y calcula el porcentaje de target del rango
    como la proporción entre la suma del target y la cantidad de clientes.

    Args:
        i: Índice del rango dentro de las agregaciones calculadas.
        stats: Fila de agregaciones con los valores mínimos, máximos, conteos
            de clientes y sumas del target por rango.

    Returns:
        Tupla con el límite inferior, el límite superior y el porcentaje de
        target correspondiente al rango indicado.
    """
    min_val = stats[f"min_{i}"]
    max_val = stats[f"max_{i}"]
    cli_sum = stats[f"cli_{i}"] or 0
    tgt_sum = stats[f"tgt_{i}"] or 0

    low = (float(min_val) - 0.01) if isinstance(min_val, (int, float)) else 0.0
    high = (float(max_val) + 0.01) if isinstance(max_val, (int, float)) else 0.0
    val = (float(tgt_sum) / float(cli_sum) * 100.0) if cli_sum > 0 else 0.0
    return low, high, val


class BinRange(NamedTuple):
    """Rango inclusivo de bines que se agrupan en un mismo valor.

    Representa un intervalo cerrado sobre la numeración de bines generada por
    el análisis bivariado. Los bines comprendidos entre ambos extremos reciben
    el mismo porcentaje de target durante la transformación.

    Attributes:
        start: Número de bin inicial del intervalo.
        end: Número de bin final del intervalo.
    """

    start: float
    end: float


def _group_bins_by_ranges(
    column: str,
    bin_ranges: list[BinRange],
    table: pl.DataFrame,
    settings: Settings | None = None,
) -> pl.Expr:
    """Construye una expresión de transformación por rangos de bines.

    Agrupa las estadísticas de la tabla de análisis bivariado según los rangos
    indicados y calcula el porcentaje de target de cada grupo. Los valores
    fuera de todos los rangos reciben el porcentaje del grupo residual.

    Args:
        column: Columna a transformar.
        bin_ranges: Rangos de bines que definen cada grupo de agregación.
        table: Tabla de análisis bivariado de la columna, con las columnas de
            bin, mínimo, máximo, clientes y target.
        settings: Configuración con el nombre de la columna target. Si no se
            indica, se obtiene la configuración global.

    Returns:
        Expresión de Polars que asigna a cada fila el porcentaje de target
        del rango al que pertenece.
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
    """Asociación entre una columna y sus rangos de bines.

    Define cómo deben combinarse los bines de una variable analizada para
    reemplazar sus valores originales por el porcentaje de target de cada grupo.

    Attributes:
        column: Nombre de la columna a transformar.
        bin_ranges: Rangos de bines que conforman cada grupo.
    """

    column: str
    bin_ranges: list[BinRange]


class RedundantColumnFilter:
    """Filtro secuencial de columnas redundantes con análisis bivariado.

    Aplica tres etapas de reducción sobre los conjuntos de entrenamiento y prueba:
    eliminación de columnas constantes, eliminación de columnas binarias
    desbalanceadas y eliminación de columnas numéricas altamente correlacionadas.
    Conserva los resultados intermedios para su inspección y acumula las tablas
    de análisis bivariado para usarlas posteriormente en la transformación por bines.

    Attributes:
        constant_cols: Nombres de las columnas constantes detectadas.
        reduced_train: Conjunto de entrenamiento sin columnas constantes.
        imbalanced_binary_columns: Nombres de las columnas binarias desbalanceadas.
        correlated_train: Conjunto de entrenamiento sin columnas desbalanceadas.
        correlated_test: Conjunto de prueba sin columnas desbalanceadas.
        uncorrelated_train: Conjunto de entrenamiento sin columnas correlacionadas.
        uncorrelated_test: Conjunto de prueba sin columnas correlacionadas.
    """

    def __init__(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        imbalanced_binary_threshold: float = 0.10,
        correlation_threshold: float = 0.80,
        settings: Settings | None = None,
    ) -> None:
        """Ejecuta las etapas de reducción.

        Detecta columnas constantes, columnas binarias desbalanceadas y columnas
        correlacionadas, y genera las versiones reducidas de los conjuntos de
        entrenamiento y prueba.

        Args:
            train: Conjunto de entrenamiento sobre el que se detectan las columnas
                a eliminar.
            test: Conjunto de prueba al que se aplican las mismas eliminaciones.
            imbalanced_binary_threshold: Proporción mínima aceptada para la clase
                minoritaria en columnas binarias.
            correlation_threshold: Umbral de correlación para considerar
                redundante una columna numérica.
            settings: Configuración con los nombres de las columnas identificadora
                y target. Si no se indica, se obtiene la configuración global.
        """
        self._settings = settings or get_settings()

        self.constant_cols = _get_constant_columns(train)
        self.reduced_train = train.drop(self.constant_cols)
        reduced_test = test.drop(self.constant_cols)
        self.imbalanced_binary_columns = _get_imbalanced_binary_columns(
            self.reduced_train, imbalanced_binary_threshold, self._settings
        )
        self.correlated_train = self.reduced_train.drop(self.imbalanced_binary_columns)
        self.correlated_test = reduced_test.drop(self.imbalanced_binary_columns)
        self._corr_df = self.correlated_train.drop(
            self._settings.col_id, self._settings.col_target
        ).corr()
        to_delete = _get_redundant_correlated_columns(
            self._corr_df, correlation_threshold
        )
        self.uncorrelated_train = self.correlated_train.drop(to_delete)
        self.uncorrelated_test = self.correlated_test.drop(to_delete)
        self._train_analysis: dict[str, pl.DataFrame] = {}

    def print_constant_cols(self) -> None:
        """Muestra las columnas constantes y la forma del conjunto reducido."""
        mo.output.append(mo.md("### constant_cols:"))
        mo.output.append(self.constant_cols)
        mo.output.append(f"train without constant_cols: {self.reduced_train.shape}")

    def print_imbalanced_binary_columns(self) -> None:
        """Muestra el conteo de las columnas binarias desbalanceadas."""
        mo.output.append(mo.md("### imbalanced_binary_columns:"))
        mo.output.append(
            low_cardinality_value_counts(
                self.reduced_train.select(self.imbalanced_binary_columns)
            )
        )
        mo.output.append(
            mo.md(
                f"train without imbalanced_binary_columns: {self.correlated_train.shape}"
            )
        )

    def _build_tables_and_plot(
        self,
        df: pl.DataFrame,
        columns: list[str],
        analysis_name: str,
        max_bins_quantity: int,
        save_analysis: bool = True,
    ) -> None:
        """Construye tablas de análisis bivariado y genera los gráficos.

        Args:
            df: DataFrame base para el análisis bivariado.
            columns: Columnas a analizar y graficar.
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.
            save_analysis: Indica si las tablas generadas se guardan para su
                uso posterior en transformaciones.
        """
        temp_tables = _get_bivariate_tables(
            df, columns, max_bins_quantity, settings=self._settings
        )
        if save_analysis:
            self._train_analysis = _merge_without_duplicates(
                self._train_analysis, temp_tables
            )
        plot_bivariate_charts(temp_tables, analysis_name, settings=self._settings)

    def plot_uncorrelated(
        self, columns: list[str], analysis_name: str, max_bins_quantity: int = 20
    ) -> None:
        """Genera gráficos de análisis bivariado de variables no correlacionadas.

        Utiliza el conjunto de entrenamiento sin columnas redundantes y guarda
        las tablas generadas para su reutilización en transformaciones por bines.

        Args:
            columns: Columnas del conjunto sin correlación a analizar.
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.
        """
        self._build_tables_and_plot(
            self.uncorrelated_train, columns, analysis_name, max_bins_quantity
        )

    def plot_correlated(
        self,
        correlated_columns: list[str],
        analysis_name: str,
        max_bins_quantity: int = 20,
    ) -> None:
        """Genera gráficos de análisis bivariado de variables correlacionadas eliminadas.

        Utiliza el conjunto previo a la eliminación por correlación, lo que permite
        comparar variables descartadas con sus equivalentes conservadas y evaluar
        posibles reemplazos más interpretables.

        Args:
            correlated_columns: Columnas eliminadas por correlación a analizar.
                No deben pertenecer al conjunto sin correlación.
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.

        Raises:
            ValueError: Si alguna columna indicada aún existe en el conjunto sin
                correlación.
        """
        overlapping = [
            col for col in correlated_columns if col in self.uncorrelated_train.columns
        ]

        if overlapping:
            raise ValueError(
                f"Correlated_columns must not overlap uncorrelated_train columns. "
                f"Expected only removed columns, got overlap:"
                f"\n{overlapping}"
            )

        self._build_tables_and_plot(
            self.correlated_train, correlated_columns, analysis_name, max_bins_quantity
        )

    def plot_adhoc(
        self,
        df: pl.DataFrame,
        columns: list[str],
        analysis_name: str,
        max_bins_quantity: int = 20,
    ) -> None:
        """Genera gráficos de análisis bivariado de forma puntual sin guardarlo.

        Permite visualizar variables de un DataFrame arbitrario, como el conjunto
        final transformado, sin modificar las tablas acumuladas para
        transformaciones posteriores.

        Args:
            df: DataFrame arbitrario a analizar.
            columns: Columnas a analizar y graficar.
            analysis_name: Nombre del análisis. Define el subdirectorio de salida
                de las imágenes.
            max_bins_quantity: Cantidad máxima de bines por variable.
        """
        self._build_tables_and_plot(
            df, columns, analysis_name, max_bins_quantity, save_analysis=False
        )

    def _get_correlations_for(
        self, column: str, threshold: float = 0.80
    ) -> pl.DataFrame:
        """Obtiene las variables correlacionadas con una columna dada.

        Args:
            column: Nombre de la columna de referencia presente en la matriz
                de correlación.
            threshold: Umbral mínimo de correlación para incluir una variable
                en el resultado.

        Returns:
            Nuevo DataFrame con las columnas de nombre de variable y correlación.
        """
        return (
            self._corr_df.select(
                pl.Series("feature_name", self._corr_df.columns),
                pl.col(column).alias("correlation"),
            )
            .filter(
                (pl.col("feature_name") != column)
                & (pl.col("correlation").abs() > threshold)
            )
            .sort(pl.col("correlation").abs(), descending=True)
        )

    def print_correlations_for_each(self, columns: list[str]) -> None:
        """Muestra las variables correlacionadas de cada columna indicada.

        Args:
            columns: Columnas de referencia para buscar correlaciones altas.
        """
        for column in columns:
            mo.output.append(mo.md(f"###  Columns correlated with {column}:"))
            mo.output.append(self._get_correlations_for(column))

    def apply_bin_transformations(
        self, bins_transformations: list[BinTransformation]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Aplica transformaciones por bines a los conjuntos de entrenamiento y prueba.

        Reemplaza los valores originales de cada columna indicada por el porcentaje
        de target del grupo de bines al que pertenecen, según el análisis bivariado
        previamente acumulado. La misma lógica se aplica en forma consistente a
        ambos conjuntos de entrenamiento y prueba.

        Args:
            bins_transformations: Lista con las columnas con los rangos de bines que
                definen cada transformación.

        Returns:
            Tupla con los conjuntos de entrenamiento y prueba transformados.
        """
        expr = [
            _group_bins_by_ranges(
                column,
                bin_ranges=bin_ranges,
                table=self._train_analysis[column],
                settings=self._settings,
            ).alias(column)
            for column, bin_ranges in bins_transformations
        ]
        final_train = self.correlated_train.with_columns(expr)
        final_test = self.correlated_test.with_columns(expr)

        return final_train, final_test
