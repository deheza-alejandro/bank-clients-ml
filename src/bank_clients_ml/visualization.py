"""Generación y guardado de gráficos para análisis y evaluación del modelo.

Este módulo centraliza las visualizaciones utilizadas en el notebook.
Todas las figuras se guardan como archivos SVG optimizados con SVGO a través de Bun.

## Funciones exportadas:
    plot_top_features: Genera un gráfico con las variables más importantes.
    plot_bivariate_charts: Genera gráficos bivariados para un conjunto de variables.
    plot_roc_curve: Genera una curva ROC con métricas de evaluación.
    plot_deciles: Genera un gráfico con tablas de deciles de entrenamiento y prueba.
"""

import concurrent.futures
import io
import pathlib
import subprocess
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.style
import polars as pl
import polars.selectors as cs
from matplotlib.figure import Figure

from bank_clients_ml import bivariate_table, config, evaluation

matplotlib.use("Agg")
matplotlib.style.use("default")
matplotlib.rcParams["svg.hashsalt"] = "fixed_seed_for_this_project"
matplotlib.rcParams["svg.fonttype"] = "path"


@contextmanager
def _closing_figure(fig: Figure) -> Generator[Figure]:
    """Gestiona una figura y garantiza su cierre al finalizar.

    Permite utilizar la figura dentro de un bloque y la libera
    aunque ocurra un error durante su manipulación.

    ## Args:
        fig: Figura a gestionar dentro del contexto.

    ## Yields:
        Figura recibida para su uso dentro del bloque.
    """
    try:
        yield fig
    finally:
        plt.close(fig)


def _save_fig_as_svg(
    fig: Figure,
    plot_name: str,
    images_dir: pathlib.Path,
    images_sub_dir: str = "",
) -> None:
    """Guarda una figura de Matplotlib como archivo SVG optimizado con SVGO.

    Canaliza el contenido SVG a través de SVGO mediante Bun para optimizarlo
    antes de escribirlo en disco. Crea la carpeta de destino si no existe.

    El resultado se exporta a:
    {images_dir}/{images_sub_dir}/{plot_name}.svg

    ## Args:
        fig: Figura de Matplotlib a guardar. Queda cerrada tras la operación.
        plot_name: Nombre base del archivo, sin extensión.
        images_dir: Directorio base donde se guardan las imágenes.
        images_sub_dir: Subdirectorio opcional dentro del directorio base.

    ## Raises:
        RuntimeError: Si el ejecutable de Bun no se encuentra en el PATH y no
            es posible optimizar el SVG con SVGO.
    """
    output_folder = images_dir / images_sub_dir
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{plot_name}.svg"

    with io.BytesIO() as buffer, _closing_figure(fig):
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        svg_data = buffer.getvalue()

    cmd = ["bun", "run", "svgo", "--multipass", "-i", "-", "-o", "-"]

    try:
        with svg_path.open("wb") as out_file:
            subprocess.run(  # noqa: S603
                cmd, input=svg_data, stdout=out_file, check=True
            )
    except FileNotFoundError as err:
        msg = (
            f"Bun executable not found in PATH. "
            f"Ensure Bun is installed and available in PATH to optimize SVG with SVGO: \n{svg_path}"
        )
        raise RuntimeError(msg) from err


def plot_top_features(
    variables_to_plot: pl.DataFrame,
    plot_name: str,
    *,
    roc_auc: float,
    top_n: int = 20,
    settings: config.Settings,
) -> None:
    """Genera un gráfico de barras horizontales con las variables más importantes.

    Selecciona las primeras filas del DataFrame de importancias, ya ordenado de
    forma descendente, y representa cada variable como una barra horizontal. El
    título incluye el nombre del gráfico y el valor de ROC AUC. Los tamaños de
    fuente y de figura se ajustan según la cantidad de variables a mostrar.

    El resultado se exporta a:
    {settings.paths.images}/plot_top_features/{plot_name}.svg

    ## Args:
        variables_to_plot: Tabla con las variables y sus importancias,
            ordenada de mayor a menor importancia.
        plot_name: Nombre base del archivo SVG a generar, sin extensión.
        roc_auc: Métrica ROC AUC a mostrar en el título del gráfico.
        top_n: Cantidad máxima de variables a incluir en el gráfico.
        settings: Configuración con los nombres de columnas de variable e importancia,
            y el directorio base donde se guarda la imagen.
            Si no se indica, se obtiene la configuración global.
    """
    top_n = min(top_n, variables_to_plot.height)

    top_vars = variables_to_plot.head(top_n)

    label_fontsize = min(35, top_n * 8)
    tick_fontsize = 30
    value_fontsize = 20

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.7)))

    bars = ax.barh(
        top_vars.get_column(settings.analysis.feature),
        top_vars.get_column(settings.analysis.importance),
        color="#4a90e2",
        edgecolor="black",
    )

    ax.invert_yaxis()
    ax.set_xlabel(settings.analysis.importance, fontsize=label_fontsize)
    ax.set_ylabel(settings.analysis.feature, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.bar_label(bars, fmt="%.0f", padding=5, fontsize=value_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    ax.set_title(
        f"{plot_name}: top {top_n} Features\nROC AUC: {roc_auc:.6f}",
        fontsize=label_fontsize,
    )

    _save_fig_as_svg(fig, plot_name, settings.paths.images, "plot_top_features")


def _format_floats(df: pl.DataFrame) -> pl.DataFrame:
    """Redondea a dos decimales todas las columnas de tipo float.

    ## Args:
        df: Datos de entrada con valores a redondear.

    ## Returns:
        Nuevos datos con las columnas de tipo float redondeadas a 2 decimales.
    """
    return df.with_columns(cs.float().round(2))


def _plot_single_bivariate_chart(
    bivariate_table: bivariate_table.BivariateTable,
    analysis_name: str,
    settings: config.Settings,
) -> None:
    """Genera el gráfico bivariado individual para una variable.

    Construye una figura con dos secciones: una tabla de análisis en la parte
    superior, y en la parte inferior un gráfico bivariado de barras con la cantidad de
    clientes por bin junto con una línea del porcentaje de la variable
    target en un eje secundario.

    El resultado se exporta a:
    {settings.paths.images}/bivariate_analysis/{analysis_name}/{variable_to_plot}.svg

    ## Args:
        bivariate_table: Análisis bivariado con el nombre de la variable
            analizada, las columnas de intervalo, conteo de clientes y porcentaje
            de la variable target. El nombre de la variable analizada se usa
            como título y como nombre del archivo generado.
        analysis_name: Nombre del análisis. Define el subdirectorio de salida.
        settings: Configuración con el nombre de la columna target y el directorio
            base donde se guarda la imagen. Si no se indica, se obtiene la
            configuración global.
    """
    formatted_table = _format_floats(
        bivariate_table.with_rounded_int_column(settings.analysis.target_pct)
    )

    fig, (ax_table, ax_graph) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1]}
    )

    ax_table.axis("off")
    ax_table.set_title(bivariate_table.title_for_plot, pad=1)
    ax_table.table(
        cellText=formatted_table.rows(),
        colLabels=formatted_table.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.99],
    )

    x_indices = range(len(formatted_table))

    ax_graph.bar(
        x_indices, formatted_table.get_column(settings.analysis.clients), width=0.35
    )
    ax_graph.set_ylabel(settings.analysis.clients)
    ax_graph.set_xticks(x_indices)
    ax_graph.set_xticklabels(
        formatted_table.get_column(settings.analysis.bin), rotation=0, ha="center"
    )

    ax_graph_target_pct = ax_graph.twinx()
    ax_graph_target_pct.plot(
        x_indices,
        formatted_table.get_column(settings.analysis.target_pct),
        marker="o",
        color="green",
    )
    ax_graph_target_pct.set_ylabel(f"{settings.dataset.target} pct (%)")

    fig.tight_layout()
    _save_fig_as_svg(
        fig,
        bivariate_table.snake_case_name,
        settings.paths.images,
        f"bivariate_analysis/{analysis_name}",
    )


def plot_bivariate_charts(
    bivariate_tables: Sequence[bivariate_table.BivariateTable],
    analysis_name: str,
    settings: config.Settings,
    max_workers: int | None = None,
) -> None:
    """Genera gráficos bivariados para un conjunto de variables.

    Recorre el diccionario de tablas de análisis y genera un gráfico por variable.
    Utiliza ejecución secuencial cuando se solicita un solo worker o cuando
    hay pocas tablas, y ejecución en paralelo con procesos separados en caso
    contrario para acelerar la generación de una gran cantidad de gráficos.

    Las imágenes se exportan a:
    {settings.paths.images}/bivariate_analysis/{analysis_name}/

    ## Args:
        bivariate_tables: Tablas de análisis bivariados a graficar.
        analysis_name: Nombre del análisis. Define el subdirectorio de salida.
        max_workers: Cantidad máxima de procesos en paralelo. Si no se indica,
            se usan todos los núcleos disponibles. si len(tables) < 20 se usa
            1 solo proceso. El valor 1 fuerza la ejecución secuencial.
        settings: Configuración con el nombre de la columna target y el
            directorio base donde se guardan las imágenes. Si no se indica,
            se obtiene la configuración global.
    """
    if max_workers == 1 or len(bivariate_tables) < 20 or max_workers is None:
        for bivariate_table in bivariate_tables:
            _plot_single_bivariate_chart(
                bivariate_table,
                analysis_name,
                settings,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
            futures = [
                executor.submit(
                    _plot_single_bivariate_chart,
                    bivariate_table,
                    analysis_name,
                    settings,
                )
                for bivariate_table in bivariate_tables
            ]

            for future in concurrent.futures.as_completed(futures):
                future.result()


def plot_roc_curve(
    evaluation_metrics: evaluation.EvaluationMetrics,
    plot_name: str,
    settings: config.Settings,
) -> None:
    """Genera una curva ROC con las métricas resumidas de evaluación.

    Dibuja la tasa de verdaderos positivos frente a la tasa de falsos positivos,
    junto con la línea de referencia diagonal de un clasificador aleatorio.
    Incluye una anotación con los valores de accuracy y ROC AUC.

    El resultado se exporta a:
    {settings.paths.images}/plot_evaluation_metrics/{plot_name}.svg

    ## Args:
        evaluation_metrics: Métricas del modelo.
        plot_name: Nombre base del archivo SVG a generar, sin extensión.
        settings: Configuración con el directorio base donde se guarda la imagen.
            Si no se indica, se obtiene la configuración global.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(evaluation_metrics.fpr, evaluation_metrics.tpr)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve - {plot_name}")

    ax.annotate(
        f"Accuracy: {evaluation_metrics.accuracy:.6f}\nROC AUC: {evaluation_metrics.roc_auc:.6f}",
        xy=(0.04, 0.93),
        xycoords="axes fraction",
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "lightgray", "lw": 1},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    _save_fig_as_svg(fig, plot_name, settings.paths.images, "plot_evaluation_metrics")


def _plot_single_deciles_table(
    ax: matplotlib.axes.Axes, deciles: pl.DataFrame, title: str
) -> None:
    """Genera una tabla de deciles con formato en un eje de Matplotlib.

    Convierte los valores numéricos a texto con dos decimales y representa el
    DataFrame como una tabla de Matplotlib. Aplica un encabezado con fondo azul
    oscuro y texto blanco en negrita, alterna el color de fondo de las filas y
    asigna el título indicado al eje.

    ## Args:
        ax: Eje de Matplotlib donde se dibuja la tabla.
        deciles: Tabla de deciles con las métricas por decil.
        title: Título a mostrar sobre la tabla.
    """
    ax.axis("tight")
    ax.axis("off")

    formatted_df = _format_floats(deciles).select(pl.all().cast(pl.String))

    columns = deciles.columns

    table = ax.table(
        cellText=formatted_df.rows(),
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    num_rows = len(formatted_df) + 1
    num_cols = len(columns)

    for col in range(num_cols):
        cell = table[0, col]
        cell.set_facecolor("#1F4E78")
        cell.set_text_props(color="white", weight="bold")

    ODD_COLOR = "#FFFFFF"
    EVEN_COLOR = "#DCE0E8"

    for row in range(1, num_rows):
        color = EVEN_COLOR if row % 2 == 0 else ODD_COLOR
        for col in range(num_cols):
            table[row, col].set_facecolor(color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")


def plot_deciles(
    evaluation_metrics: evaluation.EvaluationMetrics,
    plot_name: str,
    settings: config.Settings,
) -> None:
    """Genera un gráfico con tablas de deciles de entrenamiento y prueba.

    Crea dos tablas formateadas apiladas verticalmente, una para el conjunto de
    entrenamiento y otra para el conjunto de prueba, y guarda el resultado como
    un único archivo SVG optimizado.

    El resultado se exporta a:
    {settings.paths.images}/plot_evaluation_metrics/{plot_name}.svg

    ## Args:
        evaluation_metrics: Métricas del modelo.
        plot_name: Nombre base del archivo SVG a generar, sin extensión.
        settings: Configuración con el directorio base donde se guarda la imagen.
            Si no se indica, se obtiene la configuración global.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=300)

    _plot_single_deciles_table(
        ax1, evaluation_metrics.train_deciles_without_probs, "Train Deciles"
    )
    _plot_single_deciles_table(
        ax2, evaluation_metrics.test_deciles_without_probs, "Test Deciles"
    )

    plt.tight_layout()
    _save_fig_as_svg(fig, plot_name, settings.paths.images, "plot_evaluation_metrics")
