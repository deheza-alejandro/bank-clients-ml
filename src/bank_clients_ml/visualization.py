import io
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import polars.selectors as cs
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bank_clients_ml.config import Settings, get_settings

matplotlib.rcParams["svg.hashsalt"] = "fixed_seed_for_this_project"
matplotlib.rcParams["svg.fonttype"] = "path"


def _save_fig_as_svg(
    fig: Figure,
    plot_name: str,
    images_dir: Path,
    images_sub_dir: str = "",
) -> None:
    """Guarda una figura de Matplotlib como archivo SVG.

    Parámetros:
    -----------
    fig : Figure
        Instancia de la figura de Matplotlib.
    plot_name : str
        Nombre del archivo sin extensión.
    images_sub_dir : str
        Sub carpeta dentro del directorio images_dir.
    """
    output_folder = images_dir / images_sub_dir
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{plot_name}.svg"

    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="svg", bbox_inches="tight")
    finally:
        plt.close(fig)

    cmd = ["bun", "run", "svgo", "--multipass", "-i", "-", "-o", "-"]

    try:
        with Path.open(svg_path, "wb") as out_file:
            subprocess.run(  # noqa: S603
                cmd,
                input=buffer.getvalue(),
                stdout=out_file,
                check=True,
            )
    except FileNotFoundError as err:
        raise RuntimeError(
            "Bun no se encuentra en el PATH del sistema. Asegúrate de tener Bun instalado."
        ) from err


PROJECT_DIR: Path = Path(__file__).parent.parent.parent
IMAGES_DIR: Path = PROJECT_DIR / "notebooks" / "images"


def plot_top_features(
    variables_to_plot: pl.DataFrame,
    plot_name: str,
    roc_auc: float,
    top_n: int = 20,
    images_dir: Path = IMAGES_DIR,
    settings: Settings | None = None,
) -> None:
    """Graficar el ranking de las top_n features más importantes y lo guarda en SVG.

    Toma las top_n features con mayor importancia, arma un gráfico de barras
    horizontal y lo exporta a {images_dir}/plot_top_features/{plot_name}.svg.

    Parámetros:
    variables_to_plot (pl.DataFrame): variables con sus importancias. ya viene ordenado
    plot_name (str): Nombre del archivo SVG de salida (sin extensión).
    """
    if settings is None:
        settings = get_settings()

    top_n = min(top_n, variables_to_plot.height)

    top_vars = variables_to_plot.head(top_n)

    label_fontsize = min(35, top_n * 8)
    tick_fontsize = 30
    value_fontsize = 20

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.7)))

    bars = ax.barh(
        top_vars[settings.col_feature],
        top_vars[settings.col_importance],
        color="#4a90e2",
        edgecolor="black",
    )

    ax.invert_yaxis()
    ax.set_xlabel(settings.col_importance, fontsize=label_fontsize)
    ax.set_ylabel(settings.col_feature, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.bar_label(bars, fmt="%.0f", padding=5, fontsize=value_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    ax.set_title(
        f"{plot_name}: top {top_n} Features\nROC AUC: {roc_auc:.6f}",
        fontsize=label_fontsize,
    )

    _save_fig_as_svg(fig, plot_name, images_dir, "plot_top_features")


def _plot_single_bivariate_chart(
    table: pl.DataFrame,
    variable_to_plot: str,
    images_dir: Path,
    analysis_name: str,
    settings: Settings | None = None,
) -> None:
    """Función auxiliar (worker) que corre en un proceso independiente.

    Genera una figura con la tabla resumen y el gráfico bivariado de la variable.

    Arma una sola figura de Matplotlib con dos subplots: arriba queda la tabla
    de métricas por bin y abajo quedan las barras de clientes con la curva de
    % target en verde.

    Parámetros:
    table: Tabla de datos de los clientes con target y variable_to_plot.
    variable_to_plot: Nombre de la variable a analizar. Valores idénticos siempre van al mismo bin.
    target: Nombre de la columna objetivo.
    """
    if settings is None:
        settings = get_settings()

    fig, (ax_table, ax_graph) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1]}
    )

    ax_table.axis("off")
    ax_table.set_title(f"Variable analysis: {variable_to_plot}", pad=1)
    ax_table.table(
        cellText=table.rows(),
        colLabels=table.columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.99],
    )

    x_indices = range(len(table))

    ax_graph.bar(x_indices, table.to_series(3), width=0.35)
    ax_graph.set_ylabel("Clients")
    ax_graph.set_xticks(x_indices)
    ax_graph.set_xticklabels(table.to_series(0), rotation=0, ha="center")

    ax_graph_target_pct = ax_graph.twinx()
    ax_graph_target_pct.plot(x_indices, table.to_series(-1), marker="o", color="green")
    ax_graph_target_pct.set_ylabel(f"{settings.col_target} pct (%)")

    fig.tight_layout()
    _save_fig_as_svg(
        fig, variable_to_plot, images_dir, f"bivariate_analysis/{analysis_name}"
    )


def plot_bivariate_charts(
    tables: dict[str, pl.DataFrame],
    analysis_name: str,
    images_dir: Path = IMAGES_DIR,
    max_workers: int | None = None,
    settings: Settings | None = None,
) -> None:
    """Graficar las variables y guarda cada figura como SVG.

    Por cada columna del DataFrame arma el análisis
    bivariado con _plot_single_bivariate_chart
    y lo exporta a {images_dir}/{analysis_name}/{variable_to_plot}.svg.

    Parámetros:
    tables: todas las tablas de cada columna a graficar
    analysis_name: Nombre de la carpeta de salida dentro de {images_dir}/.
    max_workers: maxima cantidad de CPUs a usar.
    por defecto con None se usan todos los núcleos disponibles.
    si len(tables) < 20 automáticamente se usa 1 sola CPU.
    """
    if settings is None:
        settings = get_settings()

    if max_workers == 1 or len(tables) < 20:
        for variable_to_plot, table in tables.items():
            _plot_single_bivariate_chart(
                table,
                variable_to_plot,
                images_dir,
                analysis_name,
                settings,
            )
    else:
        with ProcessPoolExecutor(max_workers) as executor:
            futures = [
                executor.submit(
                    _plot_single_bivariate_chart,
                    table,
                    variable_to_plot,
                    images_dir,
                    analysis_name,
                    settings,
                )
                for variable_to_plot, table in tables.items()
            ]

            for future in as_completed(futures):
                future.result()


def plot_evaluation_metrics(
    roc_auc: float,
    accuracy: float,
    fpr: np.ndarray,
    tpr: np.ndarray,
    plot_name: str,
    images_dir: Path = IMAGES_DIR,
) -> None:
    """Dibuja la curva ROC y la guarda como SVG.

    Arma el gráfico de la curva ROC con las
    anotaciones y lo exporta a {images_dir}/plot_evaluation_metrics/{plot_name}.svg.

    Parámetros:
    plot_name: Nombre del archivo SVG de salida (sin extensión).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve - {plot_name}")

    ax.annotate(
        f"Accuracy: {accuracy:.6f}\nROC AUC:  {roc_auc:.6f}",
        xy=(0.04, 0.93),
        xycoords="axes fraction",
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "lightgray", "lw": 1},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    _save_fig_as_svg(fig, plot_name, images_dir, "plot_evaluation_metrics")


def _plot_single_deciles_table(ax, deciles: pl.DataFrame, title: str) -> None:
    ax.axis("tight")
    ax.axis("off")

    formatted_df = deciles.with_columns(cs.float().round(2).cast(pl.String)).select(
        pl.all().cast(pl.String)
    )

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

    for row in range(1, num_rows):
        for col in range(num_cols):
            table[row, col].set_facecolor("#DCE0E8" if row % 2 == 0 else "#FFFFFF")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")


def plot_deciles(
    train_deciles: pl.DataFrame,
    test_deciles: pl.DataFrame,
    plot_name: str,
    title_train_deciles: str = "Train Deciles",
    title_test_deciles: str = "Test Deciles",
    images_dir: Path = IMAGES_DIR,
) -> None:
    """
    Recibe dos DataFrames de Polars y genera un svg con ambas
    tablas organizadas verticalmente.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=300)

    _plot_single_deciles_table(ax1, train_deciles, title_train_deciles)
    _plot_single_deciles_table(ax2, test_deciles, title_test_deciles)

    plt.tight_layout()
    _save_fig_as_svg(fig, plot_name, images_dir, "plot_evaluation_metrics")
