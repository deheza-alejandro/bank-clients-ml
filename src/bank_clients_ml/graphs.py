from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from bank_clients_ml.config import Settings, get_settings


def _generate_single_bivariate_chart(
    df: pl.DataFrame,
    variable_to_graph: str,
    output_path: Path,
    max_bins_quantity: int = 20,
    settings: Settings | None = None,
):
    """Función auxiliar (worker) que corre en un proceso independiente.

    Genera una figura con la tabla resumen y el gráfico bivariado de la variable.

    Arma una sola figura de Matplotlib con dos subplots: arriba queda la tabla
    de métricas por bin y abajo quedan las barras de clientes con la curva de
    % target en verde.

    Parámetros:
    df: Tabla de datos de los clientes con target y variable_to_graph.
    cada fila representa a un cliente,
    variable: Nombre de la variable a analizar. Valores idénticos siempre van al mismo bin.
    target: Nombre de la columna objetivo.
    max_bins_quantity: Cantidad maxima de bins en los que se puede dividir la variable.
    """
    if settings is None:
        settings = get_settings()

    target_pct_col = f"{settings.col_target}_pct"

    if df[variable_to_graph].n_unique() > max_bins_quantity:
        group_expr = pl.col(variable_to_graph).qcut(
            quantiles=max_bins_quantity, allow_duplicates=True
        )
    else:
        group_expr = pl.col(variable_to_graph)

    table = (
        df.select(settings.col_target, variable_to_graph)
        .group_by(group_expr.alias("_bin"))
        .agg(
            pl.col(variable_to_graph).min().round(2).alias("Min"),
            pl.col(variable_to_graph).max().round(2).alias("Max"),
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
    )

    columns = ["Bin", "Min", "Max", "Clients", settings.col_target, target_pct_col]

    fig, (ax_table, ax_graph) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [1, 1]}
    )

    ax_table.axis("off")
    ax_table.set_title(f"Variable analysis: {variable_to_graph}", pad=1)
    ax_table.table(
        cellText=table.select(columns).rows(),
        colLabels=columns,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 0.99],
    )

    x_indices = range(len(table))

    ax_graph.bar(x_indices, table["Clients"], width=0.35)
    ax_graph.set_ylabel("Clients")
    ax_graph.set_xticks(x_indices)
    ax_graph.set_xticklabels(table["Bin"], rotation=0, ha="center")

    ax_graph_target_pct = ax_graph.twinx()
    ax_graph_target_pct.plot(
        x_indices, table[target_pct_col], marker="o", color="green"
    )
    ax_graph_target_pct.set_ylabel(f"{settings.col_target} pct (%)")

    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


IMAGES_DIR: str = "images"


def generate_bivariate_charts(
    df: pl.DataFrame,
    columns_to_graph: list[str],
    analysis_name: str,
    max_bins_quantity: int = 20,
    images_dir: str = IMAGES_DIR,
    max_workers: int | None = None,
    settings: Settings | None = None,
):
    """Grafica las variables y guarda cada figura como SVG.

    Por cada columna del DataFrame arma el análisis
    bivariado con _generate_single_bivariate_chart
    y lo exporta a {images_dir}/{analysis_name}/{variable_to_graph}.svg.

    Parámetros:
    df: Tabla de datos de los clientes.
    columns_to_graph: columnas a graficar
    analysis_name: Nombre de la carpeta de salida dentro de {images_dir}/.
    max_bins_quantity: Cantidad maxima de bins en los que se divide cada variable.
    max_workers: maxima cantidad de CPUs a usar.
    por defecto con None se usan todos los nucleos disponibles.
    si len(columns_to_graph) < 20 automaticamente se usa 1 sola CPU.
    """
    if settings is None:
        settings = get_settings()

    output_folder = Path(images_dir) / analysis_name
    output_folder.mkdir(parents=True, exist_ok=True)

    if max_workers == 1 or len(columns_to_graph) < 20:
        for variable_to_graph in columns_to_graph:
            _generate_single_bivariate_chart(
                df.select(settings.col_target, variable_to_graph),
                variable_to_graph,
                output_folder / f"{variable_to_graph}.svg",
                max_bins_quantity,
                settings,
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _generate_single_bivariate_chart,
                    df=df.select(settings.col_target, variable_to_graph),
                    variable_to_graph=variable_to_graph,
                    output_path=output_folder / f"{variable_to_graph}.svg",
                    max_bins_quantity=max_bins_quantity,
                    settings=settings,
                )
                for variable_to_graph in columns_to_graph
            ]

            for future in as_completed(futures):
                future.result()


def _save_fig_as_svg(
    fig: Figure,
    graphic_name: str,
    images_dir: str = IMAGES_DIR,
    images_sub_dir: str = "",
) -> None:
    """Guarda una figura de Matplotlib como archivo SVG.

    Parámetros:
    -----------
    fig : Figure
        Instancia de la figura de Matplotlib.
    graphic_name : str
        Nombre del archivo sin extensión.
    images_sub_dir : str
        Subcarpeta dentro del directorio images_dir.
    """
    output_folder = Path(images_dir) / images_sub_dir
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{graphic_name}.svg"

    try:
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_top_features(
    variables_to_graph: pl.DataFrame,
    graphic_name: str,
    top_n: int = 20,
    images_dir: str = IMAGES_DIR,
    settings: Settings | None = None,
) -> None:
    """Grafica el ranking de las top_n features más importantes y lo guarda en SVG.

    Toma las top_n features con mayor importancia, arma un gráfico de barras
    horizontal y lo exporta a {images_dir}/plot_top_features/{graphic_name}.svg.

    Parámetros:
    variables_to_graph (pl.DataFrame): variables con sus importancias. ya viene ordenado
    graphic_name (str): Nombre del archivo SVG de salida (sin extensión).
    """
    if settings is None:
        settings = get_settings()

    top_n = min(top_n, variables_to_graph.height)

    top_vars = variables_to_graph.head(top_n)

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
        f"{graphic_name}: top {top_n} Features",
        fontsize=label_fontsize
    )

    _save_fig_as_svg(fig, graphic_name, images_dir, "plot_top_features")


def plot_roc_and_metrics(
    y_true: pl.Series,
    probabilities: np.ndarray,
    y_pred: np.ndarray,
    graphic_name: str,
    images_dir: str = IMAGES_DIR,
):
    """Calcula Accuracy y ROC AUC, dibuja la curva ROC y la guarda como SVG.

    Computa las métricas básicas, arma el gráfico de la curva ROC con las
    anotaciones y lo exporta a {images_dir}/plot_roc_and_metrics/{graphic_name}.svg.

    Parámetros:
    y_true: Etiquetas reales (target).
    probabilities: Probabilidades de la clase positiva.
    y_pred: Predicciones de clase (0 o 1).
    graphic_name: Nombre del archivo SVG de salida (sin extensión).
    """
    y_true_arr = y_true.to_numpy()
    y_score_arr = probabilities[:, 1]

    roc_auc = roc_auc_score(y_true_arr, y_score_arr)
    accuracy = accuracy_score(y_true_arr, y_pred)
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve - {graphic_name}")

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

    _save_fig_as_svg(fig, graphic_name, images_dir, "plot_roc_and_metrics")
