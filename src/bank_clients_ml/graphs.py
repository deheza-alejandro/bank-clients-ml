from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def graph_bivariate(df, variable, id, target, bins_quantity):
    """Genera una figura con la tabla resumen y el gráfico bivariado de la variable.

    Arma una sola figura de Matplotlib con dos subplots: arriba queda la tabla
    de métricas por bin y abajo quedan las barras de clientes con la curva de
    % target en verde.

    Parámetros:
    df (DataFrame): Tabla de datos de los clientes.
    variable (str): Nombre de la variable a analizar.
    id (str): Nombre de la columna que identifica a cada cliente.
    target (str): Nombre de la columna objetivo.
    bins_quantity (int): Cantidad de bins en los que se divide la variable.

    Retorna:
    tuple: (fig, c) donde fig es la figura de Matplotlib y c el DataFrame resumen.
    """
    df = df.copy()
    df["rank"] = (df[variable].rank(pct=True) * bins_quantity).round()

    g = df.groupby("rank")
    c = pd.DataFrame(
        {
            "rank": g.size().index,
            "min": g[variable].min().round(1),
            "max": g[variable].max().round(1),
            "Clients": g[id].nunique().astype(int),
            target: g[target].sum().astype(int),
        }
    )

    c["target_p"] = ((c[target] / c["Clients"]) * 100).round().astype(int)
    columnas = ["rank", "min", "max", "Clients", target, "target_p"]

    fig, (ax_table, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [1, 2]}
    )

    ax_table.axis("off")
    ax_table.set_title(f"Variable analysis: {variable}")
    ax_table.table(
        cellText=c[columnas].values,
        colLabels=columnas,
        loc="center",
        cellLoc="center",
    )

    ax1.bar(range(len(c)), c["Clients"], width=0.35)
    ax1.set_ylabel("Clients")
    ax1.set_xticks(range(len(c)))
    ax1.set_xticklabels(c["rank"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(range(len(c)), c["target_p"], marker="o", color="green")
    ax2.set_ylabel("% Target")

    fig.tight_layout()
    return fig, c


def graph_variables(pDataFrame, id, target, bins_quantity, analysis_name):
    """Grafica todas las variables válidas y guarda cada figura como SVG.

    Por cada columna del DataFrame (salvo identificadores) arma el análisis
    bivariado con graph_bivariate y lo exporta a images/{analysis_name}/{variable}.svg.

    Parámetros:
    pDataFrame (DataFrame): Tabla de datos de los clientes.
    id (str): Nombre de la columna que identifica a cada cliente.
    target (str): Nombre de la columna objetivo (target).
    bins_quantity (int): Cantidad de bins en los que se divide cada variable.
    analysis_name (str): Nombre de la carpeta de salida dentro de images/.
    """
    output_folder = Path("images") / analysis_name
    output_folder.mkdir(parents=True, exist_ok=True)

    for variable in pDataFrame.columns:
        if variable in [id, "cuit", "Unnamed: 0"]:
            continue

        fig, _ = graph_bivariate(pDataFrame, variable, id, target, bins_quantity)

        svg_path = output_folder / f"{variable}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)


def graph_top_20(variables_to_graph, graphic_name: str):
    """Grafica el ranking de las 20 variables más importantes y lo guarda en SVG.

    Toma las 20 variables con mayor importancia, arma un gráfico de barras
    horizontal y lo exporta a images/graph_top_20/{graphic_name}.svg.

    Parámetros:
    variables_to_graph (Series): Importancia de cada variable.
    graphic_name (str): Nombre del archivo SVG de salida (sin extensión).
    """
    top_vars = variables_to_graph.nlargest(20)
    label_fontsize = 20
    tick_fontsize = 30
    value_fontsize = 20

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(top_vars.index, top_vars.values, color="#4a90e2", edgecolor="black")

    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=label_fontsize)
    ax.set_ylabel("Variable", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01 * top_vars.max(),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            va="center",
            fontsize=value_fontsize,
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    output_folder = Path("images/graph_top_20")
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{graphic_name}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_roc_and_metrics(y_true, y_score, y_pred, graphic_name: str):
    """Calcula Accuracy y ROC AUC, dibuja la curva ROC y la guarda como SVG.

    Computa las métricas básicas, arma el gráfico de la curva ROC con las
    anotaciones y lo exporta a images/plot_roc_and_metrics/{graphic_name}.svg.

    Parámetros:
    y_true (array-like): Etiquetas reales (target).
    y_score (array-like): Probabilidades de la clase positiva (Prob1).
    y_pred (array-like): Predicciones de clase (0 o 1).
    graphic_name (str): Nombre del archivo SVG de salida (sin extensión).
    """
    roc_auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.annotate(f"Accuracy : {accuracy}", (0.01, 0.96))
    ax.annotate(f"ROC : {roc_auc}", (0.01, 0.91))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output_folder = Path("images/plot_roc_and_metrics")
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{graphic_name}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_axes_as_svg(axes, graphic_name: str, path: str=""):
    fig = axes.get_figure()
    output_folder = Path(f"images/{path}")
    output_folder.mkdir(parents=True, exist_ok=True)
    svg_path = output_folder / f"{graphic_name}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
