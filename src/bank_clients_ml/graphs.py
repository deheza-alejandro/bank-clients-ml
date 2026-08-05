import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from io import BytesIO
import traceback
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from IPython.display import display


# ------------------------ GRAFICO ------------------------
def Graficar_vs_TGT(df, campo, id, tgt, cant_bines):
    df = df.copy()
    df["rank"] = (df[campo].rank(pct=True) * cant_bines).round()

    g = df.groupby("rank")
    c = pd.DataFrame(
        {
            "rank": g.size().index,
            "min": g[campo].min().round(1),
            "max": g[campo].max().round(1),
            "Clientes": g[id].nunique().astype(int),
            tgt: g[tgt].sum().astype(int),
        }
    )

    c["TGT_p"] = ((c[tgt] / c["Clientes"]) * 100).round().astype(int)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Barras de clientes
    ax1.bar(range(len(c)), c["Clientes"], width=0.35)
    ax1.set_ylabel("Clientes")
    ax1.set_xticks(range(len(c)))
    ax1.set_xticklabels(c["rank"], rotation=45, ha="right")

    # Curva TGT en VERDE
    ax2 = ax1.twinx()
    ax2.plot(range(len(c)), c["TGT_p"], marker="o", color="green")
    ax2.set_ylabel("% TGT")

    columnas = ["rank", "min", "max", "Clientes", tgt, "TGT_p"]

    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    print(c[columnas].to_string(index=False))  # Mostrar tabla en el notebook
    plt.show()  # Mostrar grafico en el notebook
    plt.close(fig)
    return c, img_buffer, columnas


# ------------------------ ENCABEZADO + GRAFICO + TABLA ------------------------
def agregar_a_pdf(
    pdf, titulo, df, columnas, img_buffer, x_inicial=10, renglon_inicial=45
):
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 12, titulo, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf.set_xy(x_inicial, renglon_inicial)
    pdf.set_font("helvetica", "B", 10)

    # Encabezados
    for col in columnas:
        pdf.cell(30, 6, col, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")
    pdf.ln()

    pdf.set_font("helvetica", "", 10)
    renglon = renglon_inicial + 6

    for _, row in df[columnas].iterrows():
        if renglon > 250:
            pdf.add_page()
            renglon = 20

        pdf.set_xy(x_inicial, renglon)
        for col in columnas:
            pdf.cell(
                30,
                6,
                str(row[col]),
                border=1,
                new_x=XPos.RIGHT,
                new_y=YPos.TOP,
                align="C",
            )
        pdf.ln()
        renglon += 6

    pdf.ln(5)
    pdf.image(img_buffer, x=10, w=180)  # Insertar imagen desde memoria


# ------------------------ FUNCION PRINCIPAL ------------------------
def Graficar_Variables(pDataFrame, id, tgt, cant_bines, destino, nombre_pdf):
    pdf = FPDF()

    for campo in pDataFrame.columns:
        if campo in [id, "cuit", "Unnamed: 0"]:
            continue

        try:
            print("\n---------------------", campo, "---------------------")
            df, img_buffer, columnas = Graficar_vs_TGT(
                pDataFrame, campo, id, tgt, cant_bines
            )

            agregar_a_pdf(
                pdf, f"Análisis de la variable: {campo}", df, columnas, img_buffer
            )

        except Exception as e:
            print(f"Error en {campo}: {e}")
            traceback.print_exc()

    # ---- GUARDAR SOLO UN PDF ----

    # Obtener la ruta de la carpeta donde está este archivo:
    # para archivo .ipybn
    RAIZ_PROYECTO = Path.cwd()
    # para archivo .py
    # RAIZ_PROYECTO = Path(__file__).resolve().parent

    # Construir la ruta completa combinando la raíz, destino y nombre_pdf
    ruta_final = RAIZ_PROYECTO / destino / f"{nombre_pdf}.pdf"

    # Asegura que la carpeta exista (si no existe, la crea)
    ruta_final.parent.mkdir(parents=True, exist_ok=True)

    # Guardar el PDF (fpdf2 acepta el objeto Path directamente)
    pdf.output(ruta_final)


def graficar_top_20(variables_a_graficar):
    # Seleccionar las top 20
    top_vars = variables_a_graficar.nlargest(20)

    # Parámetros personalizables
    label_fontsize = 20  # tamaño de etiquetas de ejes
    tick_fontsize = 30  # tamaño de los ticks (nombres de variables)
    value_fontsize = 20  # tamaño de los valores sobre las barras

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(top_vars.index, top_vars.values, color="#4a90e2", edgecolor="black")

    # Invertir el eje Y para que la más importante quede arriba
    ax.invert_yaxis()

    # Etiquetas
    ax.set_xlabel("Importancia", fontsize=label_fontsize)
    ax.set_ylabel("Variable", fontsize=label_fontsize)

    # Ajustar tamaño de los ticks
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Mostrar los valores al final de cada barra
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01 * top_vars.max(),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",  # 🔹 sin decimales
            va="center",
            fontsize=value_fontsize,
        )

    # Estética del gráfico
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_roc_and_metrics(y_true, y_score, y_pred):
    """Calcula las métricas de Accuracy y ROC AUC, y dibuja la curva ROC.

    Parámetros:
    y_true (array-like): Etiquetas reales (Target).
    y_score (array-like): Probabilidades de la clase positiva (Prob1).
    y_pred (array-like): Predicciones de clases (0 o 1).
    """
    # Calcular métricas básicas
    roc_auc = roc_auc_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Configurar y dibujar el gráfico de la curva ROC
    plt.plot(fpr, tpr)
    plt.plot([0, 1])  # Línea diagonal de referencia (azar)
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    # Añadir las anotaciones con los resultados en el gráfico
    plt.annotate("Accuracy : {}".format(accuracy), (0.01, 0.96))
    plt.annotate("ROC : {}".format(roc_auc), (0.01, 0.91))

    # Definir límites de los ejes y mostrar
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    # Llamada al display original del entorno (ej. Jupyter)
    display()
