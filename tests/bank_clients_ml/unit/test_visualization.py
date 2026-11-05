"""Tests unitarios para la generación y guardado de gráficos."""

import pathlib
from collections.abc import Generator
from typing import cast

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import polars as pl
import pytest
import pytest_mock
from matplotlib.figure import Figure

from bank_clients_ml import bivariate_table, config, evaluation, visualization


@pytest.fixture(autouse=True)
def close_figures_after_each_test() -> Generator[None]:
    """Cierra todas las figuras abiertas al finalizar cada test."""
    yield
    plt.close("all")


@pytest.fixture
def plot_settings(tmp_path: pathlib.Path) -> config.Settings:
    """Devuelve una configuración con las imágenes en un directorio temporal.

    ## Args:
        tmp_path: Directorio temporal proporcionado por pytest.

    ## Returns:
        Configuración aislada para las pruebas de escritura de gráficos.
    """
    return config.Settings(paths=config.DirectoryPaths(project=tmp_path))


@pytest.fixture
def capture_saver() -> tuple[
    list[tuple[Figure, pathlib.Path]], visualization.FigureSaver
]:
    """Devuelve una estrategia de guardado que captura figuras y rutas.

    ## Returns:
        Lista de capturas y la estrategia que las acumula.
    """
    saved: list[tuple[Figure, pathlib.Path]] = []

    def saver(fig: Figure, path: pathlib.Path) -> None:
        saved.append((fig, path))

    return saved, saver


@pytest.fixture
def top_features_table() -> pl.DataFrame:
    """Devuelve una tabla de importancias ordenada de mayor a menor.

    ## Returns:
        Tabla con cinco variables y sus importancias.
    """
    return pl.DataFrame(
        {
            "Feature": ["Age", "Income", "Balance", "Tenure", "Score"],
            "Importance": [50.0, 30.0, 15.0, 4.0, 1.0],
        }
    )


@pytest.fixture
def bivariate_age() -> bivariate_table.BivariateTable:
    """Devuelve una tabla bivariada de ejemplo para la edad.

    ## Returns:
        Tabla con tres bines y valores numéricos para formatear.
    """
    table = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": [18.456, 35.5, 60.0],
            "Max": [34.0, 59.5, 80.0],
            "Clients": pl.Series([10, 20, 30], dtype=pl.UInt32),
            "Target": pl.Series([1, 4, 9], dtype=pl.Int64),
            "Target_pct": [10.4, 19.6, 30.49],
        }
    )
    return bivariate_table.BivariateTable("age", "Age", table)


@pytest.fixture
def evaluation_metrics() -> evaluation.EvaluationMetrics:
    """Devuelve métricas de evaluación con deciles de entrenamiento y prueba.

    ## Returns:
        Métricas con curva ROC y tablas de deciles de ejemplo.
    """
    train_deciles = pl.DataFrame(
        {
            "decile": ["1", "2"],
            "count": [10, 20],
            "target_1_count": [3, 7],
            "min_prob": [10.0, 30.0],
            "max_prob": [20.0, 40.0],
            "target_1_rate": [30.0, 35.0],
            "cum_gain": [30.0, 100.0],
            "lift": [1.234, 1.456],
            "ks": [10.0, 20.0],
        }
    )
    test_deciles = pl.DataFrame(
        {
            "decile": ["1", "2"],
            "count": [12, 18],
            "target_1_count": [2, 8],
            "min_prob": [15.0, 35.0],
            "max_prob": [25.0, 45.0],
            "target_1_rate": [16.666, 44.444],
            "cum_gain": [20.0, 100.0],
            "lift": [0.8, 1.2],
            "ks": [5.0, 25.0],
        }
    )
    return evaluation.EvaluationMetrics(
        roc_auc=0.876543,
        accuracy=0.765432,
        fpr=np.array([0.0, 0.2, 1.0]),
        tpr=np.array([0.0, 0.8, 1.0]),
        train_deciles=train_deciles,
        test_deciles=test_deciles,
    )


def test_should_close_figure_after_plotting_top_features(
    mocker: pytest_mock.MockerFixture,
    top_features_table: pl.DataFrame,
    plot_settings: config.Settings,
) -> None:
    """Verifica que la API pública cierre la figura después de guardarla.

    ## Args:
        mocker: Utilidad para evitar la ejecución de SVGO.
        top_features_table: Tabla de importancias ordenada.
        plot_settings: Configuración con un directorio de imágenes temporal.
    """
    mocker.patch.object(visualization.subprocess, "run")

    visualization.plot_top_features(
        top_features_table,
        "chart",
        roc_auc=0.9,
        settings=plot_settings,
    )

    assert plt.get_fignums() == []


def test_should_raise_runtime_error_and_close_figure_when_bun_is_missing(
    mocker: pytest_mock.MockerFixture,
    top_features_table: pl.DataFrame,
    plot_settings: config.Settings,
) -> None:
    """Verifica el error informativo cuando Bun no está disponible.

    ## Args:
        mocker: Utilidad para simular la ausencia del ejecutable.
        top_features_table: Tabla de importancias ordenada.
        plot_settings: Configuración con un directorio de imágenes temporal.
    """
    mocker.patch.object(
        visualization.subprocess,
        "run",
        side_effect=FileNotFoundError,
    )

    with pytest.raises(RuntimeError, match="Bun executable not found"):
        visualization.plot_top_features(
            top_features_table,
            "chart",
            roc_auc=0.9,
            settings=plot_settings,
        )

    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    ("top_n", "expected_features"),
    [
        pytest.param(2, ["Age", "Income"], id="requested-count"),
        pytest.param(
            20,
            ["Age", "Income", "Balance", "Tenure", "Score"],
            id="clamped-to-available-rows",
        ),
    ],
)
def test_should_build_top_features_figure_with_requested_limit_and_expected_title(
    top_features_table: pl.DataFrame,
    plot_settings: config.Settings,
    top_n: int,
    expected_features: list[str],
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica el límite de variables y el contenido del gráfico de importancia.

    ## Args:
        top_features_table: Tabla de importancias ordenada.
        plot_settings: Configuración con nombres de columnas y directorio temporal.
        top_n: Cantidad solicitada de variables.
        expected_features: Variables que deben aparecer en el gráfico.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    saved, saver = capture_saver

    visualization.plot_top_features(
        top_features_table,
        "model_a",
        roc_auc=0.987654321,
        top_n=top_n,
        settings=plot_settings,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        plot_settings.paths.images / "plot_top_features" / "model_a.svg"
    )
    assert plt.get_fignums() == []

    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == expected_features
    widths = [
        cast(matplotlib.patches.Rectangle, patch).get_width() for patch in ax.patches
    ]
    assert widths == pytest.approx(
        top_features_table.get_column("Importance")
        .head(len(expected_features))
        .to_list()
    )
    assert ax.get_xlabel() == plot_settings.analysis.importance
    assert ax.get_ylabel() == plot_settings.analysis.feature
    assert ax.get_title() == (
        f"model_a: top {len(expected_features)} Features\nROC AUC: 0.987654"
    )


def test_should_use_configured_column_names_in_top_features_figure(
    tmp_path: pathlib.Path,
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica que los ejes usen los nombres de columna configurados.

    ## Args:
        tmp_path: Directorio temporal proporcionado por pytest.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    custom_settings = config.Settings(
        paths=config.DirectoryPaths(project=tmp_path),
        analysis=config.AnalysisColumns(feature="Variable", importance="Weight"),
    )
    variables = pl.DataFrame({"Variable": ["A", "B"], "Weight": [7.0, 3.0]})
    saved, saver = capture_saver

    visualization.plot_top_features(
        variables,
        "custom",
        roc_auc=0.5,
        settings=custom_settings,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        custom_settings.paths.images / "plot_top_features" / "custom.svg"
    )

    ax = fig.axes[0]
    assert ax.get_ylabel() == "Variable"
    assert ax.get_xlabel() == "Weight"
    assert [label.get_text() for label in ax.get_yticklabels()] == ["A", "B"]


def test_should_build_bivariate_figure_with_table_bars_and_target_percentage_line(
    bivariate_age: bivariate_table.BivariateTable,
    plot_settings: config.Settings,
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica el contenido de la tabla, las barras y la línea del target.

    ## Args:
        bivariate_age: Tabla bivariada de ejemplo.
        plot_settings: Configuración con nombres de columnas y directorio temporal.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    saved, saver = capture_saver

    visualization.plot_bivariate_charts(
        [bivariate_age],
        "test_analysis",
        plot_settings,
        max_workers=1,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        plot_settings.paths.images / "bivariate_analysis" / "test_analysis" / "age.svg"
    )

    ax_table, ax_graph, ax_target_pct = fig.axes
    table = ax_table.tables[0]
    headers = [table[0, column].get_text().get_text() for column in range(6)]
    assert ax_table.get_title() == "Variable analysis: Age"
    assert headers == ["Bin", "Min", "Max", "Clients", "Target", "Target_pct"]
    assert table[1, 1].get_text().get_text() == "18.46"
    assert table[1, 5].get_text().get_text() == "10"
    heights = [
        cast(matplotlib.patches.Rectangle, patch).get_height()
        for patch in ax_graph.patches
    ]
    assert heights == [10.0, 20.0, 30.0]
    assert [label.get_text() for label in ax_graph.get_xticklabels()] == [
        "1",
        "2",
        "3",
    ]
    assert ax_graph.get_ylabel() == plot_settings.analysis.clients
    target_pct_values = cast(
        numpy.typing.NDArray[np.float64],
        ax_target_pct.lines[0].get_ydata(),
    )
    np.testing.assert_allclose(
        target_pct_values,
        [10.0, 20.0, 30.0],
    )
    assert ax_target_pct.get_ylabel() == f"{plot_settings.dataset.target} pct (%)"


def test_should_use_configured_columns_in_bivariate_figure(
    tmp_path: pathlib.Path,
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica que la tabla y los ejes respeten la configuración personalizada.

    ## Args:
        tmp_path: Directorio temporal proporcionado por pytest.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    custom_settings = config.Settings(
        paths=config.DirectoryPaths(project=tmp_path),
        dataset=config.DatasetColumns(target="label"),
        analysis=config.AnalysisColumns(
            bin="bucket",
            clients="customers",
            target_pct="pct",
        ),
    )
    table = pl.DataFrame(
        {
            "bucket": [1, 2],
            "Min": [0.0, 1.0],
            "Max": [0.5, 2.0],
            "customers": [4, 6],
            "label": [1, 2],
            "pct": [25.0, 33.333],
        }
    )
    bivariate = bivariate_table.BivariateTable("income", "Income", table)
    saved, saver = capture_saver

    visualization.plot_bivariate_charts(
        [bivariate],
        "custom_analysis",
        custom_settings,
        max_workers=1,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        custom_settings.paths.images
        / "bivariate_analysis"
        / "custom_analysis"
        / "income.svg"
    )

    ax_table, ax_graph, ax_target_pct = fig.axes
    table_artist = ax_table.tables[0]
    headers = [table_artist[0, column].get_text().get_text() for column in range(6)]
    assert headers == ["bucket", "Min", "Max", "customers", "label", "pct"]
    assert table_artist[1, 5].get_text().get_text() == "25"
    assert table_artist[2, 5].get_text().get_text() == "33"
    assert ax_graph.get_ylabel() == "customers"
    assert ax_target_pct.get_ylabel() == "label pct (%)"


def _line_data(
    line,
) -> tuple[numpy.typing.NDArray[np.float64], numpy.typing.NDArray[np.float64]]:
    """Devuelve los datos de una línea con el tipo esperado por NumPy."""
    return (
        cast(numpy.typing.NDArray[np.float64], line.get_xdata()),
        cast(numpy.typing.NDArray[np.float64], line.get_ydata()),
    )


def test_should_build_roc_curve_figure_with_metrics_annotation(
    evaluation_metrics: evaluation.EvaluationMetrics,
    plot_settings: config.Settings,
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica la curva ROC, la diagonal y la anotación de métricas.

    ## Args:
        evaluation_metrics: Métricas de evaluación de ejemplo.
        plot_settings: Configuración con un directorio de imágenes temporal.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    saved, saver = capture_saver

    visualization.plot_roc_curve(
        evaluation_metrics,
        "test_model",
        plot_settings,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        plot_settings.paths.images / "plot_evaluation_metrics" / "test_model.svg"
    )

    ax = fig.axes[0]
    xdata_0, ydata_0 = _line_data(ax.lines[0])
    xdata_1, ydata_1 = _line_data(ax.lines[1])
    np.testing.assert_allclose(xdata_0, evaluation_metrics.fpr)
    np.testing.assert_allclose(ydata_0, evaluation_metrics.tpr)
    np.testing.assert_allclose(xdata_1, [0.0, 1.0])
    np.testing.assert_allclose(ydata_1, [0.0, 1.0])
    assert ax.get_xlabel() == "False Positive Rate (FPR)"
    assert ax.get_ylabel() == "True Positive Rate (TPR)"
    assert ax.get_title() == "ROC Curve - test_model"
    assert ax.texts[0].get_text() == "Accuracy: 0.765432\nROC AUC: 0.876543"
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)


def test_should_build_train_and_test_deciles_in_single_figure(
    evaluation_metrics: evaluation.EvaluationMetrics,
    plot_settings: config.Settings,
    capture_saver: tuple[list[tuple[Figure, pathlib.Path]], visualization.FigureSaver],
) -> None:
    """Verifica títulos, ejes, cabeceras y redondeo de las tablas de deciles.

    ## Args:
        evaluation_metrics: Métricas con las tablas de deciles.
        plot_settings: Configuración con un directorio de imágenes temporal.
        capture_saver: Captura de figuras y rutas guardadas.
    """
    saved, saver = capture_saver

    visualization.plot_deciles(
        evaluation_metrics,
        "deciles",
        plot_settings,
        figure_saver=saver,
    )

    assert len(saved) == 1
    fig, svg_path = saved[0]
    assert svg_path == (
        plot_settings.paths.images / "plot_evaluation_metrics" / "deciles.svg"
    )

    assert len(fig.axes) == 2

    train_ax, test_ax = fig.axes
    assert train_ax.get_title(loc="left") == "Train Deciles"
    assert test_ax.get_title(loc="left") == "Test Deciles"
    assert not train_ax.axison
    assert not test_ax.axison

    train_table = train_ax.tables[0]
    test_table = test_ax.tables[0]
    train_columns = evaluation_metrics.train_deciles_without_probs.columns
    test_columns = evaluation_metrics.test_deciles_without_probs.columns
    train_headers = [
        train_table[0, column].get_text().get_text()
        for column in range(len(train_columns))
    ]
    test_headers = [
        test_table[0, column].get_text().get_text()
        for column in range(len(test_columns))
    ]
    assert train_headers == train_columns
    assert test_headers == test_columns
    assert "min_prob" not in train_headers
    assert "max_prob" not in test_headers
    assert train_table[1, 0].get_text().get_text() == "1"
    assert test_table[1, 0].get_text().get_text() == "1"
    assert train_table[1, train_columns.index("lift")].get_text().get_text() == "1.23"
    assert train_table[2, train_columns.index("lift")].get_text().get_text() == "1.46"
    assert (
        test_table[1, test_columns.index("target_1_rate")].get_text().get_text()
        == "16.67"
    )
    assert (
        test_table[2, test_columns.index("target_1_rate")].get_text().get_text()
        == "44.44"
    )
