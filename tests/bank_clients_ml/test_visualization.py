"""Tests unitarios para la generación y guardado de gráficos."""

import concurrent.futures
import pathlib
from collections.abc import Generator
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from bank_clients_ml import bivariate_table, config, evaluation, visualization


@pytest.fixture(autouse=True)
def close_figures_after_each_test() -> Generator[None]:
    """Cierra todas las figuras abiertas al finalizar cada test."""
    yield
    plt.close("all")


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


def test_should_close_figure_when_context_exits() -> None:
    """Verifica que la figura se cierre al salir correctamente del contexto."""
    fig, _ = plt.subplots()
    fig_number = fig.number

    with visualization._closing_figure(fig):
        assert plt.fignum_exists(fig_number)

    assert not plt.fignum_exists(fig_number)


def test_should_close_figure_when_context_body_raises() -> None:
    """Verifica que la figura se cierre aunque ocurra un error en el contexto."""
    fig, _ = plt.subplots()
    fig_number = fig.number

    with pytest.raises(RuntimeError, match="boom"), visualization._closing_figure(fig):
        msg = "boom"
        raise RuntimeError(msg)

    assert not plt.fignum_exists(fig_number)


def test_should_round_float_columns_to_two_decimals_without_modifying_input() -> None:
    """Verifica el redondeo de columnas float y la inmutabilidad de los datos."""
    df = pl.DataFrame(
        {
            "decimal": [1.234, 5.678],
            "integer": [1, 2],
            "text": ["a", "b"],
        }
    )

    result = visualization._format_floats(df)

    expected = pl.DataFrame(
        {
            "decimal": [1.23, 5.68],
            "integer": [1, 2],
            "text": ["a", "b"],
        }
    )
    polars.testing.assert_frame_equal(result, expected)
    assert df.get_column("decimal").to_list() == [1.234, 5.678]


def test_should_write_optimized_svg_to_requested_subdirectory(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    """Verifica que el SVG optimizado se escriba en la ruta configurada.

    ## Args:
        mocker: Utilidad para sustituir la ejecución de Bun.
        tmp_path: Directorio temporal para aislar la escritura en disco.
    """
    fig, _ = plt.subplots()
    fig_number = fig.number

    def fake_run(command, **kwargs):
        """Simula la salida de SVGO escribiendo un SVG optimizado."""
        kwargs["stdout"].write(b"<svg>optimized</svg>")

    run_mock = mocker.patch.object(
        visualization.subprocess,
        "run",
        side_effect=fake_run,
    )

    visualization._save_fig_as_svg(fig, "chart", tmp_path, "nested")

    svg_path = tmp_path / "nested" / "chart.svg"
    assert svg_path.read_bytes() == b"<svg>optimized</svg>"
    call = run_mock.call_args
    assert call.args[0] == [
        "bun",
        "run",
        "svgo",
        "--multipass",
        "-i",
        "-",
        "-o",
        "-",
    ]
    assert b"<svg" in call.kwargs["input"]
    assert call.kwargs["check"] is True
    assert not plt.fignum_exists(fig_number)


def test_should_raise_runtime_error_when_bun_executable_is_missing(
    mocker: pytest_mock.MockerFixture,
    tmp_path: pathlib.Path,
) -> None:
    """Verifica el error informativo cuando Bun no está disponible.

    ## Args:
        mocker: Utilidad para simular la ausencia del ejecutable.
        tmp_path: Directorio temporal para aislar la escritura en disco.
    """
    fig, _ = plt.subplots()
    fig_number = fig.number
    mocker.patch.object(
        visualization.subprocess,
        "run",
        side_effect=FileNotFoundError,
    )

    with pytest.raises(RuntimeError, match="Bun executable not found"):
        visualization._save_fig_as_svg(fig, "chart", tmp_path)

    assert not plt.fignum_exists(fig_number)


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
    settings: config.Settings,
    top_n: int,
    expected_features: list[str],
) -> None:
    """Verifica el límite de variables y el contenido del gráfico de importancia.

    ## Args:
        top_features_table: Tabla de importancias ordenada.
        settings: Configuración con los nombres de columnas.
        top_n: Cantidad solicitada de variables.
        expected_features: Variables que deben aparecer en el gráfico.
    """
    fig = visualization._build_top_features_figure(
        top_features_table,
        "model_a",
        roc_auc=0.987654321,
        top_n=top_n,
        settings=settings,
    )

    ax = fig.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == expected_features
    widths = [cast(Rectangle, patch).get_width() for patch in ax.patches]
    assert widths == pytest.approx(
        top_features_table.get_column("Importance")
        .head(len(expected_features))
        .to_list()
    )
    assert ax.get_xlabel() == settings.analysis.importance
    assert ax.get_ylabel() == settings.analysis.feature
    assert ax.get_title() == (
        f"model_a: top {len(expected_features)} Features\nROC AUC: 0.987654"
    )


def test_should_use_configured_column_names_in_top_features_figure() -> None:
    """Verifica que los ejes usen los nombres de columna configurados."""
    custom_settings = config.Settings(
        analysis=config.AnalysisColumns(feature="Variable", importance="Weight")
    )
    variables = pl.DataFrame({"Variable": ["A", "B"], "Weight": [7.0, 3.0]})

    fig = visualization._build_top_features_figure(
        variables,
        "custom",
        roc_auc=0.5,
        settings=custom_settings,
    )

    ax = fig.axes[0]
    assert ax.get_ylabel() == "Variable"
    assert ax.get_xlabel() == "Weight"
    assert [label.get_text() for label in ax.get_yticklabels()] == ["A", "B"]


def test_should_build_bivariate_figure_with_table_bars_and_target_percentage_line(
    bivariate_age: bivariate_table.BivariateTable,
    settings: config.Settings,
) -> None:
    """Verifica el contenido de la tabla, las barras y la línea del target.

    ## Args:
        bivariate_age: Tabla bivariada de ejemplo.
        settings: Configuración con los nombres de columnas.
    """
    fig = visualization._build_bivariate_chart_figure(bivariate_age, settings)

    ax_table, ax_graph, ax_target_pct = fig.axes
    table = ax_table.tables[0]
    headers = [table[0, column].get_text().get_text() for column in range(6)]
    assert ax_table.get_title() == "Variable analysis: Age"
    assert headers == ["Bin", "Min", "Max", "Clients", "Target", "Target_pct"]
    assert table[1, 1].get_text().get_text() == "18.46"
    assert table[1, 5].get_text().get_text() == "10"
    heights = [cast(Rectangle, patch).get_height() for patch in ax_graph.patches]
    assert heights == [10.0, 20.0, 30.0]
    assert [label.get_text() for label in ax_graph.get_xticklabels()] == [
        "1",
        "2",
        "3",
    ]
    assert ax_graph.get_ylabel() == settings.analysis.clients
    target_pct_values = cast(
        NDArray[np.float64],
        ax_target_pct.lines[0].get_ydata(),
    )
    np.testing.assert_allclose(
        target_pct_values,
        [10.0, 20.0, 30.0],
    )
    assert ax_target_pct.get_ylabel() == f"{settings.dataset.target} pct (%)"


def test_should_use_configured_columns_in_bivariate_figure() -> None:
    """Verifica que la tabla y los ejes respeten la configuración personalizada."""
    custom_settings = config.Settings(
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

    fig = visualization._build_bivariate_chart_figure(bivariate, custom_settings)

    ax_table, ax_graph, ax_target_pct = fig.axes
    table_artist = ax_table.tables[0]
    headers = [table_artist[0, column].get_text().get_text() for column in range(6)]
    assert headers == ["bucket", "Min", "Max", "customers", "label", "pct"]
    assert table_artist[1, 5].get_text().get_text() == "25"
    assert table_artist[2, 5].get_text().get_text() == "33"
    assert ax_graph.get_ylabel() == "customers"
    assert ax_target_pct.get_ylabel() == "label pct (%)"


@pytest.mark.parametrize(
    ("table_count", "max_workers"),
    [
        pytest.param(2, None, id="few-tables"),
        pytest.param(19, 4, id="fewer-than-twenty-tables"),
        pytest.param(20, 1, id="single-worker"),
    ],
)
def test_should_plot_bivariate_charts_sequentially_when_requested_or_few_tables(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    table_count: int,
    max_workers: int | None,
) -> None:
    """Verifica la ejecución secuencial y el orden de las tablas.

    ## Args:
        mocker: Utilidad para sustituir el dibujo individual.
        settings: Configuración del proyecto.
        table_count: Cantidad de tablas a procesar.
        max_workers: Cantidad de workers solicitada.
    """
    tables = [
        bivariate_table.BivariateTable(
            f"var_{index}",
            f"Var {index}",
            pl.DataFrame(),
        )
        for index in range(table_count)
    ]
    plot_mock = mocker.patch.object(visualization, "_plot_single_bivariate_chart")

    visualization.plot_bivariate_charts(
        tables,
        "analysis",
        settings,
        max_workers=max_workers,
    )

    assert [mock_call.args for mock_call in plot_mock.call_args_list] == [
        (table, "analysis", settings) for table in tables
    ]


def test_should_process_all_tables_with_process_pool_when_many_tables_are_requested(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica que todas las tablas se procesen mediante el pool de procesos.

    ## Args:
        mocker: Utilidad para sustituir el pool y el dibujo individual.
        settings: Configuración del proyecto.
    """
    tables = [
        bivariate_table.BivariateTable(
            f"var_{index}",
            f"Var {index}",
            pl.DataFrame(),
        )
        for index in range(20)
    ]
    plot_mock = mocker.patch.object(visualization, "_plot_single_bivariate_chart")
    executor_factory = mocker.patch.object(
        visualization.concurrent.futures,
        "ProcessPoolExecutor",
        side_effect=concurrent.futures.ThreadPoolExecutor,
    )

    visualization.plot_bivariate_charts(
        tables,
        "analysis",
        settings,
        max_workers=4,
    )

    executor_factory.assert_called_once_with(4)
    assert plot_mock.call_count == len(tables)
    plot_mock.assert_has_calls(
        [mocker.call(table, "analysis", settings) for table in tables],
        any_order=True,
    )


def _line_data(line) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Devuelve los datos de una línea con el tipo esperado por NumPy."""
    return (
        cast(NDArray[np.float64], line.get_xdata()),
        cast(NDArray[np.float64], line.get_ydata()),
    )


def test_should_build_roc_curve_figure_with_metrics_annotation(
    evaluation_metrics: evaluation.EvaluationMetrics,
) -> None:
    """Verifica la curva ROC, la diagonal y la anotación de métricas.

    ## Args:
        evaluation_metrics: Métricas de evaluación de ejemplo.
    """
    fig = visualization._build_roc_curve_figure(evaluation_metrics, "test_model")

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


def test_should_render_rounded_deciles_table_with_title() -> None:
    """Verifica el formato de los valores y el título de la tabla de deciles."""
    deciles = pl.DataFrame(
        {
            "decile": ["1", "2"],
            "count": [10, 20],
            "target_1_rate": [12.346, 67.891],
        }
    )
    _, ax = plt.subplots()

    visualization._plot_single_deciles_table(ax, deciles, "Train Deciles")

    table = ax.tables[0]
    headers = [table[0, column].get_text().get_text() for column in range(3)]
    assert headers == ["decile", "count", "target_1_rate"]
    assert table[1, 0].get_text().get_text() == "1"
    assert table[1, 2].get_text().get_text() == "12.35"
    assert table[2, 2].get_text().get_text() == "67.89"
    assert ax.get_title(loc="left") == "Train Deciles"
    assert not ax.axison


def test_should_build_train_and_test_deciles_in_single_figure(
    evaluation_metrics: evaluation.EvaluationMetrics,
) -> None:
    """Verifica las dos tablas del gráfico de deciles.

    ## Args:
        evaluation_metrics: Métricas con las tablas de deciles.
    """
    fig = visualization._build_deciles_figure(evaluation_metrics)

    assert len(fig.axes) == 2

    train_ax, test_ax = fig.axes
    assert train_ax.get_title(loc="left") == "Train Deciles"
    assert test_ax.get_title(loc="left") == "Test Deciles"

    train_table = train_ax.tables[0]
    test_table = test_ax.tables[0]
    train_headers = [
        train_table[0, column].get_text().get_text()
        for column in range(len(evaluation_metrics.train_deciles_without_probs.columns))
    ]
    test_headers = [
        test_table[0, column].get_text().get_text()
        for column in range(len(evaluation_metrics.test_deciles_without_probs.columns))
    ]
    assert train_headers == evaluation_metrics.train_deciles_without_probs.columns
    assert test_headers == evaluation_metrics.test_deciles_without_probs.columns
    assert "min_prob" not in train_headers
    assert "max_prob" not in test_headers
    assert train_table[1, 0].get_text().get_text() == "1"
    assert test_table[1, 0].get_text().get_text() == "1"


def _assert_saved_figure(
    save_mock,
    plot_name: str,
    images_sub_dir: str,
    settings: config.Settings,
) -> None:
    """Verifica los argumentos con los que se solicitó guardar una figura.

    ## Args:
        save_mock: Mock del guardado de figuras.
        plot_name: Nombre base esperado del archivo.
        images_sub_dir: Subdirectorio esperado de imágenes.
        settings: Configuración con el directorio base de imágenes.
    """
    assert save_mock.call_count == 1
    figure, actual_plot_name, images_dir, actual_sub_dir = save_mock.call_args.args
    assert isinstance(figure, Figure)
    assert actual_plot_name == plot_name
    assert images_dir == settings.paths.images
    assert actual_sub_dir == images_sub_dir


def test_should_save_top_features_figure_with_expected_name_and_directory(
    mocker: pytest_mock.MockerFixture,
    top_features_table: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica el nombre y el directorio usados al guardar el gráfico de importancia.

    ## Args:
        mocker: Utilidad para sustituir el guardado en disco.
        top_features_table: Tabla de importancias ordenada.
        settings: Configuración con el directorio de imágenes.
    """
    save_mock = mocker.patch.object(visualization, "_save_fig_as_svg")

    visualization.plot_top_features(
        top_features_table,
        "model_a",
        roc_auc=0.9,
        settings=settings,
    )

    _assert_saved_figure(save_mock, "model_a", "plot_top_features", settings)


def test_should_save_bivariate_figure_with_variable_name_and_analysis_directory(
    mocker: pytest_mock.MockerFixture,
    bivariate_age: bivariate_table.BivariateTable,
    settings: config.Settings,
) -> None:
    """Verifica el nombre y el subdirectorio usados al guardar el gráfico bivariado.

    ## Args:
        mocker: Utilidad para sustituir el guardado en disco.
        bivariate_age: Tabla bivariada de ejemplo.
        settings: Configuración con el directorio de imágenes.
    """
    save_mock = mocker.patch.object(visualization, "_save_fig_as_svg")

    visualization._plot_single_bivariate_chart(
        bivariate_age,
        "age_analysis",
        settings,
    )

    _assert_saved_figure(
        save_mock,
        "age",
        "bivariate_analysis/age_analysis",
        settings,
    )


def test_should_save_roc_curve_figure_with_expected_name_and_directory(
    mocker: pytest_mock.MockerFixture,
    evaluation_metrics: evaluation.EvaluationMetrics,
    settings: config.Settings,
) -> None:
    """Verifica el nombre y el directorio usados al guardar la curva ROC.

    ## Args:
        mocker: Utilidad para sustituir el guardado en disco.
        evaluation_metrics: Métricas de evaluación de ejemplo.
        settings: Configuración con el directorio de imágenes.
    """
    save_mock = mocker.patch.object(visualization, "_save_fig_as_svg")

    visualization.plot_roc_curve(evaluation_metrics, "test_model", settings)

    _assert_saved_figure(
        save_mock,
        "test_model",
        "plot_evaluation_metrics",
        settings,
    )


def test_should_save_deciles_figure_with_expected_name_and_directory(
    mocker: pytest_mock.MockerFixture,
    evaluation_metrics: evaluation.EvaluationMetrics,
    settings: config.Settings,
) -> None:
    """Verifica el nombre y el directorio usados al guardar el gráfico de deciles.

    ## Args:
        mocker: Utilidad para sustituir el guardado en disco.
        evaluation_metrics: Métricas con las tablas de deciles.
        settings: Configuración con el directorio de imágenes.
    """
    save_mock = mocker.patch.object(visualization, "_save_fig_as_svg")

    visualization.plot_deciles(evaluation_metrics, "deciles", settings)

    _assert_saved_figure(
        save_mock,
        "deciles",
        "plot_evaluation_metrics",
        settings,
    )
