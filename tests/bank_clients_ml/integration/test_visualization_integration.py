"""Tests de integración de la generación y escritura de gráficos SVG."""

import pathlib
import shutil

import numpy as np
import polars as pl
import pytest

from bank_clients_ml import (
    binning,
    bivariate_analysis,
    bivariate_table,
    config,
    evaluation,
    visualization,
)

pytestmark = pytest.mark.integration_svg

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SVGO_DIRECTORY = PROJECT_ROOT / "node_modules" / "svgo"


def _require_svg_tooling() -> None:
    """Omite el test cuando Bun o SVGO no están disponibles."""
    if shutil.which("bun") is None:
        pytest.skip("Bun no está disponible en PATH")
    if not SVGO_DIRECTORY.is_dir():
        pytest.skip("SVGO no está instalado en node_modules")


def test_should_write_top_features_bivariate_roc_and_deciles_svgs(
    integration_settings: config.Settings,
) -> None:
    """Verifica que los cuatro tipos de gráficos se guarden como SVG válidos.

    visualization.plot_bivariate_charts se ejecuta de forma secuencial.

    ## Args:
        integration_settings: Configuración con un directorio de imágenes temporal.
    """
    _require_svg_tooling()

    top_features = pl.DataFrame(
        {
            "Feature": ["Age", "Income", "Balance", "Tenure"],
            "Importance": [50.0, 30.0, 15.0, 5.0],
        }
    )
    bivariate_data = pl.DataFrame(
        {
            "Bin": pl.Series([1, 2, 3], dtype=pl.Int64),
            "Min": [18.0, 35.0, 60.0],
            "Max": [34.0, 59.0, 80.0],
            "Clients": pl.Series([10, 20, 30], dtype=pl.UInt32),
            "Target": pl.Series([1, 4, 9], dtype=pl.Int64),
            "Target_pct": [10.0, 20.0, 30.0],
        }
    )
    bivariate_age = bivariate_table.BivariateTable("age", "Age", bivariate_data)
    bivariate_income = bivariate_table.BivariateTable(
        "income",
        "Income",
        bivariate_data,
    )
    train_deciles = pl.DataFrame(
        {
            "decile": ["2", "1"],
            "count": [10, 20],
            "target_1_count": [7, 3],
            "min_prob": [30.0, 10.0],
            "max_prob": [40.0, 20.0],
            "target_1_rate": [70.0, 15.0],
            "cum_gain": [70.0, 100.0],
            "lift": [1.4, 0.6],
            "ks": [20.0, 0.0],
        }
    )
    test_deciles = train_deciles.reverse()
    metrics = evaluation.EvaluationMetrics(
        roc_auc=0.876543,
        accuracy=0.765432,
        fpr=np.array([0.0, 0.2, 1.0]),
        tpr=np.array([0.0, 0.8, 1.0]),
        train_deciles=train_deciles,
        test_deciles=test_deciles,
    )

    visualization.plot_top_features(
        top_features,
        "top_features",
        roc_auc=0.9,
        top_n=3,
        settings=integration_settings,
    )
    visualization.plot_bivariate_charts(
        [bivariate_age, bivariate_income],
        "integration_analysis",
        integration_settings,
        max_workers=None,
    )
    visualization.plot_roc_curve(metrics, "roc_curve", integration_settings)
    visualization.plot_deciles(metrics, "deciles", integration_settings)

    svg_paths = (
        integration_settings.paths.images / "plot_top_features" / "top_features.svg",
        integration_settings.paths.images
        / "bivariate_analysis"
        / "integration_analysis"
        / "age.svg",
        integration_settings.paths.images
        / "bivariate_analysis"
        / "integration_analysis"
        / "income.svg",
        integration_settings.paths.images / "plot_evaluation_metrics" / "roc_curve.svg",
        integration_settings.paths.images / "plot_evaluation_metrics" / "deciles.svg",
    )
    for svg_path in svg_paths:
        assert svg_path.is_file()
        content = svg_path.read_text(encoding="utf-8")
        assert len(content) > 100
        assert content.rstrip().endswith("</svg>")


@pytest.mark.integration_lgbm
@pytest.mark.integration_svg
def test_should_write_bivariate_top_features_and_evaluation_svgs(
    synthetic_train: pl.DataFrame,
    integration_settings: config.Settings,
) -> None:
    """Verifica los gráficos de una selección y un analizador bivariado reales.

    ## Args:
        synthetic_train: Conjunto sintético con variables y target.
        integration_settings: Configuración con un directorio de imágenes temporal.
    """
    _require_svg_tooling()

    feature_columns = [
        column for column in synthetic_train.columns if column.startswith("Feature_")
    ][:3]
    assert feature_columns

    columns = [integration_settings.dataset.target, *feature_columns]
    train = synthetic_train.head(1_500).select(columns)
    test = synthetic_train.tail(500).select(columns)
    name_mapping = binning.NameMapping.from_dict(
        {column: column.replace("_", " ") for column in feature_columns}
    )

    selection = bivariate_analysis.FeaturesSelection.fit_and_build(
        train=train,
        test=test,
        name_mapping=name_mapping,
        settings=integration_settings,
        n_iter=1,
    )
    analyzer = bivariate_analysis.BivariateAnalyzer(
        correlated_train=train,
        correlated_test=test,
        settings=integration_settings,
    )

    selection.plot_bivariate_analysis(
        "integration_selection",
        max_bins_quantity=5,
    )
    selection.plot_top_features("integration_selection_top_features")
    selection.evaluate_and_plot_metrics(
        roc_curve_name="integration_selection_roc",
        deciles_name="integration_selection_deciles",
    )
    analyzer.plot_bivariate(
        name_mapping,
        "integration_correlated",
        max_bins_quantity=5,
    )

    expected_bivariate_paths = [
        integration_settings.paths.images
        / "bivariate_analysis"
        / analysis_name
        / f"{column}.svg"
        for analysis_name in ("integration_selection", "integration_correlated")
        for column in feature_columns
    ]
    expected_svg_paths = [
        *expected_bivariate_paths,
        (
            integration_settings.paths.images
            / "plot_top_features"
            / "integration_selection_top_features.svg"
        ),
        (
            integration_settings.paths.images
            / "plot_evaluation_metrics"
            / "integration_selection_roc.svg"
        ),
        (
            integration_settings.paths.images
            / "plot_evaluation_metrics"
            / "integration_selection_deciles.svg"
        ),
    ]
    for svg_path in expected_svg_paths:
        assert svg_path.is_file()
        content = svg_path.read_text(encoding="utf-8")
        assert len(content) > 100
        assert content.rstrip().endswith("</svg>")
