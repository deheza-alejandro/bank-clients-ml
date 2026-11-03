"""Tests unitarios para el análisis bivariado y la selección de variables."""

from typing import cast

import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock

from bank_clients_ml import (
    binning,
    bivariate_analysis,
    bivariate_table,
    config,
    evaluation,
    training,
    visualization,
)


@pytest.fixture
def name_mapping() -> binning.NameMapping:
    """Devuelve un mapeo con dos variables renombradas.

    ## Returns:
        Mapeo entre los nombres internos y los nombres descriptivos.
    """
    return binning.NameMapping.from_dict({"age": "Age", "income": "Income"})


@pytest.fixture
def train_data() -> pl.DataFrame:
    """Devuelve un conjunto de entrenamiento con dos variables numéricas.

    ## Returns:
        DataFrame con identificador, target y dos features.
    """
    return pl.DataFrame(
        {
            "client_id": ["train_1", "train_2", "train_3", "train_4"],
            "Target": [0, 1, 0, 1],
            "age": [10.0, 20.0, 30.0, 40.0],
            "income": [100.0, 200.0, 300.0, 400.0],
        }
    )


@pytest.fixture
def test_data() -> pl.DataFrame:
    """Devuelve un conjunto de prueba con el mismo esquema que el de entrenamiento.

    ## Returns:
        DataFrame con dos observaciones.
    """
    return pl.DataFrame(
        {
            "client_id": ["test_1", "test_2"],
            "Target": [0, 1],
            "age": [15.0, 35.0],
            "income": [150.0, 350.0],
        }
    )


@pytest.fixture
def trainer_mock(mocker: pytest_mock.MockerFixture) -> pytest_mock.MockType:
    """Devuelve un entrenador simulado para verificar sus interacciones.

    Permite comprobar las llamadas realizadas por `FeaturesSelection` y
    configurar el modelo retornado por la propiedad `best_model`.

    ## Args:
        mocker: Utilidad de `pytest-mock` para crear el entrenador simulado.

    ## Returns:
        Mock que reemplaza al `LGBMTrainer` durante los tests.
    """
    return mocker.Mock()


@pytest.fixture
def features_selection(
    trainer_mock: pytest_mock.MockType,
    settings: config.Settings,
    name_mapping: binning.NameMapping,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
) -> bivariate_analysis.FeaturesSelection:
    """Devuelve una selección con un entrenador simulado.

    ## Args:
        mocker: Utilidad para construir el entrenador simulado.
        settings: Configuración global del proyecto.
        name_mapping: Mapeo entre nombres internos y descriptivos.
        train_data: Conjunto de entrenamiento.
        test_data: Conjunto de prueba.

    ## Returns:
        Selección lista para invocar sus métodos de visualización y evaluación.
    """
    return bivariate_analysis.FeaturesSelection(
        train=train_data,
        test=test_data,
        name_mapping=name_mapping,
        trainer=cast(training.LGBMTrainer, trainer_mock),
        settings=settings,
        n_iter=5,
    )


@pytest.fixture
def bivariate_analyzer(
    settings: config.Settings,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
) -> bivariate_analysis.BivariateAnalyzer:
    """Devuelve un analizador con los conjuntos previos al filtrado.

    ## Args:
        settings: Configuración global del proyecto.
        train_data: Conjunto de entrenamiento.
        test_data: Conjunto de prueba.

    ## Returns:
        Analizador bivariado listo para ser utilizado.
    """
    return bivariate_analysis.BivariateAnalyzer(
        correlated_train=train_data,
        correlated_test=test_data,
        settings=settings,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_n_iter"),
    [
        pytest.param({"n_iter": 7}, 7, id="explicit-iterations"),
        pytest.param({}, 20, id="default-iterations"),
    ],
)
def test_should_fit_trainer_with_mapped_features_when_building_features_selection(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    name_mapping: binning.NameMapping,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    kwargs: dict[str, int],
    expected_n_iter: int,
) -> None:
    """Verifica que la selección entrene al modelo con las variables mapeadas.

    ## Args:
        mocker: Utilidad para sustituir el entrenamiento de LightGBM.
        settings: Configuración global del proyecto.
        name_mapping: Mapeo entre nombres internos y descriptivos.
        train_data: Conjunto de entrenamiento.
        test_data: Conjunto de prueba.
        kwargs: Argumentos opcionales enviados a `fit_and_build`.
        expected_n_iter: Cantidad esperada de iteraciones de la búsqueda.
    """
    trainer = mocker.Mock()
    fit = mocker.patch.object(
        training.LGBMTrainer,
        "fit_and_build",
        return_value=trainer,
    )

    result = bivariate_analysis.FeaturesSelection.fit_and_build(
        train=train_data,
        test=test_data,
        name_mapping=name_mapping,
        settings=settings,
        **kwargs,
    )

    fit.assert_called_once_with(
        train_data,
        ("age", "income"),
        settings,
        n_iter=expected_n_iter,
    )
    assert result.train is train_data
    assert result.test is test_data
    assert result.name_mapping is name_mapping
    assert result.trainer is trainer
    assert result.settings is settings
    assert result.n_iter == expected_n_iter
    assert result.snake_case_names == ("age", "income")


def test_should_build_train_tables_and_plot_charts_with_requested_bin_limit(
    mocker: pytest_mock.MockerFixture,
    features_selection: bivariate_analysis.FeaturesSelection,
) -> None:
    """Verifica la construcción de tablas y el trazado de los gráficos bivariados.

    ## Args:
        mocker: Utilidad para sustituir la construcción de tablas y el trazado.
        features_selection: Selección con los conjuntos ya transformados.
    """
    tables = (mocker.Mock(),)
    build = mocker.patch.object(
        bivariate_table,
        "build_bivariate_tables",
        return_value=tables,
    )
    plot = mocker.patch.object(visualization, "plot_bivariate_charts")

    features_selection.plot_bivariate_analysis(
        "best_features",
        max_bins_quantity=6,
    )

    build.assert_called_once_with(
        features_selection.train,
        features_selection.name_mapping,
        6,
        settings=features_selection.settings,
    )
    plot.assert_called_once_with(
        tables,
        "best_features",
        settings=features_selection.settings,
    )


def test_should_delegate_top_features_plot_to_trainer_with_name_mapping(
    features_selection: bivariate_analysis.FeaturesSelection,
    trainer_mock: pytest_mock.MockType,
) -> None:
    """Verifica que el gráfico de importancias se delegue al entrenador.

    ## Args:
        features_selection: Selección con un entrenador simulado.
    """
    features_selection.plot_top_features("best_features")

    trainer_mock.plot_top_features.assert_called_once_with(
        "best_features",
        features_selection.name_mapping,
    )


def test_should_evaluate_best_model_on_selected_features_and_plot_metrics(
    mocker: pytest_mock.MockerFixture,
    features_selection: bivariate_analysis.FeaturesSelection,
    trainer_mock: pytest_mock.MockType,
) -> None:
    """Verifica la evaluación del modelo y la generación de sus gráficos.

    ## Args:
        mocker: Utilidad para sustituir la evaluación y el trazado de gráficos.
        features_selection: Selección con los conjuntos y el mapeo de nombres.
    """
    model = mocker.Mock()
    model.predict_proba.side_effect = [
        np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.4, 0.6],
            ]
        ),
        np.array([[0.6, 0.4], [0.2, 0.8]]),
    ]
    trainer_mock.best_model = model

    metrics = mocker.Mock()
    evaluate = mocker.patch.object(evaluation, "evaluate", return_value=metrics)
    plot_roc = mocker.patch.object(visualization, "plot_roc_curve")
    plot_deciles = mocker.patch.object(visualization, "plot_deciles")

    features_selection.evaluate_and_plot_metrics(
        roc_curve_name="roc_curve",
        deciles_name="deciles",
    )

    assert model.predict_proba.call_count == 2
    polars.testing.assert_frame_equal(
        model.predict_proba.call_args_list[0].args[0],
        features_selection.train.select(["age", "income"]),
    )
    polars.testing.assert_frame_equal(
        model.predict_proba.call_args_list[1].args[0],
        features_selection.test.select(["age", "income"]),
    )
    evaluate.assert_called_once()
    assert evaluate.call_args.kwargs["train"] is features_selection.train
    assert evaluate.call_args.kwargs["test"] is features_selection.test
    assert evaluate.call_args.kwargs["settings"] is features_selection.settings
    np.testing.assert_array_equal(
        evaluate.call_args.kwargs["probabilities_train"],
        [0.2, 0.7, 0.1, 0.6],
    )
    np.testing.assert_array_equal(
        evaluate.call_args.kwargs["probabilities_test"],
        [0.4, 0.8],
    )
    plot_roc.assert_called_once_with(
        metrics,
        "roc_curve",
        features_selection.settings,
    )
    plot_deciles.assert_called_once_with(
        metrics,
        "deciles",
        settings=features_selection.settings,
    )


def test_should_build_analyzer_from_correlated_filter_sets(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
) -> None:
    """Verifica que el analizador reutilice los conjuntos del filtro de columnas.

    ## Args:
        mocker: Utilidad para construir el filtro simulado.
        settings: Configuración global del proyecto.
        train_data: Conjunto de entrenamiento previo al filtrado.
        test_data: Conjunto de prueba previo al filtrado.
    """
    column_filter = mocker.Mock()
    column_filter.correlated_train = train_data
    column_filter.correlated_test = test_data
    column_filter.settings = settings

    result = bivariate_analysis.BivariateAnalyzer.from_redundant_column_filter(
        column_filter
    )

    assert result.correlated_train is train_data
    assert result.correlated_test is test_data
    assert result.settings is settings


def test_should_plot_bivariate_charts_from_correlated_train_tables(
    mocker: pytest_mock.MockerFixture,
    bivariate_analyzer: bivariate_analysis.BivariateAnalyzer,
    name_mapping: binning.NameMapping,
) -> None:
    """Verifica que los gráficos se construyan con el conjunto previo al filtrado.

    ## Args:
        mocker: Utilidad para sustituir la construcción de tablas y el trazado.
        bivariate_analyzer: Analizador con los conjuntos correlacionados.
        name_mapping: Mapeo entre nombres internos y descriptivos.
    """
    tables = (mocker.Mock(),)
    build = mocker.patch.object(
        bivariate_table,
        "build_bivariate_tables",
        return_value=tables,
    )
    plot = mocker.patch.object(visualization, "plot_bivariate_charts")

    bivariate_analyzer.plot_bivariate(
        name_mapping,
        "correlated",
        max_bins_quantity=8,
    )

    build.assert_called_once_with(
        bivariate_analyzer.correlated_train,
        name_mapping,
        8,
        settings=bivariate_analyzer.settings,
    )
    plot.assert_called_once_with(
        tables,
        "correlated",
        settings=bivariate_analyzer.settings,
    )


def test_should_apply_binning_to_both_sets_and_keep_only_final_columns(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica la codificación por bines y la selección de columnas finales.

    ## Args:
        mocker: Utilidad para sustituir el entrenamiento de LightGBM.
        settings: Configuración global del proyecto.
    """
    train = pl.DataFrame(
        {
            "client_id": [
                "train_1",
                "train_2",
                "train_3",
                "train_4",
                "train_5",
                "train_6",
            ],
            "Target": [0, 1, 0, 1, 1, 1],
            "age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "income": [100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
            "unused": [1, 2, 3, 4, 5, 6],
        }
    )
    test = pl.DataFrame(
        {
            "client_id": ["test_1", "test_2"],
            "Target": [0, 1],
            "age": [15.0, 45.0],
            "income": [150.0, 300.0],
            "unused": [7, 8],
        }
    )
    expected_train = pl.DataFrame(
        {
            "client_id": [
                "train_1",
                "train_2",
                "train_3",
                "train_4",
                "train_5",
                "train_6",
            ],
            "Target": [0, 1, 0, 1, 1, 1],
            "age": [50.0, 50.0, 75.0, 75.0, 75.0, 75.0],
            "income": [50.0, 50.0, 50.0, 50.0, 100.0, 100.0],
        }
    )
    expected_test = pl.DataFrame(
        {
            "client_id": ["test_1", "test_2"],
            "Target": [0, 1],
            "age": [50.0, 75.0],
            "income": [50.0, 100.0],
        }
    )
    binners = [
        binning.Binner("age", "Age").merge_bins(1, 2).merge_bins(3, 6),
        binning.Binner("income", "Income").merge_bins(1, 2).merge_bins(3),
    ]
    analyzer = bivariate_analysis.BivariateAnalyzer(
        correlated_train=train,
        correlated_test=test,
        settings=settings,
    )
    trainer = mocker.Mock()
    fit = mocker.patch.object(
        training.LGBMTrainer,
        "fit_and_build",
        return_value=trainer,
    )

    result = analyzer.apply_binning(binners)

    polars.testing.assert_frame_equal(result.train, expected_train)
    polars.testing.assert_frame_equal(result.test, expected_test)
    assert result.trainer is trainer
    assert result.settings is settings
    assert result.n_iter == 20
    assert result.name_mapping.snake_case_names == ("age", "income")
    assert result.name_mapping.pretty_names == ("Age", "Income")
    fit.assert_called_once()
    fitted_train, fitted_columns, fitted_settings = fit.call_args.args
    assert fitted_columns == ("age", "income")
    assert fitted_settings is settings
    assert fit.call_args.kwargs == {"n_iter": 20}
    polars.testing.assert_frame_equal(fitted_train, expected_train)
